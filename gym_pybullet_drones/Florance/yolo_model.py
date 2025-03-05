import torch.nn as nn
from transformers import BertModel, BertTokenizer
from PIL import Image
import torch
from torch import nn
import time
import numpy as np
import torch.nn.functional as F
import os
import numpy as np
import torch.nn as nn
import ncps
from ncps import wirings
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import torch
from ultralytics import YOLO


#Note that loading and saving of model is not yet defined
class DroneControlSystem(nn.Module):
    def __init__(self, base_model = "yolo11s-seg.pt", yolo_layer = 16, yolo_output = 614400, bert_model_name='E:/ml/bert-base-uncased/', text = None, seed = 22222, DEVICE = 'cpu'):
        super(DroneControlSystem, self).__init__()
        self.DEVICE = DEVICE

        if yolo_output is None:
            raise ValueError("Error: `yolo_output` is required.")

        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.DEVICE)  # BERT model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.yolo = YOLO(base_model)   # same as YOLO("yolo11n.pt")
        for param in self.yolo.parameters():
            param.requires_grad = True  # yolo trainable

        for param in self.bert_model.parameters():
            param.requires_grad = False  # Freeze BERT

        # Text encoding using BERT (done without updating BERT)
        if not text:
            text = "Maintain your position at 1 meter height"
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.DEVICE)
        with torch.no_grad():
            self.bert_output = self.bert_model(**inputs).last_hidden_state.mean(dim=1)

        self.modified_output_from_hook = None

        self.wiring = wirings.NCP(
            inter_neurons=18,
            command_neurons=12,
            motor_neurons=4,
            sensory_fanout=6,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=6,
            seed=seed,
        )

        self.c3k2_module = self.yolo.model.model[yolo_layer]
        self.hook_handle = self.c3k2_module.register_forward_hook(self.hook_function)

        bert_size = 768
        self.input_size =  yolo_output + bert_size

        self.wiring.set_input_dim(self.input_size)
        self.wiring.build(self.input_size)
        self.rnn_cell = ncps.torch.LTCCell(wiring=self.wiring).to(self.DEVICE)

        self.current_state = torch.zeros(1, self.rnn_cell.state_size).to(self.DEVICE)

    def forward(self, image, text=None, new_sequence=False):
        if new_sequence:
            self.current_state = torch.zeros(1, self.rnn_cell.state_size).to(self.DEVICE)

        # Perform forward pass with no gradient computation for YOLO and BERT
        with torch.no_grad():
            start = time.time()
            yolo_output = self.yolo(image)  # YOLO forward pass
            end = time.time()
            print(f"Time taken for image encoding : {end - start} seconds")

        if text is not None:
            start = time.time()
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.DEVICE)
            with torch.no_grad():
                self.bert_output = self.bert_model(**inputs).last_hidden_state.mean(dim=1)  # BERT output
            end = time.time()
            print(f"Time taken for text encoding : {end - start} seconds")

        start = time.time()
        flattened_output = self.modified_output_from_hook.view(self.modified_output_from_hook.shape[0], -1).clone()

        #combined_output = torch.cat((flattened_output, self.bert_output), dim=1)
        # Duplicate the bert_output to match the batch size of flattened_output
        bert_output_expanded = self.bert_output.expand(flattened_output.shape[0], -1)
        combined_output = torch.cat((flattened_output, bert_output_expanded), dim=1)

        motor_output, next_state = self.rnn_cell(combined_output, self.current_state)
        self.current_state = next_state
        end = time.time()

        print(f"Time taken for Liquid network: {end - start} seconds")
        return {"motor_output": motor_output, "next_state": next_state, "visual_embed": self.modified_output_from_hook}

    # Hook function to capture the output of C3k2 Mid
    def hook_function(self, module, input, output):
        #print(f"C3k2 for layer {self.yolo_layer} output shape: {output.shape}")
        self.modified_output_from_hook = output  # Store the output from the hook
        return output

    # Mind do not use this function NEVER in good sense
    def remove_hook(self):
        self.hook_handle.remove()
