import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM, BertModel, BertTokenizer
from PIL import Image
import requests
import copy
import torch
from torch import nn
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.nn as nn
import ncps
from ncps import wirings
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import pytorch_lightning as pl
import torch
import torch.utils.data as data



class NCPModel(nn.Module):
    def __init__(self, seed=22222, input_dim=519936, model_id = 'Florance2-base', rnn_name = "ltccell_weight.pt", DEVICE = "cpu"):
        super(NCPModel, self).__init__()
        self.DEVICE = DEVICE
        self.input_dim = input_dim
        self.input_ids = None
        self.attention_mask = None

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').to(self.DEVICE)
        for param in self.model.parameters():
          param.requires_grad = False
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

        model_path = f"{model_id}/{rnn_name}"
        if os.path.isfile(model_path):
          checkpoint = torch.load(f"{model_path}", weights_only=True)
          self.input_dim = checkpoint['input_dim']

        self.wiring.set_input_dim(self.input_dim)
        self.wiring.build(self.input_dim)
        self.rnn_cell = ncps.torch.LTCCell(wiring=self.wiring).to(self.DEVICE)

        if os.path.isfile(model_path):
          print(f"Loading model from: {model_path}")
          self.rnn_cell.load_state_dict(checkpoint['state'])
          self.current_state = checkpoint['next_state']
        else:
          self.current_state =  torch.zeros(1, self.rnn_cell.state_size).to(self.DEVICE)
          print(f"RNN file not found. Initializing New")


    def forward(self, pixel_values, input_ids= None, attention_mask = None):
        forward_start = time.time()
        if input_ids is not None and attention_mask is not None:
            self.input_ids = input_ids
            self.attention_mask = attention_mask
        elif self.input_ids is None and self.attention_mask is None:
            raise ValueError("Both input_ids and attention_mask cannot be None. Atleast provide Once")

        start = time.time()
        with torch.no_grad():
          encoded_output = self.model(input_ids = self.input_ids, pixel_values = pixel_values, attention_mask = self.attention_mask)
        end = time.time()
        print(f"Time taken for image encoding : {end - start} seconds")

        start = time.time()
        last_hidden_state = encoded_output["last_hidden_state"]
        flattened_output = last_hidden_state.view(last_hidden_state.shape[0], -1)
        motor_output, next_state = self.rnn_cell(flattened_output, self.current_state)

        self.current_state = next_state
        end = time.time()
        print(f"Time taken for Liquid network: {end - start} seconds")
        print(f"Total time taken by network: {end - forward_start} seconds")
        return {"motor_output" : motor_output, "next_state" : next_state, "encoded_output" : encoded_output}

    def text_length_change(self, input_ids, attention_mask, pixel_values):

        """
        Default = 100 words
        Use this function if you change the length of text string or pixal_values. Increase or decrease.
        Process the new string and Image size and pass its input_ids, attention_mask, pixel_values.

        Note that changing length will result in initializing new model so remember to save its weights (State).
        """

        encoded_output = self.model(input_ids = input_ids, pixel_values = pixel_values, attention_mask = attention_mask)
        last_hidden_state = encoded_output["last_hidden_state"]
        flattened_output = torch.flatten(last_hidden_state)
        self.input_dim = flattened_output.size(0)

        self.wiring.set_input_dim(self.input_dim)
        self.wiring.build(self.input_dim)
        self.rnn_cell = ncps.torch.LTCCell(wiring=self.wiring).to(self.DEVICE)

    def new_sequence(self, current_state=None):

        """ Initialize the memory state with zeros in case of New Sequence
        Also initialize memory state from given current state which will be used as new state for next iteration
        """

        if current_state is None:
          self.current_state = torch.zeros(1, self.rnn_cell.state_size).to(self.DEVICE)
        else:
          self.current_state = current_state

    def save_model(self, path):
        """
        Saves the model state dictionary and other relevant information to a checkpoint file.
        """
        torch.save({'state': self.rnn_cell.state_dict(), 'next_state': self.current_state, 'input_dim': self.input_dim}, f"{path}")
        print(f"Model saved at {path}")

    def load_model(self, path):
        checkpoint = torch.load(f"{path}")
        self.input_dim = checkpoint['input_dim']

        """ initialize the Skeleton for rnn cell"""
        self.wiring.set_input_dim(self.input_dim)
        self.wiring.build(self.input_dim)
        self.rnn_cell = ncps.torch.LTCCell(wiring=self.wiring).to(self.DEVICE)

        """ Loading weights (State) into rnn_cell"""
        self.rnn_cell.load_state_dict(checkpoint['state'])
        self.current_state = checkpoint['next_state']
        print(f"Model loaded from {path}")


