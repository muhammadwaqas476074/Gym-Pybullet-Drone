import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BasePolicy
from gym_pybullet_drones.Florance.florance_modal import NCPModel


class NCPPolicy(ActorCriticPolicy, BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(NCPPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        drone_state_dim = observation_space["drone_state"].shape[0]
        text_embedding_dim = observation_space["text_embedding"].shape[0]
        motor_output_dim = action_space.shape[0]  # Get action space dimension
        self.ncp_model = NCPModel(drone_state_dim, text_embedding_dim, motor_output_dim).to(self.device)

    def forward(self, obs, deterministic=False):
        drone_state = torch.as_tensor(obs["drone_state"], dtype=torch.float32, device=self.device)
        text_embedding = torch.as_tensor(obs["text_embedding"], dtype=torch.long, device=self.device)  # Text Embedding is now Long
        actions = self.ncp_model(drone_state, text_embedding)

        return actions, None  # No Value Function for now
