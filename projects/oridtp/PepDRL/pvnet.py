# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """Convolutional policy-value network over a (vocab x length) sequence grid."""

    def __init__(self, seq_width, vocab_height):
        super(Net, self).__init__()
        self.seq_width = seq_width
        self.vocab_height = vocab_height
        flat_dim = seq_width * vocab_height

        trunk_channels = (20, 32, 64, 128)
        trunk_layers = []
        for in_c, out_c in zip(trunk_channels[:-1], trunk_channels[1:]):
            trunk_layers.append(nn.Conv1d(in_c, out_c, kernel_size=3, padding=1))
            trunk_layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*trunk_layers)

        self.policy_conv = nn.Conv1d(128, 80, kernel_size=3, padding=1)
        self.policy_head = nn.Linear(4 * flat_dim, flat_dim)

        self.value_conv = nn.Conv1d(128, 20, kernel_size=3, padding=1)
        self.value_hidden = nn.Linear(flat_dim, 128)
        self.value_out = nn.Linear(128, 1)

    def forward(self, x):
        features = self.trunk(x)

        policy = F.relu(self.policy_conv(features))
        policy = policy.reshape(policy.size(0), -1)
        policy = F.log_softmax(self.policy_head(policy), dim=1)

        value = F.relu(self.value_conv(features))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_hidden(value))
        value = torch.tanh(self.value_out(value))

        return policy, value


class PolicyValueNet:
    """Wraps :class:`Net` with train/inference helpers used by the MCTS search."""

    def __init__(self, width, height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.width = width
        self.height = height
        self.weight_decay = 1e-4
        self.device = torch.device("cuda" if use_gpu else "cpu")

        self.net = Net(width, height).to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), weight_decay=self.weight_decay
        )

        if model_file:
            self.net.load_state_dict(torch.load(model_file))

    def _to_tensor(self, array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def policy_value(self, state_batch):
        batch = self._to_tensor(state_batch)
        with torch.no_grad():
            log_probs, value = self.net(batch)
        probs = np.exp(log_probs.detach().cpu().numpy())
        return probs, value.detach().cpu().numpy()

    def policy_value_fn(self, board):
        legal_positions = board.availables
        state = np.ascontiguousarray(
            np.expand_dims(board.current_state(), axis=0)
        )
        board_tensor = self._to_tensor(state)
        with torch.no_grad():
            log_probs, value = self.net(board_tensor)
        probs = np.exp(log_probs.detach().cpu().numpy().flatten())
        legal_probs = zip(legal_positions, probs[legal_positions])
        return legal_probs, value.detach().cpu().numpy()[0][0]

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        states = self._to_tensor(state_batch)
        target_probs = self._to_tensor(mcts_probs)
        target_value = self._to_tensor(winner_batch)

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()

        log_probs, value = self.net(states)
        value_term = F.mse_loss(value.view(-1), target_value)
        policy_term = -torch.mean(torch.sum(target_probs * log_probs, dim=1))
        loss = value_term + policy_term
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_probs) * log_probs, dim=1)
            )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.net.state_dict()

    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)
