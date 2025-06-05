# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """Policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.act_conv1 = nn.Conv1d(128, 80, kernel_size=3, padding=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        self.val_conv1 = nn.Conv1d(128, 20, kernel_size=3, padding=1)
        self.val_fc1 = nn.Linear(board_width * board_height, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.board_width * self.board_height)
        x_val = torch.tanh(self.val_fc2(F.relu(self.val_fc1(x_val))))
        
        return x_act, x_val


class PolicyValueNet:
    """Policy-value network"""
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        
        self.policy_value_net = Net(board_width, board_height).cuda() if self.use_gpu else Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        state_batch_np = np.array(state_batch)
        state_batch = Variable(torch.FloatTensor(state_batch_np).cuda() if self.use_gpu else torch.FloatTensor(state_batch))
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy() if self.use_gpu else log_act_probs.data.numpy())
        return act_probs, value.data.cpu().numpy() if self.use_gpu else value.data.numpy()

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.expand_dims(board.current_state(), axis=0)
        current_state = np.ascontiguousarray(current_state)

        log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state).cuda().float() if self.use_gpu else torch.from_numpy(current_state).float()))
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten() if self.use_gpu else log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda() if self.use_gpu else torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda() if self.use_gpu else torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)).cuda() if self.use_gpu else torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)