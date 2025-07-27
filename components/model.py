import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from components import helpers

config = helpers.load_config("./config/train_config.json")
device = torch.device(config["device"])

class DuelingDQN(nn.Module):
    """Enhanced Dueling DQN with attention mechanism"""
    def __init__(self, input_size, hidden_size=128, output_size=2, num_layers=2):
        super(DuelingDQN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=config["dropout_rate"],
            bidirectional=False
        )

        # Layer normalization
        self.ln = nn.LayerNorm(hidden_size)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=config["attention_heads"], batch_first=True)

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln(lstm_out)

        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last time step
        final_hidden = attended_out[:, -1, :]

        # Dueling streams
        value = self.value_stream(final_hidden)
        advantage = self.advantage_stream(final_hidden)

        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size
        }, file_path)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class AdvancedQTrainer:
    """Enhanced Q-learning trainer with modern techniques"""
    def __init__(self, model, lr=config["LR"], gamma=config["GAMMA"], tau=config["TAU"]):
        self.model = model
        self.target_model = DuelingDQN(model.input_size, model.hidden_size).to(device)
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        self.gamma = gamma
        self.tau = tau
        self.criterion = nn.HuberLoss()  # More robust than MSE

        self.training_steps = 0
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_decay = 1000

    def get_beta(self):
        """Annealing beta for importance sampling"""
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.training_steps / self.beta_decay)
        return beta

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None

        # Sample from prioritized replay buffer
        experiences, indices, weights = replay_buffer.sample(batch_size, self.get_beta())

        states = torch.stack([torch.tensor(e.state, dtype=torch.float32) for e in experiences]).to(device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(device)
        next_states = torch.stack([torch.tensor(e.next_state, dtype=torch.float32) for e in experiences]).to(device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))

        # Calculate loss with importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * (td_errors ** 2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        priorities = (torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6)
        replay_buffer.update_priorities(indices, priorities)

        # Soft update target network
        self._soft_update()

        self.training_steps += 1

        return loss.item()

    def _soft_update(self):
        """Soft update target network"""
        for target_param, main_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
