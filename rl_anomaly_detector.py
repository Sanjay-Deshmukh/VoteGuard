"""
Deep Q-Network (DQN) based Reinforcement Learning Anomaly Detector
Implements online learning that adapts to new tampering patterns in real-time
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class DQN(nn.Module):
    """Deep Q-Network for anomaly detection"""
    def __init__(self, state_size=10, action_size=2, hidden_size=64):
        """
        Args:
            state_size: Number of features (total_votes, vote_diff_sum, diff_1-4, ratio_1-4)
            action_size: 2 actions (0=Normal, 1=Anomaly)
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class RLAnomalyDetector:
    """RL-based anomaly detector with online learning"""
    def __init__(self, state_size=10, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=1000, batch_size=32, model_path="rl_dqn_model.pth"):
        self.state_size = state_size
        self.action_size = 2  # Normal (0) or Anomaly (1)
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model_path = model_path
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, self.action_size).to(self.device)
        self.target_network = DQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Tracking
        self.rewards_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.prediction_history = []
        self.accuracy_history = []
        self.step_count = 0
        
        # Load model if exists
        self.load_model()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_state(self, features):
        """Convert features to state vector"""
        if isinstance(features, (list, np.ndarray)):
            state = np.array(features, dtype=np.float32)
        else:
            # If it's a pandas Series or dict, extract values
            state = np.array(list(features.values()) if hasattr(features, 'values') else list(features), dtype=np.float32)
        
        # Normalize state
        state = (state - state.mean()) / (state.std() + 1e-8)
        return state
    
    def predict(self, state, use_epsilon=False):
        """Predict action (0=Normal, 1=Anomaly)"""
        if use_epsilon and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 1)
        
        # Exploitation: use Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def compute_reward(self, action, ensemble_prediction, ensemble_score):
        """
        Compute reward based on action and ensemble model consensus
        Args:
            action: RL action (0=Normal, 1=Anomaly)
            ensemble_prediction: Ensemble prediction (0=Normal, 1=Anomaly)
            ensemble_score: Confidence score from ensemble
        """
        # Base reward: +1 if correct, -1 if wrong
        if action == ensemble_prediction:
            reward = 1.0
        else:
            reward = -1.0
        
        # Bonus for high confidence predictions
        if action == ensemble_prediction:
            reward += ensemble_score * 0.5
        
        # Penalty for low confidence when wrong
        if action != ensemble_prediction:
            reward -= (1 - ensemble_score) * 0.5
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Train the DQN on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (self.gamma * next_q_value * ~dones)
        
        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        if self.step_count % 10 == 0:
            self.update_target_network()
        
        self.step_count += 1
        
        return loss.item()
    
    def learn_online(self, state, ensemble_prediction, ensemble_score):
        """
        Online learning: predict, get reward, and learn
        Returns: (prediction, reward, loss)
        """
        # Get prediction
        action = self.predict(state, use_epsilon=True)
        
        # Compute reward based on ensemble consensus
        reward = self.compute_reward(action, ensemble_prediction, ensemble_score)
        
        # For next state, use current state (since we're in online learning)
        next_state = state.copy() if isinstance(state, np.ndarray) else state
        
        # Store experience
        done = False  # Continuous learning
        self.remember(state, action, reward, next_state, done)
        
        # Train
        loss = self.train_step()
        
        # Track history
        self.rewards_history.append(reward)
        if loss is not None:
            self.loss_history.append(loss)
        self.epsilon_history.append(self.epsilon)
        self.prediction_history.append(action)
        
        return action, reward, loss
    
    def save_model(self):
        """Save model to disk"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history,
        }, self.model_path)
        print(f"[OK] RL Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            try:
                # weights_only=False needed for models saved with numpy arrays
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.step_count = checkpoint.get('step_count', 0)
                self.rewards_history = checkpoint.get('rewards_history', [])
                self.loss_history = checkpoint.get('loss_history', [])
                print(f"[OK] RL Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"[WARNING] Error loading RL model: {e}")
                return False
        return False
    
    def get_learning_metrics(self):
        """Get learning metrics for visualization"""
        return {
            'rewards': self.rewards_history,
            'losses': self.loss_history,
            'epsilon': self.epsilon_history,
            'predictions': self.prediction_history,
            'step_count': self.step_count,
            'avg_reward': np.mean(self.rewards_history[-100:]) if len(self.rewards_history) > 0 else 0,
            'avg_loss': np.mean(self.loss_history[-100:]) if len(self.loss_history) > 0 else 0,
        }

