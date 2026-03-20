#!/usr/bin/env python3
"""
Deep Reinforcement Learning Agent for Adaptive Content Sequencing

Implements a sophisticated RL agent that learns optimal content sequencing
for personalized learning experiences using:
- Deep Q-Network (DQN) for content selection
- Actor-Critic for difficulty adjustment  
- Multi-armed bandit for exploration vs exploitation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])

class ActionType(Enum):
    """Types of actions the RL agent can take."""
    RECOMMEND_CONTENT = "recommend_content"
    ADJUST_DIFFICULTY = "adjust_difficulty"
    SUGGEST_QUIZ = "suggest_quiz"
    TAKE_BREAK = "take_break"
    REVIEW_TOPIC = "review_topic"

@dataclass
class LearningEnvironmentState:
    """Current state of the learning environment."""
    user_id: str
    current_topic_mastery: Dict[str, float]  # topic -> mastery (0-1)
    recent_performance: List[float]  # last 10 performance scores
    time_in_session: float  # minutes
    difficulty_progression: List[float]  # recent difficulty levels
    engagement_level: float  # 0-1
    fatigue_level: float  # 0-1
    topics_covered_today: int
    consecutive_correct: int
    consecutive_incorrect: int
    preferred_learning_style: str
    context_features: np.ndarray  # additional contextual features

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for learning optimal content sequencing actions.
    
    Takes learning environment state and outputs Q-values for different actions.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 50,  # number of possible content pieces
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layer for Q-values
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Dueling DQN architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DQN."""
        features = self.network[:-1](state)  # All layers except last
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for continuous difficulty adjustment.
    
    Actor outputs probability distribution over difficulty levels.
    Critic estimates value of current state.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 1,  # continuous difficulty (0-1)
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # mean and std
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        shared_features = self.shared(state)
        
        # Actor output: mean and std for normal distribution
        actor_output = self.actor(shared_features)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        
        # Critic output: state value
        value = self.critic(shared_features)
        
        return mean, std, value

class MultiArmedBandit:
    """
    Multi-armed bandit for exploration vs exploitation in content selection.
    
    Uses Thompson Sampling with Beta distributions for each content piece.
    """
    
    def __init__(self, num_arms: int = 100):
        self.num_arms = num_arms
        # Beta distribution parameters for each arm
        self.alpha = np.ones(num_arms)  # successes + 1
        self.beta = np.ones(num_arms)   # failures + 1
        
    def select_arm(self, exclude_arms: Optional[List[int]] = None) -> int:
        """Select arm using Thompson Sampling."""
        exclude_arms = exclude_arms or []
        
        # Sample from Beta distributions
        samples = np.random.beta(self.alpha, self.beta)
        
        # Exclude certain arms
        for arm in exclude_arms:
            if 0 <= arm < self.num_arms:
                samples[arm] = -1
        
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        """Update Beta distribution parameters based on reward."""
        if 0 <= arm < self.num_arms:
            if reward > 0.5:  # Success
                self.alpha[arm] += 1
            else:  # Failure
                self.beta[arm] += 1
    
    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all arms."""
        stats = {}
        for i in range(self.num_arms):
            mean = self.alpha[i] / (self.alpha[i] + self.beta[i])
            variance = (self.alpha[i] * self.beta[i]) / \
                      ((self.alpha[i] + self.beta[i])**2 * (self.alpha[i] + self.beta[i] + 1))
            stats[i] = {
                'mean_reward': mean,
                'variance': variance,
                'confidence': 1 / (1 + variance)
            }
        return stats

class AdaptiveLearningAgent:
    """
    Main reinforcement learning agent that coordinates all RL components
    for optimal content sequencing and difficulty adjustment.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        content_action_dim: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # DQN for content selection
        self.dqn_online = DQNNetwork(state_dim, content_action_dim).to(device)
        self.dqn_target = DQNNetwork(state_dim, content_action_dim).to(device)
        self.dqn_optimizer = optim.Adam(self.dqn_online.parameters(), lr=learning_rate)
        
        # Actor-Critic for difficulty adjustment
        self.actor_critic = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=128  # Match the hidden_dim used in initialization
        ).to(device)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Multi-armed bandit for exploration
        self.bandit = MultiArmedBandit(content_action_dim)
        
        # Experience replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # Update target network every N steps
        self.target_update_frequency = 1000
        
        print(f"Adaptive Learning Agent initialized on {device}")
    
    def state_to_tensor(self, state: LearningEnvironmentState) -> torch.Tensor:
        """Convert learning environment state to tensor."""
        
        # Extract numerical features from state
        features = [
            # Topic mastery features (pad/truncate to 20 topics)
            *list(state.current_topic_mastery.values())[:20],
            # Performance features
            *state.recent_performance[-10:],  # last 10 performances
            # Session features
            state.time_in_session / 60.0,  # normalize to hours
            state.engagement_level,
            state.fatigue_level,
            state.topics_covered_today / 10.0,  # normalize
            state.consecutive_correct / 5.0,  # normalize
            state.consecutive_incorrect / 5.0,  # normalize
            # Difficulty progression
            *state.difficulty_progression[-5:],  # last 5 difficulty levels
        ]
        
        # Pad to state_dim
        while len(features) < 128:
            features.append(0.0)
        
        return torch.tensor(features[:128], dtype=torch.float32, device=self.device)
    
    async def select_content_action(
        self, 
        state: LearningEnvironmentState,
        use_bandit: bool = True,
        explore_probability: float = 0.1
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select next content action using DQN + Multi-armed bandit.
        
        Returns:
            (action_id, action_metadata)
        """
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random exploration
            action = random.randint(0, self.dqn_online.network[-1].out_features - 1)
            method = "random_exploration"
        elif use_bandit and random.random() < explore_probability:
            # Multi-armed bandit selection
            action = self.bandit.select_arm()
            method = "bandit_selection"
        else:
            # DQN-based selection
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.dqn_online(state_tensor)
                action = int(q_values.argmax().item())
            method = "dqn_selection"
        
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * (1 - 1/self.epsilon_decay)
        )
        
        metadata = {
            'method': method,
            'epsilon': self.epsilon,
            'q_values': q_values.cpu().numpy().tolist() if 'q_values' in locals() else None
        }
        
        return action, metadata
    
    async def adjust_difficulty(
        self, 
        state: LearningEnvironmentState
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Adjust difficulty using Actor-Critic method.
        
        Returns:
            (difficulty_level, adjustment_metadata)
        """
        
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, std, value = self.actor_critic(state_tensor)
            
            # Sample difficulty from normal distribution
            difficulty_dist = torch.distributions.Normal(mean, std)
            difficulty_raw = difficulty_dist.sample()
            
            # Clip to valid range [0, 1]
            difficulty = torch.clamp(difficulty_raw, 0.0, 1.0)
        
        metadata = {
            'predicted_value': float(value.item()),
            'difficulty_mean': float(mean.item()),
            'difficulty_std': float(std.item()),
            'sampled_difficulty': float(difficulty.item())
        }
        
        return float(difficulty.item()), metadata
    
    def calculate_reward(
        self,
        state: LearningEnvironmentState,
        action: int,
        next_state: LearningEnvironmentState,
        user_feedback: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for the taken action.
        
        Reward function considers:
        - Learning progress (mastery improvement)
        - Engagement level
        - Appropriate difficulty
        - Time efficiency
        """
        
        reward = 0.0
        
        # Learning progress reward
        current_avg_mastery = np.mean(list(state.current_topic_mastery.values())) \
                             if state.current_topic_mastery else 0.0
        next_avg_mastery = np.mean(list(next_state.current_topic_mastery.values())) \
                          if next_state.current_topic_mastery else 0.0
        
        mastery_improvement = next_avg_mastery - current_avg_mastery
        reward += mastery_improvement * 10.0  # Scale up mastery improvements
        
        # Engagement reward
        engagement_change = next_state.engagement_level - state.engagement_level
        reward += engagement_change * 5.0
        
        # Appropriate difficulty reward
        if 0.6 <= user_feedback.get('difficulty_rating', 0.5) <= 0.8:
            reward += 2.0  # Reward for optimal difficulty
        
        # Performance-based reward
        performance = user_feedback.get('performance_score', 0.5)
        if performance > 0.7:
            reward += 3.0
        elif performance < 0.3:
            reward -= 1.0
        
        # Time efficiency reward
        expected_time = user_feedback.get('expected_time', 300)
        actual_time = user_feedback.get('actual_time', 300)
        if actual_time <= expected_time:
            reward += 1.0
        
        # Penalize fatigue
        if next_state.fatigue_level > 0.8:
            reward -= 2.0
        
        return reward
    
    def store_experience(
        self,
        state: LearningEnvironmentState,
        action: int,
        reward: float,
        next_state: LearningEnvironmentState,
        done: bool = False
    ):
        """Store experience in replay buffer."""
        
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        
        experience = Experience(
            state_tensor.cpu().numpy(),
            action,
            reward,
            next_state_tensor.cpu().numpy(),
            done
        )
        
        self.memory.append(experience)
    
    async def train_dqn(self) -> Dict[str, float]:
        """Train the DQN using experience replay."""
        
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.tensor([e.state for e in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([e.next_state for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        
        # Current Q-values
        current_q_values = self.dqn_online(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.dqn_target(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_online.parameters(), 1.0)
        self.dqn_optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.dqn_target.load_state_dict(self.dqn_online.state_dict())
        
        return {'loss': float(loss.item())}
    
    async def update_bandit(self, action: int, reward: float):
        """Update multi-armed bandit with reward feedback."""
        self.bandit.update(action, reward)
    
    def get_agent_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the RL agent."""
        
        bandit_stats = self.bandit.get_arm_statistics()
        top_arms = sorted(bandit_stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True)[:10]
        
        analytics = {
            'training_steps': self.steps_done,
            'current_epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'top_content_arms': [
                {
                    'content_id': arm_id,
                    'mean_reward': stats['mean_reward'],
                    'confidence': stats['confidence']
                }
                for arm_id, stats in top_arms
            ],
            'exploration_rate': self.epsilon,
            'target_updates': self.steps_done // self.target_update_frequency
        }
        
        return analytics

# Global RL agent instance
adaptive_agent = AdaptiveLearningAgent()
