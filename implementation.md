# Implementation Guide - Poker Bot Technical Specifications

## Table of Contents
1. [Environment Implementation](#environment-implementation)
2. [Neural Network Architectures](#neural-network-architectures)
3. [Deep CFR Algorithm](#deep-cfr-algorithm)
4. [Reservoir Sampling](#reservoir-sampling)
5. [PPO Implementation](#ppo-implementation)
6. [Agent Architecture](#agent-architecture)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation Framework](#evaluation-framework)

## Environment Implementation

### PettingZoo Wrapper (`poker_ai/envs/holdem_wrapper.py`)

```python
from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
from typing import Dict, List, Tuple, Optional

class HoldemWrapper:
    """
    3-player NLHE wrapper with discrete bet sizes
    """
    
    def __init__(self, num_players: int = 3, stack_size: int = 200):
        self.num_players = num_players
        self.stack_size = stack_size
        self.env = texas_holdem_no_limit_v6.env(num_players=num_players)
        
        # Discrete action mapping
        self.action_mapping = {
            0: "fold",
            1: "check_call", 
            2: "raise_half_pot",
            3: "raise_full_pot",
            4: "all_in"
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations"""
        pass
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute action and return (obs, reward, done, info)"""
        pass
    
    def get_legal_actions(self, agent: str) -> List[int]:
        """Get legal actions for current agent"""
        pass
```

**Key Features:**
- Wraps PettingZoo texas_holdem_no_limit_v6
- Supports 3+ players (configurable)
- Discrete action space with 5 actions
- Automatic action masking for legal moves
- Observation preprocessing for neural networks

### Card Utilities (`poker_ai/encoders/card_utils.py`)

```python
import numpy as np
from typing import List, Tuple
from enum import Enum

class CardEncoder:
    """
    Card encoding and hand evaluation utilities
    """
    
    def __init__(self):
        self.card_to_index = {}  # 52-card mapping
        self.index_to_card = {}
        self._build_mappings()
    
    def encode_cards(self, cards: List[str]) -> np.ndarray:
        """Convert card strings to one-hot encoding"""
        pass
    
    def evaluate_hand(self, hole_cards: List[str], 
                     community_cards: List[str]) -> int:
        """Evaluate poker hand strength (0-7462)"""
        pass
    
    def get_hand_bucket(self, hand_strength: int) -> int:
        """Map hand strength to abstraction bucket"""
        pass
```

**Features:**
- 52-card one-hot encoding
- Hand strength evaluation (using eval7 or similar)
- Hand abstraction for bucketing
- Suit and rank utilities

## Neural Network Architectures

### Information Set Transformer (`poker_ai/encoders/info_set_transformer.py`)

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class InfoSetTransformer(nn.Module):
    """
    Transformer encoder for poker information sets
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8, 
                 n_layers: int = 4,
                 n_actions: int = 5,
                 max_seq_len: int = 200):
        super().__init__()
        
        # Token embeddings
        self.card_embedding = nn.Embedding(52, d_model)
        self.action_embedding = nn.Embedding(n_actions, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output heads
        self.advantage_head = nn.Linear(d_model, n_actions)
        self.policy_head = nn.Linear(d_model, n_actions)
        
    def forward(self, 
                card_tokens: torch.Tensor,
                action_tokens: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning advantage and policy logits
        """
        # Combine card and action tokens
        card_emb = self.card_embedding(card_tokens)
        action_emb = self.action_embedding(action_tokens)
        
        # Concatenate sequences
        seq_emb = torch.cat([card_emb, action_emb], dim=1)
        
        # Add positional encoding
        seq_len = seq_emb.size(1)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=seq_emb.device))
        seq_emb = seq_emb + pos_emb
        
        # Transformer encoding
        encoded = self.transformer(seq_emb, src_key_padding_mask=~attention_mask)
        
        # Pool sequence (use [CLS] token or mean pooling)
        pooled = encoded.mean(dim=1)
        
        # Output heads
        advantages = self.advantage_head(pooled)
        policy_logits = self.policy_head(pooled)
        
        return advantages, policy_logits
```

**Architecture Details:**
- **Input**: Tokenized cards + action history
- **Encoding**: Multi-head self-attention with positional encoding
- **Output**: Dual heads for advantage estimation and policy distribution
- **Parameters**: ~1M parameters for 4-layer, 256-dim model

## Deep CFR Algorithm

### CFR Trainer (`poker_ai/solvers/deep_cfr/trainer.py`)

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from poker_ai.memory.reservoir import ReservoirBuffer
from poker_ai.encoders.info_set_transformer import InfoSetTransformer

class DeepCFRTrainer:
    """
    Deep CFR trainer using external sampling
    """
    
    def __init__(self, 
                 env,
                 network: InfoSetTransformer,
                 buffer_size: int = 1000000,
                 learning_rate: float = 1e-4,
                 batch_size: int = 512):
        
        self.env = env
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.buffer = ReservoirBuffer(buffer_size)
        self.batch_size = batch_size
        
        # CFR statistics
        self.iteration = 0
        self.regret_sum = {}
        self.strategy_sum = {}
        
    def train_iteration(self, num_traversals: int = 1000):
        """
        Run one CFR iteration with external sampling
        """
        for _ in range(num_traversals):
            # Sample random outcome
            self._external_sampling_traverse()
            
        # Update neural network
        self._update_network()
        
        self.iteration += 1
    
    def _external_sampling_traverse(self):
        """
        External sampling traversal for one game
        """
        # Reset environment
        obs = self.env.reset()
        
        # Play game to terminal state
        while not self.env.is_terminal():
            current_player = self.env.current_player
            
            # Get information set
            info_set = self._get_info_set(obs[current_player])
            
            # Get strategy from network
            strategy = self._get_strategy(info_set)
            
            # Sample action according to strategy
            action = np.random.choice(len(strategy), p=strategy)
            
            # Execute action
            obs, rewards, done, info = self.env.step(action)
            
            # Store experience for training
            self._store_experience(info_set, strategy, action, rewards[current_player])
    
    def _get_strategy(self, info_set: Dict) -> np.ndarray:
        """
        Get strategy from neural network
        """
        # Encode information set
        card_tokens, action_tokens, mask = self._encode_info_set(info_set)
        
        # Forward pass
        with torch.no_grad():
            advantages, policy_logits = self.network(card_tokens, action_tokens, mask)
        
        # Convert to strategy (regret matching)
        strategy = self._regret_matching(advantages.squeeze().numpy())
        
        return strategy
    
    def _regret_matching(self, advantages: np.ndarray) -> np.ndarray:
        """
        Convert advantages to strategy via regret matching
        """
        positive_regrets = np.maximum(advantages, 0)
        sum_regrets = np.sum(positive_regrets)
        
        if sum_regrets > 0:
            strategy = positive_regrets / sum_regrets
        else:
            strategy = np.ones(len(advantages)) / len(advantages)
            
        return strategy
    
    def _update_network(self):
        """
        Update neural network using stored experiences
        """
        if len(self.buffer) < self.batch_size:
            return
            
        # Sample batch from buffer
        batch = self.buffer.sample(self.batch_size)
        
        # Prepare training data
        card_tokens = torch.stack([item['card_tokens'] for item in batch])
        action_tokens = torch.stack([item['action_tokens'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        targets = torch.stack([item['regret_targets'] for item in batch])
        
        # Forward pass
        advantages, policy_logits = self.network(card_tokens, action_tokens, masks)
        
        # Compute loss (regret prediction)
        loss = nn.MSELoss()(advantages, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**Key Components:**
- **External Sampling**: Reduces variance compared to vanilla CFR
- **Neural Network**: Replaces tabular regret storage
- **Regret Matching**: Converts advantages to strategies
- **Experience Replay**: Stores and samples training data

## Reservoir Sampling

### Reservoir Buffer (`poker_ai/memory/reservoir.py`)

```python
import numpy as np
import random
from typing import List, Dict, Any

class ReservoirBuffer:
    """
    O(1) reservoir sampling buffer for experience replay
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.size = 0
        self.total_added = 0
    
    def add(self, experience: Dict[str, Any]):
        """
        Add experience to buffer using reservoir sampling
        """
        if self.size < self.capacity:
            # Fill buffer initially
            self.buffer.append(experience)
            self.size += 1
        else:
            # Reservoir sampling: replace with probability k/n
            rand_idx = random.randint(0, self.total_added)
            if rand_idx < self.capacity:
                self.buffer[rand_idx] = experience
        
        self.total_added += 1
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample batch from buffer
        """
        if self.size < batch_size:
            return random.sample(self.buffer, self.size)
        else:
            return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return self.size
```

**Properties:**
- **Time Complexity**: O(1) per addition
- **Space Complexity**: O(k) where k is capacity
- **Uniform Sampling**: Each item has equal probability k/n
- **Streaming**: Works with unknown/infinite data streams

## PPO Implementation

### RL Agent (`poker_ai/agents/rl_agent.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from poker_ai.encoders.info_set_transformer import InfoSetTransformer

class ActorCriticAgent(nn.Module):
    """
    Actor-Critic agent for PPO training
    """
    
    def __init__(self, 
                 transformer: InfoSetTransformer,
                 n_actions: int = 5):
        super().__init__()
        
        self.transformer = transformer
        self.n_actions = n_actions
        
        # Value head (critic)
        self.value_head = nn.Linear(transformer.d_model, 1)
        
    def forward(self, 
                card_tokens: torch.Tensor,
                action_tokens: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits, values, and advantages
        """
        # Transformer encoding
        encoded = self.transformer.transformer(
            self.transformer._embed_sequence(card_tokens, action_tokens),
            src_key_padding_mask=~attention_mask
        )
        
        # Pool sequence
        pooled = encoded.mean(dim=1)
        
        # Output heads
        policy_logits = self.transformer.policy_head(pooled)
        values = self.value_head(pooled)
        advantages = self.transformer.advantage_head(pooled)
        
        return policy_logits, values, advantages
    
    def get_action(self, 
                   card_tokens: torch.Tensor,
                   action_tokens: torch.Tensor,
                   attention_mask: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        """
        with torch.no_grad():
            policy_logits, values, _ = self.forward(card_tokens, action_tokens, attention_mask)
            
        # Sample action
        probs = F.softmax(policy_logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        
        return action, probs, values

class PPOTrainer:
    """
    PPO trainer for self-play fine-tuning
    """
    
    def __init__(self,
                 agent: ActorCriticAgent,
                 blueprint_agent,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 kl_coeff: float = 0.1):
        
        self.agent = agent
        self.blueprint_agent = blueprint_agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single PPO training step
        """
        # Forward pass
        policy_logits, values, _ = self.agent(
            batch['card_tokens'],
            batch['action_tokens'], 
            batch['attention_masks']
        )
        
        # Compute policy loss
        policy_loss = self._compute_policy_loss(policy_logits, batch)
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), batch['returns'])
        
        # Compute KL divergence from blueprint
        kl_loss = self._compute_kl_loss(policy_logits, batch)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + self.kl_coeff * kl_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_policy_loss(self, policy_logits: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Compute clipped policy loss
        """
        # Current policy probabilities
        current_probs = F.softmax(policy_logits, dim=-1)
        current_log_probs = torch.log(current_probs.gather(1, batch['actions'].unsqueeze(1)))
        
        # Importance sampling ratio
        ratio = torch.exp(current_log_probs - batch['old_log_probs'])
        
        # Clipped objective
        advantages = batch['advantages']
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def _compute_kl_loss(self, policy_logits: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Compute KL divergence from blueprint policy
        """
        # Current policy
        current_probs = F.softmax(policy_logits, dim=-1)
        
        # Blueprint policy
        blueprint_probs = batch['blueprint_probs']
        
        # KL divergence
        kl_loss = F.kl_div(
            torch.log(current_probs),
            blueprint_probs,
            reduction='batchmean'
        )
        
        return kl_loss
```

**Key Features:**
- **Actor-Critic**: Shared Transformer backbone with dual heads
- **PPO**: Clipped objective function for stable training
- **KL Regularization**: Prevents divergence from blueprint strategy
- **Self-Play**: Training against population of past agents

## Training Pipeline

### Training Scripts

```python
# scripts/10_run_deep_cfr.py
def train_deep_cfr():
    # Initialize environment and network
    env = HoldemWrapper(num_players=3)
    network = InfoSetTransformer()
    trainer = DeepCFRTrainer(env, network)
    
    # Training loop
    for iteration in range(1000):
        trainer.train_iteration(num_traversals=1000)
        
        # Evaluate and save checkpoint
        if iteration % 100 == 0:
            save_checkpoint(network, f'cfr_checkpoint_{iteration}.pt')
            evaluate_exploitability(trainer)

# scripts/40_train_rl_finetune.py  
def train_ppo_finetune():
    # Load blueprint agent
    blueprint_agent = load_checkpoint('cfr_checkpoint_1000.pt')
    
    # Initialize RL agent
    rl_agent = ActorCriticAgent(blueprint_agent.transformer)
    trainer = PPOTrainer(rl_agent, blueprint_agent)
    
    # Self-play training
    for epoch in range(100):
        # Collect experience
        experience = collect_selfplay_experience(rl_agent)
        
        # Train on experience
        for _ in range(10):
            trainer.train_step(experience)
        
        # Evaluate performance
        evaluate_against_baselines(rl_agent)
```

## Evaluation Framework

### Metrics and Benchmarks

```python
# poker_ai/evaluation/evaluator.py
class PokerEvaluator:
    """
    Evaluation framework for poker agents
    """
    
    def __init__(self):
        self.baseline_agents = [
            RandomAgent(),
            HeuristicAgent(),
            CallStationAgent()
        ]
    
    def evaluate_exploitability(self, agent) -> float:
        """
        Approximate exploitability using best response
        """
        pass
    
    def evaluate_winrate(self, agent, num_games: int = 10000) -> Dict[str, float]:
        """
        Evaluate win rate against baseline agents
        """
        pass
    
    def tournament_evaluation(self, agents: List) -> Dict[str, float]:
        """
        Round-robin tournament evaluation
        """
        pass
```

**Evaluation Metrics:**
- **Exploitability**: Distance from Nash equilibrium
- **Win Rate**: Big blinds per 100 hands against baselines
- **Variance**: Consistency of performance
- **Robustness**: Performance against diverse opponents

## Configuration Management

### Hydra Configuration

```yaml
# configs/deep_cfr/default.yaml
trainer:
  iterations: 1000
  traversals_per_iteration: 1000
  buffer_size: 1000000
  batch_size: 512
  learning_rate: 1e-4

network:
  d_model: 256
  n_heads: 8
  n_layers: 4
  dropout: 0.1

environment:
  num_players: 3
  stack_size: 200
  ante: 0
  small_blind: 1
  big_blind: 2

# configs/ppo/default.yaml
trainer:
  epochs: 100
  games_per_epoch: 1000
  learning_rate: 3e-4
  clip_epsilon: 0.2
  kl_coeff: 0.1
  value_coeff: 0.5

agent:
  shared_backbone: true
  separate_value_head: true
```

This implementation guide provides comprehensive technical specifications for building the poker bot system. Each component is designed to be modular and extensible, allowing for experimentation with different architectures and hyperparameters.