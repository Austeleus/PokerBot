"""
CFR Agent implementation that uses trained Deep CFR policy for gameplay.

This agent wraps a trained Deep CFR model and provides the standard
agent interface for tournaments, evaluation, and further training.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from poker_ai.agents.base_agent import BaseAgent
from poker_ai.models.info_set_transformer import InfoSetTransformer
from poker_ai.encoders.card_utils import CardEncoder


class CFRAgent(BaseAgent):
    """
    Agent that uses a trained Deep CFR policy for decision making.
    
    This agent wraps a trained InfoSetTransformer model and provides
    the standard agent interface for gameplay and evaluation.
    """
    
    def __init__(self, 
                 network: InfoSetTransformer,
                 agent_id: str = "cfr_agent",
                 device: str = "cpu",
                 temperature: float = 1.0):
        """
        Initialize CFR agent with trained network.
        
        Args:
            network: Trained InfoSetTransformer model
            agent_id: Unique identifier for this agent
            device: Device to run inference on
            temperature: Temperature for action sampling (1.0 = no change)
        """
        super().__init__(agent_id, num_actions=5)
        
        self.network = network.to(device)
        self.device = device
        self.temperature = temperature
        self.card_encoder = CardEncoder()
        
        # Set network to evaluation mode
        self.network.eval()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.game_count = 0
        
    def get_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Select action using trained CFR policy.
        
        Args:
            observation: Game state observation
            legal_actions: List of legal action indices
            
        Returns:
            Selected action index
        """
        if not legal_actions:
            return 0  # Default to fold
        
        # Get strategy from network
        strategy = self.get_strategy(observation)
        
        # Filter to legal actions only
        legal_strategy = np.zeros_like(strategy)
        for action in legal_actions:
            legal_strategy[action] = strategy[action]
        
        # Normalize legal strategy
        strategy_sum = np.sum(legal_strategy)
        if strategy_sum > 0:
            legal_strategy = legal_strategy / strategy_sum
        else:
            # Fallback to uniform over legal actions
            prob = 1.0 / len(legal_actions)
            legal_strategy = np.zeros_like(strategy)
            for action in legal_actions:
                legal_strategy[action] = prob
        
        # Sample action
        if self.is_training:
            # Sample according to strategy during training
            action = np.random.choice(len(legal_strategy), p=legal_strategy)
        else:
            # Take best action during evaluation
            best_actions = [i for i in legal_actions if legal_strategy[i] == np.max(legal_strategy[legal_actions])]
            action = np.random.choice(best_actions)
        
        return action
    
    def get_strategy(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Get full strategy distribution for current state.
        
        Args:
            observation: Game state observation
            
        Returns:
            Probability distribution over all actions
        """
        # Encode information set
        info_set_encoding = self._encode_information_set(observation)
        
        if info_set_encoding is None:
            # Fallback to uniform strategy
            return np.ones(self.num_actions) / self.num_actions
        
        try:
            with torch.no_grad():
                # Forward pass through network
                tokens = info_set_encoding['tokens']
                attention_mask = info_set_encoding['attention_mask']
                
                advantages, policy_logits, values = self.network(tokens, attention_mask)
                advantages = advantages.squeeze().cpu().numpy()
            
            # Convert advantages to strategy using regret matching
            legal_actions = observation.get('legal_actions', list(range(self.num_actions)))
            strategy = self._regret_matching(advantages, legal_actions)
            
            # Apply temperature scaling if specified
            if self.temperature != 1.0:
                strategy = self._apply_temperature(strategy, legal_actions)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Failed to get strategy from network: {e}")
            # Fallback to uniform strategy
            strategy = np.zeros(self.num_actions)
            legal_actions = observation.get('legal_actions', list(range(self.num_actions)))
            if legal_actions:
                prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strategy[action] = prob
            return strategy
    
    def _encode_information_set(self, observation: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode poker information set for neural network input.
        
        Args:
            observation: Game observation
            
        Returns:
            Encoded information set or None if encoding fails
        """
        try:
            # Extract game state information
            hole_cards = observation.get('hole_cards', [])
            community_cards = observation.get('community_cards', [])
            action_history = observation.get('action_history', [])
            
            # Use the transformer's built-in encoding method
            tokens = self.network.encode_game_state(
                hole_cards=hole_cards,
                community_cards=community_cards,
                action_history=action_history[-20:],  # Last 20 actions
                current_player=0  # Simplified for now
            )
            
            # Create attention mask (all tokens are valid)
            seq_len = tokens.size(1)
            attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=self.device)
            
            # Apply device
            tokens = tokens.to(self.device)
            
            return {
                'tokens': tokens,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            # Log detailed information for debugging
            self.logger.error(
                f"CFR Agent failed to encode information set: {e}\n"
                f"Observation: {observation}\n"
                f"Hole cards: {observation.get('hole_cards', [])}\n"
                f"Community cards: {observation.get('community_cards', [])}\n"
                f"Action history: {observation.get('action_history', [])}"
            )
            return None
    
    def _regret_matching(self, advantages: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """
        Convert advantages to strategy using regret matching.
        
        Args:
            advantages: Advantage values for each action
            legal_actions: List of legal action indices
            
        Returns:
            Strategy probability distribution
        """
        strategy = np.zeros(len(advantages))
        
        if not legal_actions:
            return strategy
        
        # Get advantages for legal actions
        legal_advantages = advantages[legal_actions]
        
        # Regret matching: positive regrets only
        positive_regrets = np.maximum(legal_advantages, 0)
        sum_positive_regrets = np.sum(positive_regrets)
        
        if sum_positive_regrets > 0:
            # Proportional to positive regrets
            for i, action in enumerate(legal_actions):
                strategy[action] = positive_regrets[i] / sum_positive_regrets
        else:
            # Uniform over legal actions
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = prob
        
        return strategy
    
    def _apply_temperature(self, strategy: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """
        Apply temperature scaling to strategy.
        
        Args:
            strategy: Original strategy
            legal_actions: Legal actions
            
        Returns:
            Temperature-scaled strategy
        """
        if self.temperature <= 0:
            # Greedy action selection
            best_action = max(legal_actions, key=lambda a: strategy[a])
            temp_strategy = np.zeros_like(strategy)
            temp_strategy[best_action] = 1.0
            return temp_strategy
        
        # Temperature scaling
        temp_strategy = np.zeros_like(strategy)
        legal_probs = strategy[legal_actions]
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        legal_probs = np.maximum(legal_probs, epsilon)
        
        # Apply temperature
        scaled_logits = np.log(legal_probs) / self.temperature
        scaled_probs = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        
        # Set scaled probabilities
        for i, action in enumerate(legal_actions):
            temp_strategy[action] = scaled_probs[i]
        
        return temp_strategy
    
    def reset(self) -> None:
        """Reset agent state for new game."""
        self.game_count += 1
    
    def update(self, experience: Dict[str, Any]) -> None:
        """
        Update agent with experience (no-op for CFR agent).
        
        CFR agents are trained offline, so no online updates.
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save the agent
        """
        state = {
            'network_state_dict': self.network.state_dict(),
            'agent_id': self.agent_id,
            'temperature': self.temperature,
            'game_count': self.game_count
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load the agent from
        """
        state = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(state['network_state_dict'])
        self.agent_id = state.get('agent_id', self.agent_id)
        self.temperature = state.get('temperature', self.temperature)
        self.game_count = state.get('game_count', 0)
    
    @classmethod
    def from_checkpoint(cls, 
                       checkpoint_path: str, 
                       network_config: Optional[Dict] = None,
                       agent_id: str = "cfr_agent",
                       device: str = "cpu",
                       temperature: float = 1.0) -> 'CFRAgent':
        """
        Create CFR agent from saved checkpoint.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            network_config: Network configuration (if creating new network)
            agent_id: Agent identifier
            device: Device to run on
            temperature: Temperature for action sampling
            
        Returns:
            Initialized CFRAgent
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create network
        if network_config is None:
            network_config = {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'n_actions': 5,
                'max_seq_len': 500
            }
        
        network = InfoSetTransformer(**network_config)
        network.load_state_dict(checkpoint['network_state_dict'])
        
        # Create agent
        agent = cls(network, agent_id, device, temperature)
        
        return agent