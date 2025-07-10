"""
Abstract base class for poker agents.

This defines the standard interface that all poker agents must implement,
ensuring consistency across different agent types (CFR, RL, heuristic, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all poker agents.
    
    This interface ensures that all agents can be used interchangeably
    in tournaments, training, and evaluation scenarios.
    """
    
    def __init__(self, agent_id: str, num_actions: int = 5):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            num_actions: Number of possible actions (default: 5 for NLHE)
        """
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.is_training = True
        
    @abstractmethod
    def get_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Select an action given the current game state.
        
        Args:
            observation: Game state observation from environment
            legal_actions: List of legal action indices
            
        Returns:
            Selected action index
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset agent state for a new game/episode.
        """
        pass
    
    def update(self, experience: Dict[str, Any]) -> None:
        """
        Update agent with experience (optional for some agents).
        
        Args:
            experience: Dictionary containing game experience data
        """
        pass
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set whether agent is in training or evaluation mode.
        
        Args:
            training: True for training mode, False for evaluation
        """
        self.is_training = training
    
    def get_strategy(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Get the full strategy distribution for the current state.
        
        Args:
            observation: Game state observation
            
        Returns:
            Probability distribution over actions
        """
        # Default implementation: uniform over legal actions
        legal_actions = observation.get('legal_actions', list(range(self.num_actions)))
        strategy = np.zeros(self.num_actions)
        if legal_actions:
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = prob
        return strategy
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file (optional).
        
        Args:
            filepath: Path to save the agent
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file (optional).
        
        Args:
            filepath: Path to load the agent from
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, actions={self.num_actions})"


class RandomAgent(BaseAgent):
    """
    Simple random agent for testing and baseline comparison.
    """
    
    def __init__(self, agent_id: str = "random_agent", num_actions: int = 5):
        """Initialize random agent."""
        super().__init__(agent_id, num_actions)
    
    def get_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """Select random legal action."""
        if not legal_actions:
            return 0  # Default to fold if no legal actions
        return np.random.choice(legal_actions)
    
    def reset(self) -> None:
        """Reset agent state (no-op for random agent)."""
        pass
    
    def get_strategy(self, observation: Dict[str, Any]) -> np.ndarray:
        """Return uniform strategy over legal actions."""
        legal_actions = observation.get('legal_actions', list(range(self.num_actions)))
        strategy = np.zeros(self.num_actions)
        if legal_actions:
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = prob
        return strategy