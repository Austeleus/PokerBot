"""
Reservoir sampling buffer for experience replay.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


class ReservoirBuffer:
    """
    O(1) reservoir sampling buffer for experience replay.
    
    Implements the classic reservoir sampling algorithm to maintain
    a uniform random sample from a stream of experiences, ensuring
    each experience has equal probability of being in the buffer.
    """
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize reservoir buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.buffer = []
        self.size = 0
        self.total_added = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(self, experience: Dict[str, Any]):
        """
        Add experience to buffer using reservoir sampling.
        
        Each experience has probability min(1, k/n) of being in the buffer,
        where k is capacity and n is total number of experiences seen.
        
        Args:
            experience: Dictionary containing experience data
        """
        self.total_added += 1
        
        if self.size < self.capacity:
            # Fill buffer initially
            self.buffer.append(experience.copy())
            self.size += 1
        else:
            # Reservoir sampling: replace with probability k/n
            # Generate random index from 0 to total_added-1
            rand_idx = random.randint(0, self.total_added - 1)
            
            # If random index is within buffer capacity, replace that position
            if rand_idx < self.capacity:
                self.buffer[rand_idx] = experience.copy()
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if self.size == 0:
            return []
        
        # Sample with replacement if batch_size > buffer size
        if batch_size >= self.size:
            return [exp.copy() for exp in self.buffer]
        
        # Sample without replacement
        sampled_indices = random.sample(range(self.size), batch_size)
        return [self.buffer[idx].copy() for idx in sampled_indices]
    
    def sample_indices(self, batch_size: int) -> List[int]:
        """
        Sample indices from buffer (useful for weighted sampling).
        
        Args:
            batch_size: Number of indices to sample
            
        Returns:
            List of sampled indices
        """
        if self.size == 0:
            return []
        
        if batch_size >= self.size:
            return list(range(self.size))
        
        return random.sample(range(self.size), batch_size)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all experiences in buffer.
        
        Returns:
            List of all experiences
        """
        return [exp.copy() for exp in self.buffer]
    
    def clear(self):
        """Clear the buffer and reset counters."""
        self.buffer.clear()
        self.size = 0
        self.total_added = 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get experience at index."""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for buffer of size {self.size}")
        return self.buffer[index].copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'capacity': self.capacity,
            'size': self.size,
            'total_added': self.total_added,
            'utilization': self.size / self.capacity if self.capacity > 0 else 0,
            'acceptance_rate': self.size / self.total_added if self.total_added > 0 else 0
        }


class PrioritizedReservoirBuffer(ReservoirBuffer):
    """
    Reservoir buffer with priority-based sampling.
    
    Maintains reservoir sampling for storage but allows
    priority-weighted sampling for training.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, seed: Optional[int] = None):
        """
        Initialize prioritized reservoir buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            seed: Random seed for reproducible sampling
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.priorities = []
        self.max_priority = 1.0
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None):
        """
        Add experience with optional priority.
        
        Args:
            experience: Dictionary containing experience data
            priority: Priority value (defaults to max priority)
        """
        if priority is None:
            priority = self.max_priority
        
        self.total_added += 1
        
        if self.size < self.capacity:
            # Fill buffer initially
            self.buffer.append(experience.copy())
            self.priorities.append(priority)
            self.size += 1
        else:
            # Reservoir sampling
            rand_idx = random.randint(0, self.total_added - 1)
            
            if rand_idx < self.capacity:
                self.buffer[rand_idx] = experience.copy()
                self.priorities[rand_idx] = priority
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, beta: float = 1.0) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Sample batch with priority weighting.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of (experiences, weights, indices)
        """
        if self.size == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:self.size]) ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(self.size, size=min(batch_size, self.size), 
                                 p=probabilities, replace=batch_size > self.size)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)  # Normalize by max weight
        
        # Get experiences
        experiences = [self.buffer[idx].copy() for idx in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for given indices.
        
        Args:
            indices: Array of buffer indices
            priorities: Array of new priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def clear(self):
        """Clear the buffer and reset counters."""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0


class ExperienceCollector:
    """
    Helper class to collect and format experiences for reservoir buffer.
    """
    
    @staticmethod
    def create_cfr_experience(info_set: Dict[str, Any], 
                            action: int,
                            regret: float,
                            strategy: np.ndarray,
                            reward: float) -> Dict[str, Any]:
        """
        Create CFR training experience.
        
        Args:
            info_set: Information set representation
            action: Action taken
            regret: Regret value for the action
            strategy: Strategy distribution
            reward: Reward received
            
        Returns:
            Formatted experience dictionary
        """
        return {
            'type': 'cfr',
            'info_set': info_set.copy(),
            'action': action,
            'regret': regret,
            'strategy': strategy.copy(),
            'reward': reward,
            'timestamp': random.random()  # For tie-breaking in sampling
        }
    
    @staticmethod
    def create_rl_experience(observation: Dict[str, Any],
                           action: int,
                           reward: float,
                           next_observation: Dict[str, Any],
                           done: bool,
                           info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create RL training experience.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            info: Additional info
            
        Returns:
            Formatted experience dictionary
        """
        return {
            'type': 'rl',
            'observation': observation.copy(),
            'action': action,
            'reward': reward,
            'next_observation': next_observation.copy(),
            'done': done,
            'info': info.copy(),
            'timestamp': random.random()
        }