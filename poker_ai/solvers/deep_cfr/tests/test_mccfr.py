
import unittest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from poker_ai.solvers.deep_cfr.trainer import DeepCFRTrainer
from poker_ai.models.info_set_transformer import InfoSetTransformer

class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, num_players=3):
        self.num_players = num_players
        self.agents = [f'player_{i}' for i in range(num_players)]
        self.done = False
        self.current_player_idx = 0
        self.step_count = 0
        
    def reset(self):
        self.done = False
        self.current_player_idx = 0
        self.step_count = 0
        observations = {}
        for i in range(self.num_players):
            observations[f'player_{i}'] = {
                'hole_cards': ['As', 'Kd'],
                'community_cards': [],
                'action_history': [],
                'legal_actions': [0, 1, 2]
            }
        return observations
    
    def current_player(self):
        if self.done:
            return None
        return self.agents[self.current_player_idx]
    
    def get_legal_actions(self, player):
        if self.done:
            return []
        return [0, 1, 2]
    
    def is_terminal(self):
        return self.done
    
    def step(self, action):
        self.step_count += 1
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        
        if self.step_count >= 9:
            self.done = True
        
        observations = {}
        if not self.done:
            for i in range(self.num_players):
                observations[f'player_{i}'] = {
                    'hole_cards': ['As', 'Kd'],
                    'community_cards': [],
                    'action_history': [action],
                    'legal_actions': [0, 1, 2]
                }
            
        rewards = {f'player_{i}': 1.0/self.num_players for i in range(self.num_players)}
        done = self.done
        info = {}
        
        return observations, rewards, done, info
    
    def get_final_rewards(self):
        return {f'player_{i}': 1.0/self.num_players for i in range(self.num_players)}

class TestMCCFR(unittest.TestCase):
    """Test cases for MCCFR implementation in DeepCFRTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        self.network = InfoSetTransformer(
            d_model=64, n_heads=2, n_layers=1, n_actions=5, max_seq_len=100
        )
        self.trainer = DeepCFRTrainer(
            env=self.mock_env,
            network=self.network,
            buffer_size=1000,
            batch_size=32,
        )

    def test_estimate_utility_from_hand_strength(self):
        """Test the heuristic for estimating utility from hand strength."""
        strong_hand_obs = {'hole_cards': ['As', 'Ad'], 'community_cards': []}
        weak_hand_obs = {'hole_cards': ['2s', '7d'], 'community_cards': []}
        
        # For a strong hand, aggressive actions should have higher utility
        utility_strong_aggressive = self.trainer._estimate_utility_from_hand_strength(
            action=2, observations=strong_hand_obs, player_name='player_0'
        )
        utility_strong_fold = self.trainer._estimate_utility_from_hand_strength(
            action=0, observations=strong_hand_obs, player_name='player_0'
        )
        self.assertGreater(utility_strong_aggressive, utility_strong_fold)

        # For a weak hand, folding should have higher utility than aggressive actions
        utility_weak_aggressive = self.trainer._estimate_utility_from_hand_strength(
            action=2, observations=weak_hand_obs, player_name='player_0'
        )
        utility_weak_fold = self.trainer._estimate_utility_from_hand_strength(
            action=0, observations=weak_hand_obs, player_name='player_0'
        )
        self.assertLess(utility_weak_aggressive, utility_weak_fold)

if __name__ == '__main__':
    unittest.main()
