"""
Unit tests for Deep CFR trainer.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from poker_ai.solvers.deep_cfr.trainer import DeepCFRTrainer
from poker_ai.models.info_set_transformer import InfoSetTransformer
from poker_ai.memory.reservoir import ReservoirBuffer


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self):
        self.done = False
        self.current_player_idx = 0
        self.players = ['player_0', 'player_1', 'player_2']
        self.step_count = 0
        
    def reset(self):
        self.done = False
        self.current_player_idx = 0
        self.step_count = 0
        return {
            'player_0': {
                'hole_cards': ['As', 'Kd'],
                'community_cards': [],
                'action_history': [],
                'legal_actions': [0, 1, 2]
            },
            'player_1': {
                'hole_cards': ['Qh', 'Jc'],
                'community_cards': [],
                'action_history': [],
                'legal_actions': [0, 1, 2]
            },
            'player_2': {
                'hole_cards': ['Ts', '9h'],
                'community_cards': [],
                'action_history': [],
                'legal_actions': [0, 1, 2]
            }
        }
    
    def current_player(self):
        if self.done:
            return None
        return self.players[self.current_player_idx]
    
    def get_legal_actions(self, player):
        if self.done:
            return []
        return [0, 1, 2]
    
    def is_done(self):
        return self.done
    
    def step(self, action):
        self.step_count += 1
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        
        # End game after a few steps
        if self.step_count >= 9:
            self.done = True
        
        # Don't call reset()! Just return current observations or empty if done
        if not self.done:
            observations = {
                'player_0': {
                    'hole_cards': ['As', 'Kd'],
                    'community_cards': [],
                    'action_history': [],
                    'legal_actions': [0, 1, 2]
                },
                'player_1': {
                    'hole_cards': ['Qh', 'Jc'],
                    'community_cards': [],
                    'action_history': [],
                    'legal_actions': [0, 1, 2]
                },
                'player_2': {
                    'hole_cards': ['Ts', '9h'],
                    'community_cards': [],
                    'action_history': [],
                    'legal_actions': [0, 1, 2]
                }
            }
        else:
            observations = {}
            
        rewards = {'player_0': 1.0, 'player_1': -0.5, 'player_2': -0.5}
        done = self.done
        info = {}
        
        return observations, rewards, done, info
    
    def get_final_rewards(self):
        return {'player_0': 1.0, 'player_1': -0.5, 'player_2': -0.5}


class TestDeepCFRTrainer(unittest.TestCase):
    """Test cases for DeepCFRTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment
        self.mock_env = MockEnvironment()
        
        # Create transformer network
        self.network = InfoSetTransformer(
            d_model=64,
            n_heads=2,
            n_layers=1,
            n_actions=5,
            max_seq_len=100,
            dropout=0.0
        )
        
        # Create trainer
        self.trainer = DeepCFRTrainer(
            env=self.mock_env,
            network=self.network,
            buffer_size=1000,
            learning_rate=1e-3,
            batch_size=32,
            device="cpu"
        )
    
    def test_trainer_initialization(self):
        """Test DeepCFRTrainer initialization."""
        self.assertIsInstance(self.trainer.env, MockEnvironment)
        self.assertIsInstance(self.trainer.network, InfoSetTransformer)
        self.assertIsInstance(self.trainer.buffer, ReservoirBuffer)
        self.assertEqual(self.trainer.device, "cpu")
        self.assertEqual(self.trainer.batch_size, 32)
        self.assertEqual(self.trainer.iteration, 0)
        self.assertEqual(self.trainer.total_games, 0)
    
    def test_information_set_encoding(self):
        """Test information set encoding."""
        observation = {
            'hole_cards': ['As', 'Kd'],
            'community_cards': ['Qh', 'Jc', 'Ts'],
            'action_history': [1, 2, 1],
            'legal_actions': [0, 1, 2]
        }
        
        encoding = self.trainer._encode_information_set(observation, 'player_0')
        
        self.assertIsNotNone(encoding)
        self.assertIn('tokens', encoding)
        self.assertIn('attention_mask', encoding)
        
        # Check tensor properties
        tokens = encoding['tokens']
        mask = encoding['attention_mask']
        
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(tokens.shape[0], 1)  # Batch size 1
        self.assertEqual(mask.shape[0], 1)    # Batch size 1
        self.assertEqual(tokens.shape[1], mask.shape[1])  # Same sequence length
    
    def test_information_set_encoding_edge_cases(self):
        """Test information set encoding with edge cases."""
        # Empty observation
        empty_obs = {}
        encoding = self.trainer._encode_information_set(empty_obs, 'player_0')
        self.assertIsNotNone(encoding)
        
        # Observation with invalid cards
        invalid_obs = {
            'hole_cards': ['XX', 'YY'],
            'community_cards': ['ZZ'],
            'action_history': [99],
            'legal_actions': [0, 1]
        }
        encoding = self.trainer._encode_information_set(invalid_obs, 'player_0')
        self.assertIsNotNone(encoding)
    
    def test_strategy_generation(self):
        """Test strategy generation from network."""
        # Create mock encoding
        encoding = {
            'tokens': torch.randint(0, 61, (1, 10)),
            'attention_mask': torch.ones((1, 10), dtype=torch.bool)
        }
        legal_actions = [0, 1, 2]
        
        strategy = self.trainer._get_strategy(encoding, legal_actions)
        
        # Check strategy properties
        self.assertEqual(len(strategy), 5)
        self.assertAlmostEqual(np.sum(strategy), 1.0, places=5)
        self.assertTrue(np.all(strategy >= 0))
        
        # Check that illegal actions have zero probability
        for action in range(5):
            if action not in legal_actions:
                self.assertEqual(strategy[action], 0.0)
    
    def test_regret_matching(self):
        """Test regret matching algorithm."""
        # Test with positive regrets
        advantages = np.array([0.5, 0.3, 0.0, 0.2, 0.1])
        legal_actions = [0, 1, 2, 3, 4]
        
        strategy = self.trainer._regret_matching(advantages, legal_actions)
        
        self.assertAlmostEqual(np.sum(strategy), 1.0, places=5)
        self.assertTrue(np.all(strategy >= 0))
        
        # Higher advantages should get higher probability
        self.assertGreater(strategy[0], strategy[1])  # 0.5 > 0.3
        self.assertGreater(strategy[1], strategy[3])  # 0.3 > 0.2
        
        # Test with all negative regrets (should be uniform)
        negative_advantages = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        uniform_strategy = self.trainer._regret_matching(negative_advantages, legal_actions)
        
        expected_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            self.assertAlmostEqual(uniform_strategy[action], expected_prob, places=5)
        
        # Test with limited legal actions
        limited_legal = [1, 3]
        limited_strategy = self.trainer._regret_matching(advantages, limited_legal)
        
        for action in range(5):
            if action in limited_legal:
                self.assertGreater(limited_strategy[action], 0)
            else:
                self.assertEqual(limited_strategy[action], 0)
    
    def test_external_sampling_traverse(self):
        """Test external sampling game traversal."""
        experiences = self.trainer._external_sampling_traverse()
        
        # Should return a list (possibly empty)
        self.assertIsInstance(experiences, list)
        
        # If experiences are generated, check their structure
        for exp in experiences:
            self.assertIsInstance(exp, dict)
            self.assertIn('info_set', exp)
            self.assertIn('strategy', exp)
            self.assertIn('regrets', exp)
            self.assertIn('reward', exp)
            self.assertIn('legal_actions', exp)
    
    def test_experience_creation(self):
        """Test experience creation from game history."""
        game_history = [
            {
                'player': 'player_0',
                'info_set': {
                    'tokens': torch.randint(0, 61, (1, 5)),
                    'attention_mask': torch.ones((1, 5), dtype=torch.bool)
                },
                'strategy': np.array([0.5, 0.3, 0.2, 0.0, 0.0]),
                'action': 0,
                'legal_actions': [0, 1, 2]
            }
        ]
        final_rewards = {'player_0': 1.0, 'player_1': -0.5, 'player_2': -0.5}
        
        experiences = self.trainer._create_experiences(game_history, final_rewards)
        
        self.assertEqual(len(experiences), 1)
        exp = experiences[0]
        
        self.assertIn('info_set', exp)
        self.assertIn('strategy', exp)
        self.assertIn('regrets', exp)
        self.assertIn('reward', exp)
        self.assertIn('legal_actions', exp)
        
        self.assertEqual(exp['reward'], 1.0)
        self.assertEqual(len(exp['regrets']), 5)
    
    def test_training_iteration(self):
        """Test training iteration."""
        initial_iteration = self.trainer.iteration
        initial_games = self.trainer.total_games
        
        stats = self.trainer.train_iteration(num_traversals=5)
        
        # Check that iteration incremented
        self.assertEqual(self.trainer.iteration, initial_iteration + 1)
        self.assertGreater(self.trainer.total_games, initial_games)
        
        # Check stats structure
        self.assertIn('games_played', stats)
        self.assertIn('experiences_collected', stats)
        self.assertIn('network_updates', stats)
        self.assertIn('avg_loss', stats)
        
        self.assertEqual(stats['games_played'], 5)
        self.assertGreaterEqual(stats['experiences_collected'], 0)
    
    def test_network_update(self):
        """Test neural network update."""
        # Add some experiences to buffer first
        for _ in range(50):
            experience = {
                'info_set': {
                    'tokens': torch.randint(0, 61, (1, 5)),
                    'attention_mask': torch.ones((1, 5), dtype=torch.bool)
                },
                'strategy': np.random.uniform(0, 1, 5),
                'regrets': np.random.uniform(-1, 1, 5),
                'reward': np.random.uniform(-1, 1),
                'legal_actions': [0, 1, 2]
            }
            self.trainer.buffer.add(experience)
        
        # Test network update
        loss = self.trainer._update_network()
        
        # Should return a valid loss value
        if loss is not None:
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import tempfile
        
        # Run a few training steps to change state
        self.trainer.train_iteration(num_traversals=3)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save checkpoint
            self.trainer.save_checkpoint(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new trainer and load checkpoint
            new_network = InfoSetTransformer(
                d_model=64,
                n_heads=2,
                n_layers=1,
                n_actions=5,
                max_seq_len=100,
                dropout=0.0
            )
            new_trainer = DeepCFRTrainer(
                env=MockEnvironment(),
                network=new_network,
                buffer_size=1000,
                learning_rate=1e-3,
                batch_size=32,
                device="cpu"
            )
            
            # Load checkpoint
            new_trainer.load_checkpoint(temp_path)
            
            # Check that state was loaded
            self.assertEqual(new_trainer.iteration, self.trainer.iteration)
            self.assertEqual(new_trainer.total_games, self.trainer.total_games)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_exploitability_placeholder(self):
        """Test exploitability calculation (placeholder)."""
        exploitability = self.trainer.get_exploitability()
        self.assertIsInstance(exploitability, float)
        self.assertGreaterEqual(exploitability, 0.0)


if __name__ == '__main__':
    unittest.main()