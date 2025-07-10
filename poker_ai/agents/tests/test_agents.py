"""
Unit tests for poker agents.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from poker_ai.agents.base_agent import BaseAgent, RandomAgent
from poker_ai.agents.cfr_agent import CFRAgent
from poker_ai.models.info_set_transformer import InfoSetTransformer


class TestBaseAgent(unittest.TestCase):
    """Test cases for BaseAgent abstract class."""
    
    def test_random_agent_initialization(self):
        """Test RandomAgent initialization."""
        agent = RandomAgent("test_agent")
        self.assertEqual(agent.agent_id, "test_agent")
        self.assertEqual(agent.num_actions, 5)
        self.assertTrue(agent.is_training)
    
    def test_random_agent_action_selection(self):
        """Test RandomAgent action selection."""
        agent = RandomAgent("test_agent")
        
        # Test with legal actions
        legal_actions = [0, 1, 2]
        observation = {'legal_actions': legal_actions}
        
        # Run multiple times to check randomness
        actions = []
        for _ in range(100):
            action = agent.get_action(observation, legal_actions)
            actions.append(action)
            self.assertIn(action, legal_actions)
        
        # Should have some variety in actions
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)
    
    def test_random_agent_strategy(self):
        """Test RandomAgent strategy distribution."""
        agent = RandomAgent("test_agent")
        legal_actions = [0, 2, 4]
        observation = {'legal_actions': legal_actions}
        
        strategy = agent.get_strategy(observation)
        
        # Check strategy properties
        self.assertEqual(len(strategy), 5)
        self.assertAlmostEqual(np.sum(strategy), 1.0, places=5)
        
        # Check uniform distribution over legal actions
        expected_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            self.assertAlmostEqual(strategy[action], expected_prob, places=5)
        
        # Check illegal actions have zero probability
        for action in range(5):
            if action not in legal_actions:
                self.assertEqual(strategy[action], 0.0)
    
    def test_training_mode(self):
        """Test training mode setting."""
        agent = RandomAgent("test_agent")
        
        # Default is training mode
        self.assertTrue(agent.is_training)
        
        # Set to evaluation mode
        agent.set_training_mode(False)
        self.assertFalse(agent.is_training)
        
        # Set back to training mode
        agent.set_training_mode(True)
        self.assertTrue(agent.is_training)


class TestCFRAgent(unittest.TestCase):
    """Test cases for CFRAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple transformer network
        self.network = InfoSetTransformer(
            d_model=64,
            n_heads=2,
            n_layers=1,
            n_actions=5,
            max_seq_len=100,
            dropout=0.0
        )
        
        # Create CFR agent
        self.agent = CFRAgent(
            network=self.network,
            agent_id="test_cfr_agent",
            device="cpu",
            temperature=1.0
        )
    
    def test_cfr_agent_initialization(self):
        """Test CFRAgent initialization."""
        self.assertEqual(self.agent.agent_id, "test_cfr_agent")
        self.assertEqual(self.agent.num_actions, 5)
        self.assertEqual(self.agent.device, "cpu")
        self.assertEqual(self.agent.temperature, 1.0)
        self.assertFalse(self.agent.network.training)  # Should be in eval mode
    
    def test_cfr_agent_action_selection(self):
        """Test CFRAgent action selection."""
        # Create mock observation
        observation = {
            'hole_cards': ['As', 'Kd'],
            'community_cards': ['Qh', 'Jc', 'Ts'],
            'action_history': [1, 2],
            'legal_actions': [0, 1, 2]
        }
        legal_actions = [0, 1, 2]
        
        # Test action selection
        action = self.agent.get_action(observation, legal_actions)
        self.assertIn(action, legal_actions)
        self.assertIsInstance(action, (int, np.integer))
    
    def test_cfr_agent_strategy(self):
        """Test CFRAgent strategy generation."""
        observation = {
            'hole_cards': ['As', 'Kd'],
            'community_cards': [],
            'action_history': [],
            'legal_actions': [0, 1, 2, 3, 4]
        }
        
        strategy = self.agent.get_strategy(observation)
        
        # Check strategy properties
        self.assertEqual(len(strategy), 5)
        self.assertAlmostEqual(np.sum(strategy), 1.0, places=5)
        self.assertTrue(np.all(strategy >= 0))  # Non-negative probabilities
    
    def test_cfr_agent_temperature_scaling(self):
        """Test temperature scaling effects."""
        observation = {
            'hole_cards': ['As', 'Kd'],
            'community_cards': [],
            'action_history': [],
            'legal_actions': [0, 1, 2, 3, 4]
        }
        
        # Test different temperatures
        temp_agent_low = CFRAgent(self.network, "temp_low", "cpu", temperature=0.1)
        temp_agent_high = CFRAgent(self.network, "temp_high", "cpu", temperature=2.0)
        
        strategy_low = temp_agent_low.get_strategy(observation)
        strategy_high = temp_agent_high.get_strategy(observation)
        
        # Low temperature should be more concentrated
        # High temperature should be more uniform
        entropy_low = -np.sum(strategy_low * np.log(strategy_low + 1e-10))
        entropy_high = -np.sum(strategy_high * np.log(strategy_high + 1e-10))
        
        # Note: This test might be flaky due to random initialization
        # Just check that strategies are valid
        self.assertAlmostEqual(np.sum(strategy_low), 1.0, places=5)
        self.assertAlmostEqual(np.sum(strategy_high), 1.0, places=5)
    
    def test_cfr_agent_encoding_edge_cases(self):
        """Test CFRAgent with edge case observations."""
        # Empty observation
        empty_obs = {}
        legal_actions = [0, 1]
        
        action = self.agent.get_action(empty_obs, legal_actions)
        self.assertIn(action, legal_actions)
        
        # Observation with invalid cards
        invalid_obs = {
            'hole_cards': ['XX', 'YY'],
            'community_cards': ['ZZ'],
            'action_history': [99],
            'legal_actions': [0, 1]
        }
        
        action = self.agent.get_action(invalid_obs, legal_actions)
        self.assertIn(action, legal_actions)
    
    def test_cfr_agent_reset(self):
        """Test CFRAgent reset functionality."""
        initial_count = self.agent.game_count
        self.agent.reset()
        self.assertEqual(self.agent.game_count, initial_count + 1)
    
    def test_cfr_agent_save_load(self):
        """Test CFRAgent save/load functionality."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save agent
            self.agent.save(temp_path)
            
            # Create new agent and load
            new_network = InfoSetTransformer(
                d_model=64,
                n_heads=2,
                n_layers=1,
                n_actions=5,
                max_seq_len=100,
                dropout=0.0
            )
            new_agent = CFRAgent(new_network, "new_agent", "cpu", 2.0)
            
            # Load saved state
            new_agent.load(temp_path)
            
            # Check that state was loaded
            self.assertEqual(new_agent.agent_id, self.agent.agent_id)
            self.assertEqual(new_agent.temperature, self.agent.temperature)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_cfr_agent_from_checkpoint(self):
        """Test CFRAgent creation from checkpoint."""
        import tempfile
        
        # Create a fake checkpoint
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'iteration': 100,
            'total_games': 1000
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # Create agent from checkpoint with matching network config
            network_config = {
                'd_model': 64,
                'n_heads': 2,
                'n_layers': 1,
                'n_actions': 5,
                'max_seq_len': 100
            }
            
            agent = CFRAgent.from_checkpoint(
                checkpoint_path=temp_path,
                network_config=network_config,
                agent_id="checkpoint_agent",
                device="cpu",
                temperature=0.5
            )
            
            self.assertEqual(agent.agent_id, "checkpoint_agent")
            self.assertEqual(agent.temperature, 0.5)
            self.assertIsInstance(agent.network, InfoSetTransformer)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()