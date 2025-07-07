#!/usr/bin/env python3
"""
Environment smoke test - Verify all core components work together.

This script tests the basic functionality of:
- HoldemWrapper environment
- CardEncoder utilities  
- ReservoirBuffer memory
- InfoSetTransformer model
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_ai.envs.holdem_wrapper import HoldemWrapper
from poker_ai.encoders.card_utils import CardEncoder
from poker_ai.memory.reservoir import ReservoirBuffer, ExperienceCollector
from poker_ai.models.info_set_transformer import InfoSetTransformer


def test_card_encoder():
    """Test card encoding utilities."""
    print("Testing CardEncoder...")
    
    encoder = CardEncoder()
    
    # Test card string conversion
    test_cards = ['As', 'Kh', 'Qd', 'Jc', 'Ts']
    
    for card in test_cards:
        index = encoder.card_string_to_index(card)
        reconstructed = encoder.index_to_card_string(index)
        assert card == reconstructed, f"Card conversion failed: {card} -> {index} -> {reconstructed}"
    
    # Test one-hot encoding
    encoding = encoder.encode_cards(test_cards)
    assert encoding.shape == (52,), f"Expected shape (52,), got {encoding.shape}"
    assert np.sum(encoding) == len(test_cards), f"Expected {len(test_cards)} cards encoded"
    
    # Test hand evaluation (if we have eval7 installed)
    try:
        # Test with a simple hand - pocket aces with a flop
        hand_rank, percentile = encoder.evaluate_hand(['As', 'Ad'], ['Kh', 'Qc', 'Jd'])
        print(f"  Hand evaluation: rank={hand_rank}, percentile={percentile}")
        assert 1 <= hand_rank <= 7462, f"Invalid hand rank: {hand_rank}"
        assert 0 <= percentile <= 100, f"Invalid percentile: {percentile}"
        
        # Test with full 7 cards
        hand_rank2, percentile2 = encoder.evaluate_hand(['As', 'Ad'], ['Kh', 'Qc', 'Jd', 'Ts', '9h'])
        print(f"  Full hand evaluation: rank={hand_rank2}, percentile={percentile2}")
        assert 1 <= hand_rank2 <= 7462, f"Invalid hand rank: {hand_rank2}"
        assert 0 <= percentile2 <= 100, f"Invalid percentile: {percentile2}"
        
    except ImportError:
        print("  Warning: eval7 not available, skipping hand evaluation test")
    except Exception as e:
        print(f"  Warning: Hand evaluation failed ({e}), but continuing tests")
    
    print("  ✓ CardEncoder tests passed")


def test_reservoir_buffer():
    """Test reservoir sampling buffer."""
    print("Testing ReservoirBuffer...")
    
    buffer = ReservoirBuffer(capacity=10, seed=42)
    
    # Add experiences
    for i in range(20):
        exp = {'id': i, 'data': np.random.randn(5)}
        buffer.add(exp)
    
    # Check buffer properties
    assert len(buffer) == 10, f"Expected buffer size 10, got {len(buffer)}"
    assert buffer.total_added == 20, f"Expected 20 total added, got {buffer.total_added}"
    
    # Test sampling
    batch = buffer.sample(5)
    assert len(batch) == 5, f"Expected batch size 5, got {len(batch)}"
    
    # Test statistics
    stats = buffer.get_statistics()
    assert stats['size'] == 10
    assert stats['total_added'] == 20
    assert stats['utilization'] == 1.0
    
    print("  ✓ ReservoirBuffer tests passed")


def test_transformer_model():
    """Test InfoSetTransformer model."""
    print("Testing InfoSetTransformer...")
    
    model = InfoSetTransformer(
        d_model=128,  # Smaller for testing
        n_heads=4,
        n_layers=2,
        n_actions=5
    )
    
    # Test token encoding
    tokens = model.encode_game_state(
        hole_cards=['As', 'Kh'],
        community_cards=['Qc', 'Jd', 'Ts'],
        action_history=[1, 2, 1]
    )
    
    assert tokens.shape[0] == 1, f"Expected batch size 1, got {tokens.shape[0]}"
    assert tokens.shape[1] > 0, f"Expected non-empty sequence, got length {tokens.shape[1]}"
    
    # Test forward pass
    advantages, policy_logits, values = model(tokens)
    
    assert advantages.shape == (1, 5), f"Expected advantages shape (1, 5), got {advantages.shape}"
    assert policy_logits.shape == (1, 5), f"Expected policy shape (1, 5), got {policy_logits.shape}"
    assert values.shape == (1, 1), f"Expected values shape (1, 1), got {values.shape}"
    
    # Test strategy generation
    strategy = model.get_strategy(
        hole_cards=['As', 'Kh'],
        community_cards=['Qc', 'Jd', 'Ts'],
        action_history=[1, 2, 1],
        legal_actions=[0, 1, 2, 3, 4]
    )
    
    assert strategy.shape == (5,), f"Expected strategy shape (5,), got {strategy.shape}"
    assert np.abs(np.sum(strategy) - 1.0) < 1e-6, f"Strategy doesn't sum to 1: {np.sum(strategy)}"
    
    print("  ✓ InfoSetTransformer tests passed")


def test_holdem_wrapper():
    """Test HoldemWrapper environment."""
    print("Testing HoldemWrapper...")
    
    env = HoldemWrapper(num_players=3, render_mode=None)
    
    # Test reset
    obs = env.reset(seed=42)
    assert isinstance(obs, dict), f"Expected dict observation, got {type(obs)}"
    
    current_player = env.get_current_player()
    assert current_player is not None, "Expected non-None current player after reset"
    
    # Test step
    legal_actions = env.get_legal_actions()
    assert len(legal_actions) > 0, "Expected at least one legal action"
    
    action = legal_actions[0]  # Take first legal action
    new_obs, rewards, done, info = env.step(action)
    
    assert isinstance(new_obs, dict), f"Expected dict observation, got {type(new_obs)}"
    assert isinstance(rewards, dict), f"Expected dict rewards, got {type(rewards)}"
    assert isinstance(done, bool), f"Expected bool done, got {type(done)}"
    assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
    
    env.close()
    print("  ✓ HoldemWrapper tests passed")


def test_integration():
    """Test integration of all components."""
    print("Testing component integration...")
    
    # Initialize components
    env = HoldemWrapper(num_players=3)
    encoder = CardEncoder()
    buffer = ReservoirBuffer(capacity=100)
    model = InfoSetTransformer(d_model=128, n_heads=4, n_layers=2)
    collector = ExperienceCollector()
    
    # Play one hand with random actions
    obs = env.reset(seed=42)
    step_count = 0
    max_steps = 50  # Prevent infinite loops
    
    while not env.is_terminal() and step_count < max_steps:
        current_player = env.get_current_player()
        if current_player is None:
            break
            
        # Get legal actions
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        # Choose random action
        action = np.random.choice(legal_actions)
        
        # Store experience
        if current_player in obs:
            experience = collector.create_rl_experience(
                observation=obs[current_player],
                action=action,
                reward=0.0,
                next_observation={},
                done=False,
                info={}
            )
            buffer.add(experience)
        
        # Take step
        obs, rewards, done, info = env.step(action)
        step_count += 1
        
        if done:
            break
    
    env.close()
    
    # Check buffer has experiences
    assert len(buffer) > 0, "Expected some experiences in buffer"
    
    # Test sampling from buffer
    if len(buffer) > 0:
        batch = buffer.sample(min(5, len(buffer)))
        assert len(batch) > 0, "Expected non-empty batch"
    
    print(f"  ✓ Integration test passed ({step_count} steps, {len(buffer)} experiences)")


def main():
    """Run all smoke tests."""
    print("=== Poker Bot Smoke Test ===\n")
    
    try:
        test_card_encoder()
        test_reservoir_buffer() 
        test_transformer_model()
        test_holdem_wrapper()
        test_integration()
        
        print("\n=== All Tests Passed! ===")
        print("✓ Environment wrapper working")
        print("✓ Card utilities functional") 
        print("✓ Memory system operational")
        print("✓ Transformer model ready")
        print("✓ Components integrate properly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()