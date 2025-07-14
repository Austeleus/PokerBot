#!/usr/bin/env python3
"""
Simple test script for Kuhn Poker environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_ai.envs.kuhn.kuhn_poker import KuhnPokerEnv, Action, get_action_name, simulate_random_game


def test_kuhn_poker_basic():
    """Test basic functionality of Kuhn poker environment"""
    print("Testing Kuhn Poker Environment")
    print("=" * 50)
    
    # Test 1: Initialization
    env = KuhnPokerEnv(seed=42)
    state = env.get_state()
    
    print(f"✅ Initialization: {state['cards']}, history='{state['history']}'")
    assert len(state['cards']) == 2
    assert state['cards'][0] != state['cards'][1]
    assert state['history'] == ""
    assert state['current_player'] == 0
    
    # Test 2: Game flow - both check
    env.reset()
    env.set_cards(3, 1)  # King vs Jack
    print(f"Initial: {env}")
    
    env.step(Action.CHECK_CALL.value)  # P0 checks
    print(f"After P0 check: {env}")
    
    env.step(Action.CHECK_CALL.value)  # P1 checks
    print(f"After P1 check: {env}")
    
    assert env.is_terminal_state
    assert env.get_payoff(0) == 1  # King wins
    assert env.get_payoff(1) == -1
    print("✅ Both check scenario works")
    
    # Test 3: Game flow - bet and call
    env.reset()
    env.set_cards(1, 3)  # Jack vs King
    print(f"\nBet-call scenario:")
    print(f"Initial: {env}")
    
    env.step(Action.BET_FOLD.value)  # P0 bets
    print(f"After P0 bet: {env}")
    
    env.step(Action.CHECK_CALL.value)  # P1 calls
    print(f"After P1 call: {env}")
    
    assert env.is_terminal_state
    assert env.get_payoff(0) == -2  # Jack loses bigger pot
    assert env.get_payoff(1) == 2
    print("✅ Bet-call scenario works")
    
    # Test 4: Game flow - bet and fold
    env.reset()
    env.set_cards(1, 2)  # Jack vs Queen
    print(f"\nBet-fold scenario:")
    print(f"Initial: {env}")
    
    env.step(Action.BET_FOLD.value)  # P0 bets
    print(f"After P0 bet: {env}")
    
    env.step(Action.BET_FOLD.value)  # P1 folds
    print(f"After P1 fold: {env}")
    
    assert env.is_terminal_state
    assert env.get_payoff(0) == 1  # P0 wins by fold
    assert env.get_payoff(1) == -1
    print("✅ Bet-fold scenario works")
    
    # Test 5: Information sets
    env.reset()
    env.set_cards(2, 3)  # Queen vs King
    
    assert env.get_info_set(0) == "2/"
    assert env.get_info_set(1) == "3/"
    
    env.step(Action.CHECK_CALL.value)  # P0 checks
    assert env.get_info_set(0) == "2/C"
    assert env.get_info_set(1) == "3/C"
    
    env.step(Action.BET_FOLD.value)  # P1 bets
    assert env.get_info_set(0) == "2/CB"
    assert env.get_info_set(1) == "3/CB"
    print("✅ Information sets work correctly")
    
    # Test 6: All possible deals
    deals = env.get_all_possible_deals()
    assert len(deals) == 6
    expected_deals = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    for deal in expected_deals:
        assert deal in deals
    print("✅ All possible deals generated correctly")
    
    # Test 7: Random simulation
    for i in range(5):
        game = simulate_random_game(seed=i)
        assert game.is_terminal_state
        assert game.history in ["CC", "BC", "BF", "CBF", "CBC"]
        assert sum(game.payoffs) == 0  # Zero-sum
    print("✅ Random simulations work")
    
    print("\n🎉 All tests passed!")


def demonstrate_all_game_paths():
    """Demonstrate all possible game paths in Kuhn poker"""
    print("\nDemonstrating All Possible Game Paths")
    print("=" * 50)
    
    # All terminal patterns with example outcomes
    scenarios = [
        ("CC", "Both players check"),
        ("BC", "Player 0 bets, Player 1 calls"),
        ("BF", "Player 0 bets, Player 1 folds"),
        ("CBF", "Player 0 checks, Player 1 bets, Player 0 folds"),
        ("CBC", "Player 0 checks, Player 1 bets, Player 0 calls")
    ]
    
    for pattern, description in scenarios:
        print(f"\nScenario: {description} (History: {pattern})")
        
        env = KuhnPokerEnv()
        env.set_cards(2, 3)  # Queen vs King
        
        # Simulate the action sequence
        for action_char in pattern:
            if action_char == 'C':
                if env.history.endswith('B'):  # Call
                    action = Action.CHECK_CALL.value
                    action_name = "Call"
                else:  # Check
                    action = Action.CHECK_CALL.value
                    action_name = "Check"
            elif action_char == 'B':
                action = Action.BET_FOLD.value
                action_name = "Bet"
            elif action_char == 'F':
                action = Action.BET_FOLD.value
                action_name = "Fold"
            
            player = env.current_player
            print(f"  Player {player} {action_name}s")
            env.step(action)
        
        print(f"  Final: {env}")
        print(f"  Payoffs: P0={env.get_payoff(0)}, P1={env.get_payoff(1)}")


if __name__ == "__main__":
    test_kuhn_poker_basic()
    demonstrate_all_game_paths()
    
    print("\n" + "=" * 50)
    print("Kuhn Poker Environment is ready for MCCFR implementation! 🚀")