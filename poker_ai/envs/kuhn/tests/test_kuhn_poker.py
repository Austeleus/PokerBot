"""
Comprehensive tests for Kuhn Poker environment
"""

import pytest
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from poker_ai.envs.kuhn.kuhn_poker import KuhnPokerEnv, Action, get_action_name, simulate_random_game


class TestKuhnPokerEnvironment:
    """Test suite for Kuhn Poker environment"""
    
    def setup_method(self):
        """Setup test environment"""
        self.env = KuhnPokerEnv(seed=42)
    
    def test_initialization(self):
        """Test environment initialization"""
        state = self.env.get_state()
        
        assert len(state['cards']) == 2
        assert all(card in [1, 2, 3] for card in state['cards'])
        assert state['cards'][0] != state['cards'][1]  # Different cards
        assert state['history'] == ""
        assert state['current_player'] == 0
        assert state['pot'] == 2
        assert not state['is_terminal']
        assert len(state['legal_actions']) == 2
    
    def test_card_dealing(self):
        """Test that cards are dealt correctly"""
        # Test multiple resets to ensure randomness works
        dealt_combinations = set()
        
        for _ in range(20):
            self.env.reset()
            cards = tuple(self.env.get_state()['cards'])
            dealt_combinations.add(cards)
        
        # Should see multiple different combinations
        assert len(dealt_combinations) > 1
        
        # All combinations should be valid
        for combo in dealt_combinations:
            assert len(combo) == 2
            assert combo[0] in [1, 2, 3]
            assert combo[1] in [1, 2, 3]
            assert combo[0] != combo[1]
    
    def test_set_cards(self):
        """Test setting specific cards"""
        self.env.set_cards(1, 3)
        assert self.env.player_cards == [1, 3]
        
        # Test invalid cards
        with pytest.raises(ValueError):
            self.env.set_cards(1, 1)  # Same card
            
        with pytest.raises(ValueError):
            self.env.set_cards(0, 2)  # Invalid card
    
    def test_legal_actions(self):
        """Test legal actions"""
        legal_actions = self.env.get_legal_actions()
        assert legal_actions == [0, 1]  # CHECK_CALL and BET_FOLD
        
        # Test when terminal
        self.env.is_terminal_state = True
        assert self.env.get_legal_actions() == []
    
    def test_info_sets(self):
        """Test information set generation"""
        self.env.set_cards(2, 3)
        
        # Initial info sets
        assert self.env.get_info_set(0) == "2/"
        assert self.env.get_info_set(1) == "3/"
        
        # After some actions
        self.env.step(Action.CHECK_CALL.value)  # P0 checks
        assert self.env.get_info_set(0) == "2/C"
        assert self.env.get_info_set(1) == "3/C"
        
        self.env.step(Action.BET_FOLD.value)  # P1 bets
        assert self.env.get_info_set(0) == "2/CB"
        assert self.env.get_info_set(1) == "3/CB"
    
    def test_game_flow_both_check(self):
        """Test game flow: both players check"""
        self.env.set_cards(3, 1)  # King vs Jack
        
        # P0 checks
        state, is_terminal = self.env.step(Action.CHECK_CALL.value)
        assert not is_terminal
        assert state['history'] == "C"
        assert state['current_player'] == 1
        
        # P1 checks  
        state, is_terminal = self.env.step(Action.CHECK_CALL.value)
        assert is_terminal
        assert state['history'] == "CC"
        
        # King wins
        assert self.env.get_payoff(0) == 1
        assert self.env.get_payoff(1) == -1
    
    def test_game_flow_bet_call(self):
        """Test game flow: bet and call"""
        self.env.set_cards(1, 3)  # Jack vs King
        
        # P0 bets
        state, is_terminal = self.env.step(Action.BET_FOLD.value)
        assert not is_terminal
        assert state['history'] == "B"
        assert state['current_player'] == 1
        
        # P1 calls
        state, is_terminal = self.env.step(Action.CHECK_CALL.value)
        assert is_terminal
        assert state['history'] == "BC"
        
        # King wins bigger pot
        assert self.env.get_payoff(0) == -2
        assert self.env.get_payoff(1) == 2
    
    def test_game_flow_bet_fold(self):
        """Test game flow: bet and fold"""
        self.env.set_cards(1, 2)  # Jack vs Queen
        
        # P0 bets
        state, is_terminal = self.env.step(Action.BET_FOLD.value)
        assert not is_terminal
        
        # P1 folds
        state, is_terminal = self.env.step(Action.BET_FOLD.value)
        assert is_terminal
        assert state['history'] == "BF"
        
        # P0 wins by fold
        assert self.env.get_payoff(0) == 1
        assert self.env.get_payoff(1) == -1
    
    def test_game_flow_check_bet_fold(self):
        """Test game flow: check, bet, fold"""
        self.env.set_cards(1, 3)  # Jack vs King
        
        # P0 checks
        self.env.step(Action.CHECK_CALL.value)
        
        # P1 bets
        self.env.step(Action.BET_FOLD.value)
        
        # P0 folds
        state, is_terminal = self.env.step(Action.BET_FOLD.value)
        assert is_terminal
        assert state['history'] == "CBF"
        
        # P1 wins by fold
        assert self.env.get_payoff(0) == -1
        assert self.env.get_payoff(1) == 1
    
    def test_game_flow_check_bet_call(self):
        """Test game flow: check, bet, call"""
        self.env.set_cards(3, 1)  # King vs Jack
        
        # P0 checks
        self.env.step(Action.CHECK_CALL.value)
        
        # P1 bets
        self.env.step(Action.BET_FOLD.value)
        
        # P0 calls
        state, is_terminal = self.env.step(Action.CHECK_CALL.value)
        assert is_terminal
        assert state['history'] == "CBC"
        
        # King wins bigger pot
        assert self.env.get_payoff(0) == 2
        assert self.env.get_payoff(1) == -2
    
    def test_all_terminal_patterns(self):
        """Test all possible terminal game patterns"""
        terminal_patterns = ["CC", "BC", "BF", "CBF", "CBC"]
        
        for pattern in terminal_patterns:
            env = KuhnPokerEnv()
            env.set_cards(2, 3)  # Queen vs King
            
            assert env._is_terminal_history(pattern)
            
            # Non-terminal patterns
            assert not env._is_terminal_history("")
            assert not env._is_terminal_history("C")
            assert not env._is_terminal_history("B")
            assert not env._is_terminal_history("CB")
    
    def test_payoff_calculations(self):
        """Test payoff calculations for all scenarios"""
        # Test all card combinations with CC (both check)
        for p0_card in [1, 2, 3]:
            for p1_card in [1, 2, 3]:
                if p0_card != p1_card:
                    env = KuhnPokerEnv()
                    env.set_cards(p0_card, p1_card)
                    env.history = "CC"
                    env.is_terminal_state = True
                    payoffs = env._calculate_payoffs()
                    
                    if p0_card > p1_card:
                        assert payoffs == [1, -1]
                    else:
                        assert payoffs == [-1, 1]
    
    def test_clone(self):
        """Test environment cloning"""
        self.env.set_cards(2, 3)
        self.env.step(Action.CHECK_CALL.value)
        
        cloned = self.env.clone()
        
        assert cloned.player_cards == self.env.player_cards
        assert cloned.history == self.env.history
        assert cloned.current_player == self.env.current_player
        assert cloned.pot == self.env.pot
        assert cloned.is_terminal_state == self.env.is_terminal_state
        
        # Ensure they're independent
        cloned.step(Action.BET_FOLD.value)
        assert cloned.history != self.env.history
    
    def test_get_all_possible_deals(self):
        """Test getting all possible card deals"""
        deals = self.env.get_all_possible_deals()
        
        assert len(deals) == 6  # 3 * 2 = 6 possible combinations
        
        expected_deals = [
            (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)
        ]
        
        for deal in expected_deals:
            assert deal in deals
    
    def test_action_names(self):
        """Test action name generation"""
        assert get_action_name(Action.CHECK_CALL.value, "") == "Check"
        assert get_action_name(Action.CHECK_CALL.value, "B") == "Call"
        assert get_action_name(Action.BET_FOLD.value, "") == "Bet"
        assert get_action_name(Action.BET_FOLD.value, "B") == "Fold"
    
    def test_random_game_simulation(self):
        """Test random game simulation"""
        game = simulate_random_game(seed=123)
        
        assert game.is_terminal_state
        assert len(game.history) >= 2  # At least 2 actions
        assert game.history in ["CC", "BC", "BF", "CBF", "CBC"]
        assert sum(game.payoffs) == 0  # Zero-sum game
    
    def test_error_handling(self):
        """Test error handling"""
        # Test stepping when terminal
        self.env.is_terminal_state = True
        with pytest.raises(ValueError):
            self.env.step(Action.CHECK_CALL.value)
        
        # Reset for next test
        self.env.reset()
        
        # Test invalid action (this shouldn't happen in practice)
        with pytest.raises(ValueError):
            self.env.step(999)
    
    def test_deterministic_with_seed(self):
        """Test deterministic behavior with seed"""
        env1 = KuhnPokerEnv(seed=42)
        env2 = KuhnPokerEnv(seed=42)
        
        state1 = env1.get_state()
        state2 = env2.get_state()
        
        assert state1['cards'] == state2['cards']
    
    def test_string_representation(self):
        """Test string representation"""
        self.env.set_cards(1, 2)
        
        # Non-terminal
        str_repr = str(self.env)
        assert "KuhnPoker" in str_repr
        assert "cards=[1, 2]" in str_repr
        assert "history=''" in str_repr
        
        # Terminal
        self.env.step(Action.BET_FOLD.value)
        self.env.step(Action.BET_FOLD.value)
        str_repr = str(self.env)
        assert "TERMINAL" in str_repr
        assert "payoffs=" in str_repr


if __name__ == "__main__":
    # Run a quick test
    test_suite = TestKuhnPokerEnvironment()
    test_suite.setup_method()
    
    print("Running Kuhn Poker Environment Tests...")
    
    # Run a few key tests
    test_suite.test_initialization()
    test_suite.test_game_flow_both_check()
    test_suite.test_game_flow_bet_call()
    test_suite.test_payoff_calculations()
    
    print("All tests passed! ✅")
    
    # Demonstrate the environment
    print("\nDemonstrating Kuhn Poker Environment:")
    print("=" * 50)
    
    env = KuhnPokerEnv(seed=42)
    print(f"Initial state: {env}")
    
    # Play through a game
    while not env.is_terminal_state:
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        action_name = get_action_name(action, env.history)
        
        print(f"Player {env.current_player} chooses: {action_name}")
        env.step(action)
        print(f"After action: {env}")
    
    print(f"\nFinal payoffs: P0={env.get_payoff(0)}, P1={env.get_payoff(1)}")