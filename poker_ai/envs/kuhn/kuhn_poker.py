"""
Kuhn Poker Environment Implementation

A two-player, simplified poker game with:
- 3 cards: Jack (1), Queen (2), King (3)  
- Each player receives 1 card, 1 card is dead
- Single betting round with Check/Bet actions
- Perfect information for testing CFR algorithms

Game Flow:
1. Each player antes 1 chip (pot = 2)
2. Player 0 acts first: Check or Bet
3. If P0 checks: P1 can Check (showdown) or Bet (P0 can Call/Fold)
4. If P0 bets: P1 can Call (showdown) or Fold
5. Showdown: Higher card wins the pot
"""

import random
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


class Action(Enum):
    """Actions available in Kuhn poker"""
    CHECK_CALL = 0  # Check when no bet, Call when facing bet
    BET_FOLD = 1    # Bet when no bet, Fold when facing bet


class KuhnPokerEnv:
    """
    Kuhn Poker Environment
    
    State representation:
    - Cards: [player0_card, player1_card] where cards are 1,2,3 (J,Q,K)
    - History: string of actions (C=check/call, B=bet, F=fold)
    - Current player: 0 or 1
    - Pot size: accumulated chips
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.cards = [1, 2, 3]  # Jack, Queen, King
        self.reset(seed)
        
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset the game and return initial state"""
        if seed is not None:
            random.seed(seed)
            
        # Deal cards randomly
        dealt_cards = random.sample(self.cards, 2)
        self.player_cards = dealt_cards  # [p0_card, p1_card]
        
        # Initialize game state
        self.history = ""
        self.current_player = 0
        self.pot = 2  # Both players ante 1 chip
        self.is_terminal_state = False
        self.payoffs = [0, 0]
        
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current game state"""
        return {
            'cards': self.player_cards.copy(),
            'history': self.history,
            'current_player': self.current_player,
            'pot': self.pot,
            'is_terminal': self.is_terminal_state,
            'legal_actions': self.get_legal_actions()
        }
    
    def get_legal_actions(self) -> List[int]:
        """Get legal actions for current player"""
        if self.is_terminal_state:
            return []
        
        # In Kuhn poker, both actions are always legal
        # The interpretation changes based on context:
        # - No previous bet: CHECK_CALL=check, BET_FOLD=bet  
        # - Facing bet: CHECK_CALL=call, BET_FOLD=fold
        return [Action.CHECK_CALL.value, Action.BET_FOLD.value]
    
    def step(self, action: int) -> Tuple[Dict, bool]:
        """
        Execute action and return (new_state, is_terminal)
        
        Args:
            action: Action.CHECK_CALL or Action.BET_FOLD
            
        Returns:
            new_state: Updated game state
            is_terminal: Whether game has ended
        """
        if self.is_terminal_state:
            raise ValueError("Game is already terminal")
            
        if action not in self.get_legal_actions():
            raise ValueError(f"Illegal action {action}")
        
        # Convert action to character and update history
        action_char = self._action_to_char(action)
        self.history += action_char
        
        # Check if game is now terminal
        if self._is_terminal_history(self.history):
            self.is_terminal_state = True
            self.payoffs = self._calculate_payoffs()
        else:
            # Switch to other player
            self.current_player = 1 - self.current_player
            
        return self.get_state(), self.is_terminal_state
    
    def _action_to_char(self, action: int) -> str:
        """Convert action to history character based on context"""
        if action == Action.CHECK_CALL.value:
            # Check if there's a bet to call
            if self.history.endswith('B'):
                return 'C'  # Call
            else:
                return 'C'  # Check (we use same char for simplicity)
        else:  # BET_FOLD
            # Check if there's a bet to fold to
            if self.history.endswith('B'):
                return 'F'  # Fold
            else:
                return 'B'  # Bet
    
    def _is_terminal_history(self, history: str) -> bool:
        """Check if action history represents terminal state"""
        terminal_patterns = [
            "CC",   # Both check
            "BC",   # Bet-call  
            "BF",   # Bet-fold
            "CBF",  # Check-bet-fold
            "CBC"   # Check-bet-call
        ]
        return history in terminal_patterns
    
    def _calculate_payoffs(self) -> List[int]:
        """Calculate final payoffs for both players"""
        history = self.history
        p0_card, p1_card = self.player_cards
        
        if history == "CC":
            # Both check - showdown with pot of 2
            winner = 0 if p0_card > p1_card else 1
            return [1, -1] if winner == 0 else [-1, 1]
            
        elif history == "BC":
            # Bet-call - showdown with pot of 4  
            winner = 0 if p0_card > p1_card else 1
            return [2, -2] if winner == 0 else [-2, 2]
            
        elif history == "BF":
            # Player 0 bet, Player 1 folded
            return [1, -1]
            
        elif history == "CBF":
            # Player 0 checked, Player 1 bet, Player 0 folded
            return [-1, 1]
            
        elif history == "CBC":
            # Check-bet-call - showdown with pot of 4
            winner = 0 if p0_card > p1_card else 1
            return [2, -2] if winner == 0 else [-2, 2]
            
        else:
            raise ValueError(f"Unknown terminal history: {history}")
    
    def get_payoff(self, player: int) -> int:
        """Get payoff for specified player"""
        if not self.is_terminal_state:
            return 0
        return self.payoffs[player]
    
    def get_info_set(self, player: int) -> str:
        """
        Get information set for specified player
        
        Information set format: "{card}/{history}"
        Examples: "1/", "2/C", "3/CB"
        """
        return f"{self.player_cards[player]}/{self.history}"
    
    def clone(self) -> 'KuhnPokerEnv':
        """Create a copy of current game state"""
        new_env = KuhnPokerEnv()
        new_env.player_cards = self.player_cards.copy()
        new_env.history = self.history
        new_env.current_player = self.current_player
        new_env.pot = self.pot
        new_env.is_terminal_state = self.is_terminal_state
        new_env.payoffs = self.payoffs.copy()
        return new_env
    
    def get_all_possible_deals(self) -> List[Tuple[int, int]]:
        """Get all possible card deals (6 total combinations)"""
        deals = []
        for p0_card in self.cards:
            for p1_card in self.cards:
                if p0_card != p1_card:  # Cards must be different
                    deals.append((p0_card, p1_card))
        return deals
    
    def set_cards(self, p0_card: int, p1_card: int):
        """Set specific cards for deterministic testing"""
        if p0_card == p1_card:
            raise ValueError("Players cannot have the same card")
        if p0_card not in self.cards or p1_card not in self.cards:
            raise ValueError("Invalid card values")
        self.player_cards = [p0_card, p1_card]
    
    def __str__(self) -> str:
        """String representation of game state"""
        if not self.is_terminal_state:
            return (f"KuhnPoker(cards={self.player_cards}, history='{self.history}', "
                   f"player={self.current_player}, pot={self.pot})")
        else:
            return (f"KuhnPoker(cards={self.player_cards}, history='{self.history}', "
                   f"TERMINAL, payoffs={self.payoffs})")


def get_action_name(action: int, context_history: str = "") -> str:
    """Get human-readable action name based on context"""
    if action == Action.CHECK_CALL.value:
        if context_history.endswith('B'):
            return "Call"
        else:
            return "Check"
    else:  # BET_FOLD
        if context_history.endswith('B'):
            return "Fold"
        else:
            return "Bet"


def simulate_random_game(seed: Optional[int] = None) -> KuhnPokerEnv:
    """Simulate a random game for testing"""
    env = KuhnPokerEnv(seed)
    
    while not env.is_terminal_state:
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        env.step(action)
        
    return env


if __name__ == "__main__":
    # Test the environment
    print("Testing Kuhn Poker Environment")
    print("=" * 40)
    
    # Test all possible card combinations
    env = KuhnPokerEnv()
    
    for p0_card in [1, 2, 3]:
        for p1_card in [1, 2, 3]:
            if p0_card != p1_card:
                print(f"\nTesting with cards P0={p0_card}, P1={p1_card}")
                env.reset()
                env.set_cards(p0_card, p1_card)
                
                # Test one path: Check-Bet-Call
                print(f"Initial: {env}")
                
                # P0 checks
                env.step(Action.CHECK_CALL.value)
                print(f"After P0 check: {env}")
                
                # P1 bets  
                env.step(Action.BET_FOLD.value)
                print(f"After P1 bet: {env}")
                
                # P0 calls
                env.step(Action.CHECK_CALL.value)
                print(f"Final: {env}")
                
                break
        break
    
    print("\n" + "=" * 40)
    print("Random game simulation:")
    random_game = simulate_random_game(42)
    print(random_game)