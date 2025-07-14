"""
Kuhn Poker Environment

A simplified poker variant with:
- 3 cards: Jack (1), Queen (2), King (3)
- 2 players, each dealt 1 card
- Single betting round
- Perfect for testing CFR algorithms
"""

from .kuhn_poker import KuhnPokerEnv

__all__ = ['KuhnPokerEnv']