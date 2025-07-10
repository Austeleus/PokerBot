"""
Information Set Manager for MCCFR

This module handles information set abstraction, key generation, and regret storage
separate from the neural network. This is crucial for proper MCCFR implementation.
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pickle
import logging


class InfoSetManager:
    """
    Manages information sets, keys, and external regret storage for MCCFR.
    
    This class is responsible for:
    1. Generating consistent information set keys
    2. Storing and retrieving regrets externally (not in network)
    3. Strategy averaging across iterations
    4. Information set abstraction for scalability
    """
    
    def __init__(self, 
                 n_actions: int = 5,
                 regret_decay: float = 0.99,
                 strategy_decay: float = 0.999,
                 use_cfr_plus: bool = True,
                 regret_clip: float = 10.0):
        """
        Initialize information set manager.
        
        Args:
            n_actions: Number of possible actions
            regret_decay: Decay factor for regrets  
            strategy_decay: Decay factor for strategy averaging
            use_cfr_plus: Whether to use CFR+ enhancements
            regret_clip: Maximum absolute regret value
        """
        self.n_actions = n_actions
        self.regret_decay = regret_decay
        self.strategy_decay = strategy_decay
        self.use_cfr_plus = use_cfr_plus
        self.regret_clip = regret_clip
        
        # External regret storage (separate from network)
        self.regrets = defaultdict(lambda: np.zeros(n_actions))
        self.cumulative_regrets = defaultdict(lambda: np.zeros(n_actions))
        
        # Strategy storage and averaging
        self.strategies = defaultdict(lambda: np.ones(n_actions) / n_actions)
        self.cumulative_strategies = defaultdict(lambda: np.zeros(n_actions))
        self.strategy_counts = defaultdict(int)
        
        # Information set statistics
        self.info_set_counts = defaultdict(int)
        self.total_iterations = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def generate_info_set_key(self, 
                             hole_cards: List[str],
                             community_cards: List[str], 
                             action_history: List[int],
                             position: int,
                             betting_round: int,
                             stack_size: float,
                             pot_size: float) -> str:
        """
        Generate deterministic information set key from game state.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            action_history: Sequence of actions in current round
            position: Player position (0-2 for 3-player)
            betting_round: Current betting round (0=preflop, 1=flop, etc.)
            stack_size: Player's stack size
            pot_size: Current pot size
            
        Returns:
            Unique string key for this information set
        """
        try:
            # Create abstracted representation
            abstracted_state = self._abstract_game_state(
                hole_cards, community_cards, action_history,
                position, betting_round, stack_size, pot_size
            )
            
            # Generate hash from abstracted state
            state_str = str(abstracted_state)
            key = hashlib.md5(state_str.encode()).hexdigest()[:16]
            
            return key
            
        except Exception as e:
            self.logger.warning(f"Failed to generate info set key: {e}")
            # Fallback to simple key
            return f"fallback_{position}_{betting_round}_{len(action_history)}"
    
    def _abstract_game_state(self,
                           hole_cards: List[str],
                           community_cards: List[str],
                           action_history: List[int],
                           position: int,
                           betting_round: int,
                           stack_size: float,
                           pot_size: float) -> Tuple:
        """
        Create abstracted representation of game state for key generation.
        
        Returns:
            Tuple representing abstracted game state
        """
        # Abstract hole cards (use hand strength bucket instead of exact cards)
        hand_strength_bucket = self._get_hand_strength_bucket(hole_cards, community_cards)
        
        # Abstract community cards (use texture instead of exact cards)
        board_texture = self._get_board_texture(community_cards)
        
        # Abstract action history (compress to essential patterns)
        action_pattern = self._compress_action_history(action_history)
        
        # Abstract stack and pot sizes (use ratio buckets)
        stack_pot_ratio = self._get_stack_pot_bucket(stack_size, pot_size)
        
        return (
            hand_strength_bucket,
            board_texture,
            action_pattern,
            position,
            betting_round,
            stack_pot_ratio
        )
    
    def _get_hand_strength_bucket(self, hole_cards: List[str], community_cards: List[str]) -> int:
        """Get hand strength bucket (0-9, with 9 being strongest)."""
        try:
            if not hole_cards or len(hole_cards) < 2:
                return 5  # Middle bucket
            
            # Simple hand strength estimation
            card1, card2 = hole_cards[0], hole_cards[1]
            
            if len(card1) < 2 or len(card2) < 2:
                return 5
            
            rank1, rank2 = card1[0], card2[0]
            suit1, suit2 = card1[1], card2[1]
            
            # Convert ranks to values
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
            rank_values.update({str(i): i for i in range(2, 10)})
            
            val1 = rank_values.get(rank1, 7)
            val2 = rank_values.get(rank2, 7)
            
            # Pocket pairs
            if rank1 == rank2:
                if val1 >= 10:
                    return 9  # High pairs
                elif val1 >= 7:
                    return 7  # Medium pairs
                else:
                    return 6  # Low pairs
            
            # Suited cards
            elif suit1 == suit2:
                avg_rank = (val1 + val2) / 2
                if avg_rank >= 11:
                    return 8  # High suited
                elif avg_rank >= 8:
                    return 6  # Medium suited
                else:
                    return 4  # Low suited
            
            # Offsuit cards
            else:
                avg_rank = (val1 + val2) / 2
                if avg_rank >= 12:
                    return 7  # High offsuit
                elif avg_rank >= 9:
                    return 5  # Medium offsuit
                else:
                    return 3  # Low offsuit
            
        except Exception:
            return 5  # Default middle bucket
    
    def _calculate_preflop_strength(self, val1: int, val2: int, suited: bool) -> int:
        """Calculate preflop hand strength with more nuanced evaluation."""
        # Pocket pairs
        if val1 == val2:
            if val1 >= 14:  # AA
                return 9
            elif val1 >= 11:  # JJ-KK
                return 8
            elif val1 >= 8:  # 88-TT
                return 7
            elif val1 >= 5:  # 55-77
                return 6
            else:  # 22-44
                return 5
        
        # High card hands
        else:
            high_card_strength = (val1 + val2) / 2
            connectedness = self._calculate_connectedness(val1, val2)
            
            # Base strength from high cards
            if high_card_strength >= 13:  # AK, AQ, AJ, etc.
                base = 8
            elif high_card_strength >= 11:  # QJ, KJ, etc.
                base = 7
            elif high_card_strength >= 9:   # TJ, T9, etc.
                base = 6
            elif high_card_strength >= 7:   # 89, 78, etc.
                base = 5
            elif high_card_strength >= 5:   # 67, 56, etc.
                base = 4
            else:
                base = 3
            
            # Adjust for suitedness
            if suited:
                base += 1
            
            # Adjust for connectedness
            base += connectedness
            
            return min(9, max(0, base))
    
    def _calculate_connectedness(self, val1: int, val2: int) -> int:
        """Calculate connectedness bonus for straight potential."""
        gap = abs(val1 - val2)
        
        if gap == 1:  # Connected (KQ, 98, etc.)
            return 1
        elif gap == 2:  # 1-gap (KJ, 97, etc.)
            return 0
        elif gap == 3:  # 2-gap (KT, 96, etc.)
            return -1
        else:  # 3+ gap
            return -1
    
    def _calculate_postflop_strength(self, hole_cards: List[str], community_cards: List[str]) -> int:
        """Calculate postflop hand strength considering made hands and draws."""
        try:
            all_cards = hole_cards + community_cards
            
            # Simple postflop evaluation based on hand types
            strength = self._evaluate_hand_type(all_cards)
            
            # Add drawing potential if not a made hand
            if strength < 7:  # Not a strong made hand
                draw_strength = self._evaluate_drawing_potential(hole_cards, community_cards)
                strength += draw_strength
            
            return min(9, max(0, strength))
            
        except Exception:
            return 5
    
    def _evaluate_hand_type(self, cards: List[str]) -> int:
        """Evaluate made hand strength (simplified)."""
        if len(cards) < 5:
            return 4  # No made hand yet
        
        try:
            # Extract ranks and suits
            ranks = [card[0] for card in cards if len(card) >= 2]
            suits = [card[1] for card in cards if len(card) >= 2]
            
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
            rank_values.update({str(i): i for i in range(2, 10)})
            
            rank_nums = [rank_values.get(rank, 7) for rank in ranks]
            rank_counts = {}
            for rank in rank_nums:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            # Check for various hand types
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            # Four of a kind
            if max_rank_count >= 4:
                return 9
            
            # Full house or three of a kind
            if max_rank_count >= 3:
                if len([count for count in rank_counts.values() if count >= 2]) >= 2:
                    return 8  # Full house
                else:
                    return 7  # Three of a kind
            
            # Flush
            if max_suit_count >= 5:
                return 6
            
            # Straight (simplified check)
            if self._has_straight(rank_nums):
                return 6
            
            # Two pair
            pairs = [count for count in rank_counts.values() if count >= 2]
            if len(pairs) >= 2:
                return 5
            
            # One pair
            if max_rank_count >= 2:
                return 4
            
            # High card
            return 3
            
        except Exception:
            return 4
    
    def _has_straight(self, rank_nums: List[int]) -> bool:
        """Check for straight (simplified)."""
        try:
            unique_ranks = sorted(set(rank_nums))
            
            # Check for 5 consecutive ranks
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    return True
            
            # Check for A-2-3-4-5 straight
            if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _evaluate_drawing_potential(self, hole_cards: List[str], community_cards: List[str]) -> int:
        """Evaluate drawing potential (flush draws, straight draws)."""
        try:
            if len(community_cards) < 3:  # Need at least flop
                return 0
            
            all_cards = hole_cards + community_cards
            suits = [card[1] for card in all_cards if len(card) >= 2]
            
            # Flush draw potential
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            if max_suit_count == 4:  # Flush draw
                return 2
            elif max_suit_count == 3:  # Backdoor flush draw
                return 1
            
            # Could add straight draw evaluation here
            # For now, return 0
            return 0
            
        except Exception:
            return 0
    
    def _get_board_texture(self, community_cards: List[str]) -> str:
        """Get board texture description."""
        if not community_cards:
            return "preflop"
        
        num_cards = len(community_cards)
        if num_cards == 3:
            return f"flop_{self._analyze_flop_texture(community_cards)}"
        elif num_cards == 4:
            return f"turn_{self._analyze_turn_texture(community_cards)}"
        elif num_cards >= 5:
            return f"river_{self._analyze_river_texture(community_cards)}"
        else:
            return f"partial_{num_cards}"
    
    def _analyze_flop_texture(self, cards: List[str]) -> str:
        """Analyze flop texture (rainbow, monotone, paired, etc.)."""
        if len(cards) < 3:
            return "incomplete"
        
        try:
            suits = [card[1] for card in cards if len(card) >= 2]
            ranks = [card[0] for card in cards if len(card) >= 2]
            
            # Check for pairs
            if len(set(ranks)) == 2:
                return "paired"
            
            # Check suits
            if len(set(suits)) == 1:
                return "monotone"
            elif len(set(suits)) == 2:
                return "twotone"
            else:
                return "rainbow"
                
        except Exception:
            return "unknown"
    
    def _analyze_turn_texture(self, cards: List[str]) -> str:
        """Analyze turn texture."""
        return "turn"  # Simplified for now
    
    def _analyze_river_texture(self, cards: List[str]) -> str:
        """Analyze river texture."""
        return "river"  # Simplified for now
    
    def _compress_action_history(self, action_history: List[int]) -> str:
        """Compress action history to essential betting patterns."""
        if not action_history:
            return "none"
        
        # Analyze recent action sequence for patterns
        recent_actions = action_history[-10:]  # Last 10 actions
        
        # Enhanced pattern recognition
        pattern_features = []
        
        # 1. Aggression level
        aggression = self._calculate_aggression_level(recent_actions)
        pattern_features.append(f"agg{aggression}")
        
        # 2. Action sequence pattern
        sequence_pattern = self._analyze_action_sequence(recent_actions)
        pattern_features.append(sequence_pattern)
        
        # 3. Betting round summary
        betting_rounds = self._summarize_betting_rounds(action_history)
        pattern_features.append(betting_rounds)
        
        return "_".join(pattern_features)
    
    def _calculate_aggression_level(self, actions: List[int]) -> int:
        """Calculate aggression level (0-3) based on action types."""
        if not actions:
            return 0
        
        aggressive_actions = [2, 3, 4]  # raises and all-in
        aggressive_count = sum(1 for action in actions if action in aggressive_actions)
        
        aggression_ratio = aggressive_count / len(actions)
        
        if aggression_ratio >= 0.6:
            return 3  # Very aggressive
        elif aggression_ratio >= 0.3:
            return 2  # Moderately aggressive
        elif aggression_ratio >= 0.1:
            return 1  # Slightly aggressive
        else:
            return 0  # Passive
    
    def _analyze_action_sequence(self, actions: List[int]) -> str:
        """Analyze action sequence for common patterns."""
        if len(actions) < 2:
            return "short"
        
        # Look for common patterns
        action_str = "".join(str(a) for a in actions[-6:])  # Last 6 actions
        
        # Check for specific patterns
        if "234" in action_str or "324" in action_str:
            return "escalating"  # Escalating aggression
        elif "210" in action_str or "320" in action_str:
            return "backdown"    # Backing down after aggression
        elif "111" in action_str:
            return "passive"     # Consistently passive
        elif "222" in action_str or "333" in action_str:
            return "aggressive"  # Consistently aggressive
        elif len(set(actions[-3:])) == 1 and len(actions) >= 3:
            return "repetitive"  # Repeating same action
        else:
            return "mixed"       # Mixed pattern
    
    def _summarize_betting_rounds(self, full_history: List[int]) -> str:
        """Summarize betting activity across different rounds."""
        if len(full_history) <= 3:
            return "early"
        
        # Simple heuristic: divide history into betting rounds
        # Each round typically has 2-6 actions (depending on players and actions)
        round_size = max(3, len(full_history) // 4)  # Rough estimate
        
        round_summaries = []
        for i in range(0, len(full_history), round_size):
            round_actions = full_history[i:i+round_size]
            if round_actions:
                aggression = self._calculate_aggression_level(round_actions)
                round_summaries.append(str(aggression))
        
        # Limit to last 3 rounds to keep key manageable
        recent_rounds = round_summaries[-3:] if len(round_summaries) >= 3 else round_summaries
        return "r" + "".join(recent_rounds)
    
    def _get_stack_pot_bucket(self, stack_size: float, pot_size: float) -> int:
        """Get stack-to-pot ratio bucket (0-4)."""
        if pot_size <= 0:
            return 2  # Middle bucket
        
        ratio = stack_size / pot_size
        
        if ratio < 0.5:
            return 0  # Very small stack
        elif ratio < 1.0:
            return 1  # Small stack
        elif ratio < 3.0:
            return 2  # Medium stack
        elif ratio < 10.0:
            return 3  # Large stack
        else:
            return 4  # Very large stack
    
    def update_regrets(self, info_set_key: str, regret_update: np.ndarray):
        """
        Update regrets for an information set.
        
        Args:
            info_set_key: Information set key
            regret_update: New regret values to add
        """
        current_regrets = self.regrets[info_set_key]
        
        if self.use_cfr_plus:
            # CFR+ update: handle positive and negative regrets differently
            updated_regrets = np.zeros_like(current_regrets)
            for i in range(len(current_regrets)):
                if regret_update[i] > 0:
                    # Accumulate positive regrets
                    updated_regrets[i] = current_regrets[i] + regret_update[i]
                else:
                    # Discount negative regrets
                    updated_regrets[i] = max(0, current_regrets[i] + regret_update[i] * 0.5)
        else:
            # Standard CFR update
            updated_regrets = current_regrets + regret_update
        
        # Apply decay and clipping
        updated_regrets = updated_regrets * self.regret_decay
        updated_regrets = np.clip(updated_regrets, -self.regret_clip, self.regret_clip)
        
        # Store updated regrets
        self.regrets[info_set_key] = updated_regrets
        self.cumulative_regrets[info_set_key] += np.abs(regret_update)
        
        # Update statistics
        self.info_set_counts[info_set_key] += 1
    
    def get_strategy(self, info_set_key: str, legal_actions: List[int]) -> np.ndarray:
        """
        Get strategy for information set using regret matching.
        
        Args:
            info_set_key: Information set key
            legal_actions: Legal actions for this state
            
        Returns:
            Strategy probability distribution
        """
        regrets = self.regrets[info_set_key]
        
        # Initialize strategy
        strategy = np.zeros(self.n_actions)
        
        if not legal_actions:
            return strategy
        
        # Get regrets for legal actions
        legal_regrets = regrets[legal_actions]
        
        # Apply regret matching
        positive_regrets = np.maximum(legal_regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            # Distribute probability proportional to positive regrets
            for i, action in enumerate(legal_actions):
                strategy[action] = positive_regrets[i] / sum_positive
        else:
            # Uniform distribution over legal actions
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = prob
        
        # Apply strategy smoothing
        epsilon = 0.01
        uniform_strategy = np.zeros(self.n_actions)
        for action in legal_actions:
            uniform_strategy[action] = 1.0 / len(legal_actions)
        
        strategy = (1 - epsilon) * strategy + epsilon * uniform_strategy
        
        # Update cumulative strategy
        self._update_cumulative_strategy(info_set_key, strategy)
        
        # Store current strategy
        self.strategies[info_set_key] = strategy.copy()
        
        return strategy
    
    def _update_cumulative_strategy(self, info_set_key: str, strategy: np.ndarray):
        """Update cumulative strategy for averaging."""
        decay = self.strategy_decay
        self.cumulative_strategies[info_set_key] = (
            decay * self.cumulative_strategies[info_set_key] + 
            (1 - decay) * strategy
        )
        self.strategy_counts[info_set_key] += 1
    
    def get_average_strategy(self, info_set_key: str, legal_actions: List[int]) -> np.ndarray:
        """Get average strategy over all iterations."""
        if info_set_key in self.cumulative_strategies:
            avg_strategy = self.cumulative_strategies[info_set_key].copy()
            
            # Normalize to legal actions
            legal_sum = sum(avg_strategy[action] for action in legal_actions)
            if legal_sum > 0:
                normalized = np.zeros(self.n_actions)
                for action in legal_actions:
                    normalized[action] = avg_strategy[action] / legal_sum
                return normalized
        
        # Fallback to uniform
        uniform = np.zeros(self.n_actions)
        if legal_actions:
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                uniform[action] = prob
        
        return uniform
    
    def save_state(self, filepath: str):
        """Save regret and strategy state to file."""
        state = {
            'regrets': dict(self.regrets),
            'cumulative_regrets': dict(self.cumulative_regrets),
            'strategies': dict(self.strategies),
            'cumulative_strategies': dict(self.cumulative_strategies),
            'strategy_counts': dict(self.strategy_counts),
            'info_set_counts': dict(self.info_set_counts),
            'total_iterations': self.total_iterations,
            'config': {
                'n_actions': self.n_actions,
                'regret_decay': self.regret_decay,
                'strategy_decay': self.strategy_decay,
                'use_cfr_plus': self.use_cfr_plus,
                'regret_clip': self.regret_clip
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load regret and strategy state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.regrets = defaultdict(lambda: np.zeros(self.n_actions), state['regrets'])
            self.cumulative_regrets = defaultdict(lambda: np.zeros(self.n_actions), state['cumulative_regrets'])
            self.strategies = defaultdict(lambda: np.ones(self.n_actions) / self.n_actions, state['strategies'])
            self.cumulative_strategies = defaultdict(lambda: np.zeros(self.n_actions), state['cumulative_strategies'])
            self.strategy_counts = defaultdict(int, state['strategy_counts'])
            self.info_set_counts = defaultdict(int, state['info_set_counts'])
            self.total_iterations = state['total_iterations']
            
            self.logger.info(f"Loaded state with {len(self.regrets)} information sets")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about information sets and regrets."""
        return {
            'num_info_sets': len(self.regrets),
            'total_iterations': self.total_iterations,
            'avg_regret_magnitude': np.mean([np.sum(np.abs(regrets)) for regrets in self.regrets.values()]) if self.regrets else 0,
            'max_regret_magnitude': np.max([np.max(np.abs(regrets)) for regrets in self.regrets.values()]) if self.regrets else 0,
            'most_visited_info_sets': sorted(self.info_set_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }