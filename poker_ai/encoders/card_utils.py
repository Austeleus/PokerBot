"""
Card encoding and hand evaluation utilities for poker.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import eval7


class CardEncoder:
    """
    Card encoding and hand evaluation utilities.
    
    Provides methods to encode poker cards into numerical representations
    suitable for neural network processing and evaluates hand strengths
    using the eval7 library.
    """
    
    def __init__(self):
        """Initialize card encoder with mappings."""
        self.card_to_index = {}
        self.index_to_card = {}
        self._build_mappings()
        
        # Standard card string representations
        self.rank_strings = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suit_strings = ['c', 'd', 'h', 's']
        
        # eval7 uses different format, so we need converters
        self.rank_to_eval7 = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
            '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
        }
        self.suit_to_eval7 = {'c': 'c', 'd': 'd', 'h': 'h', 's': 's'}
        
    def _build_mappings(self):
        """Build bidirectional mappings between cards and indices."""
        index = 0
        # Order: clubs, diamonds, hearts, spades (0-12 each suit)
        for suit in range(4):
            for rank in range(13):
                card = (rank, suit)
                self.card_to_index[card] = index
                self.index_to_card[index] = card
                index += 1
    
    def card_string_to_index(self, card_str: str) -> int:
        """
        Convert card string to index.
        
        Args:
            card_str: Card string like 'As' (Ace of Spades) or 'Tc' (Ten of Clubs)
            
        Returns:
            Card index (0-51)
        """
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank_char, suit_char = card_str[0].upper(), card_str[1].lower()
        
        # Map rank character to rank index
        rank_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }
        
        # Map suit character to suit index
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        
        if rank_char not in rank_map:
            raise ValueError(f"Invalid rank: {rank_char}")
        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit: {suit_char}")
        
        rank = rank_map[rank_char]
        suit = suit_map[suit_char]
        
        return self.card_to_index[(rank, suit)]
    
    def index_to_card_string(self, index: int) -> str:
        """
        Convert card index to string representation.
        
        Args:
            index: Card index (0-51)
            
        Returns:
            Card string like 'As'
        """
        if index < 0 or index >= 52:
            raise ValueError(f"Invalid card index: {index}")
        
        rank, suit = self.index_to_card[index]
        return self.rank_strings[rank] + self.suit_strings[suit]
    
    def string_to_eval7_card(self, card_str: str) -> eval7.Card:
        """
        Convert card string to eval7 Card object.
        
        Args:
            card_str: Card string like 'As'
            
        Returns:
            eval7.Card object
        """
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        # Ensure proper format: rank (uppercase) + suit (lowercase)
        rank = card_str[0].upper()
        suit = card_str[1].lower()
        
        # Validate rank and suit
        valid_ranks = '23456789TJQKA'
        valid_suits = 'cdhs'
        
        if rank not in valid_ranks:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in valid_suits:
            raise ValueError(f"Invalid suit: {suit}")
        
        return eval7.Card(rank + suit)
    
    def encode_cards(self, cards: List[str]) -> np.ndarray:
        """
        Convert list of card strings to one-hot encoding.
        
        Args:
            cards: List of card strings
            
        Returns:
            One-hot encoded array of shape (52,)
        """
        encoding = np.zeros(52, dtype=np.float32)
        
        for card_str in cards:
            if card_str and card_str.strip():  # Skip empty strings
                try:
                    index = self.card_string_to_index(card_str)
                    encoding[index] = 1.0
                except ValueError:
                    continue  # Skip invalid cards
        
        return encoding
    
    def decode_cards(self, encoding: np.ndarray) -> List[str]:
        """
        Convert one-hot encoding back to card strings.
        
        Args:
            encoding: One-hot encoded array of shape (52,)
            
        Returns:
            List of card strings in index order
        """
        cards = []
        for i, value in enumerate(encoding):
            if value > 0.5:  # Card is present
                cards.append(self.index_to_card_string(i))
        
        return cards
    
    def decode_cards_preserve_order(self, encoding: np.ndarray, original_cards: List[str]) -> List[str]:
        """
        Decode cards while preserving the original order.
        
        Args:
            encoding: One-hot encoded array of shape (52,)
            original_cards: Original card list to preserve order
            
        Returns:
            List of card strings in original order
        """
        # Get set of encoded cards
        encoded_indices = {i for i, value in enumerate(encoding) if value > 0.5}
        
        # Return original cards that are in the encoding
        preserved_cards = []
        for card in original_cards:
            try:
                card_index = self.card_string_to_index(card)
                if card_index in encoded_indices:
                    preserved_cards.append(card)
            except ValueError:
                continue
                
        return preserved_cards
    
    def evaluate_hand(self, hole_cards: List[str], community_cards: List[str]) -> Tuple[int, int]:
        """
        Evaluate poker hand strength using eval7.
        
        Args:
            hole_cards: List of hole card strings (e.g., ['As', 'Kh'])
            community_cards: List of community card strings (e.g., ['Qc', 'Jd', 'Ts'])
            
        Returns:
            Tuple of (hand_rank, percentile)
            - hand_rank: Lower is better (1 is best possible hand)
            - percentile: 0-100, where 100 is the best
        """
        # Need at least 2 hole cards
        if len(hole_cards) < 2:
            return 7462, 0  # Worst possible hand
        
        # Convert to eval7 cards
        eval7_cards = []
        
        # Add hole cards
        for card_str in hole_cards[:2]:  # Only use first 2 hole cards
            if card_str and card_str.strip():
                try:
                    eval7_cards.append(self.string_to_eval7_card(card_str))
                except Exception:
                    continue
        
        # Add community cards (up to 5)
        for card_str in community_cards[:5]:
            if card_str and card_str.strip():
                try:
                    eval7_cards.append(self.string_to_eval7_card(card_str))
                except Exception:
                    continue
        
        # Need at least 5 cards total to evaluate
        if len(eval7_cards) < 5:
            # Can't evaluate with less than 5 cards
            # Return a default weak hand value
            return 7462, 0
        
        try:
            # eval7 can handle 5-7 cards and finds the best 5-card hand
            raw_rank = eval7.evaluate(eval7_cards)
            
            # Convert eval7's raw rank to standard 1-7462 scale
            # eval7 range: ~135M (best) down to ~340K (worst)
            # Standard scale: 1 (best) to 7462 (worst)
            
            # Define eval7's observed range
            EVAL7_BEST = 135004160  # Royal flush  
            EVAL7_WORST = 340496    # 7-5-4-3-2 offsuit
            
            # Linear conversion to 1-7462 scale (inverted since higher eval7 = better)
            normalized = (raw_rank - EVAL7_WORST) / (EVAL7_BEST - EVAL7_WORST)
            hand_rank = int(1 + (1 - normalized) * 7461)  # Best eval7 -> rank 1, worst -> 7462
            
            # Ensure valid range
            hand_rank = max(1, min(7462, hand_rank))
            
            # Convert rank to percentile (lower rank is better)
            percentile = int(100 * (7462 - hand_rank) / 7461)
            
            return hand_rank, percentile
            
        except Exception:
            # If evaluation fails, return default weak hand
            return 7462, 0
    
    def get_hand_bucket(self, hand_strength: int, num_buckets: int = 200) -> int:
        """
        Map hand strength to abstraction bucket.
        
        Args:
            hand_strength: eval7 hand rank (1-7462)
            num_buckets: Number of abstraction buckets
            
        Returns:
            Bucket index (0 to num_buckets-1)
        """
        # Convert eval7 rank to bucket
        # Lower rank is better, so invert the scale
        normalized = (7462 - hand_strength) / 7461
        bucket = int(normalized * (num_buckets - 1))
        return min(max(0, bucket), num_buckets - 1)
    
    def create_card_mask(self, known_cards: List[str]) -> np.ndarray:
        """
        Create mask for known cards (to avoid sampling them).
        
        Args:
            known_cards: List of known card strings
            
        Returns:
            Boolean mask array of shape (52,) where True means card is available
        """
        mask = np.ones(52, dtype=bool)
        
        for card_str in known_cards:
            if card_str and card_str.strip():
                try:
                    index = self.card_string_to_index(card_str)
                    mask[index] = False
                except ValueError:
                    continue  # Skip invalid cards
        
        return mask
    
    def sample_random_cards(self, num_cards: int, excluded_cards: List[str]) -> List[str]:
        """
        Sample random cards excluding known cards.
        
        Args:
            num_cards: Number of cards to sample
            excluded_cards: List of card strings to exclude
            
        Returns:
            List of sampled card strings
        """
        mask = self.create_card_mask(excluded_cards)
        available_indices = np.where(mask)[0]
        
        if len(available_indices) < num_cards:
            raise ValueError(f"Not enough cards available. Need {num_cards}, have {len(available_indices)}")
        
        sampled_indices = np.random.choice(available_indices, size=num_cards, replace=False)
        return [self.index_to_card_string(idx) for idx in sampled_indices]
    
    def get_hand_features(self, hole_cards: List[str], community_cards: List[str]) -> np.ndarray:
        """
        Extract hand features for neural network input.
        
        Args:
            hole_cards: List of hole card strings
            community_cards: List of community card strings
            
        Returns:
            Feature vector containing:
            - One-hot encoding of cards (52 dims)
            - Hand strength percentile (1 dim)
            - Number of cards seen (1 dim)
        """
        # One-hot encode all cards
        all_cards = hole_cards + community_cards
        card_encoding = self.encode_cards(all_cards)
        
        # Get hand strength if we have enough cards
        if len(hole_cards) >= 2 and len(all_cards) >= 5:
            _, percentile = self.evaluate_hand(hole_cards, community_cards)
            hand_strength = percentile / 100.0
        else:
            hand_strength = 0.0
        
        # Number of cards seen (normalized)
        num_cards = len([c for c in all_cards if c and c.strip()]) / 7.0
        
        # Combine features
        features = np.concatenate([
            card_encoding,
            np.array([hand_strength, num_cards], dtype=np.float32)
        ])
        
        return features