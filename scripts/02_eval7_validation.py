#!/usr/bin/env python3
"""
Comprehensive eval7 validation script to ensure hand evaluation is correct.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_ai.encoders.card_utils import CardEncoder
import eval7

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_hand_rankings():
    """Test known hand rankings to validate eval7 integration"""
    encoder = CardEncoder()
    
    print_header("EVAL7 HAND RANKING VALIDATION")
    
    # Test cases with expected relative rankings
    test_hands = [
        {
            'name': 'Royal Flush (Spades)',
            'hole': ['As', 'Ks'],
            'community': ['Qs', 'Js', 'Ts', '7h', '2c'],
            'expected_type': 'Straight Flush'
        },
        {
            'name': 'Four Kings',
            'hole': ['Ks', 'Kd'], 
            'community': ['Kh', 'Kc', 'Qh', '7s', '2c'],
            'expected_type': 'Four of a Kind'
        },
        {
            'name': 'Four Aces',
            'hole': ['As', 'Ad'],
            'community': ['Ah', 'Ac', 'Kh', '7s', '2c'], 
            'expected_type': 'Four of a Kind'
        },
        {
            'name': 'Full House (Aces over Kings)',
            'hole': ['As', 'Ad'],
            'community': ['Ah', 'Ks', 'Kd', '7s', '2c'],
            'expected_type': 'Full House'
        },
        {
            'name': 'Flush (Ace High)',
            'hole': ['As', 'Ks'],
            'community': ['Qs', 'Js', '9s', '7h', '2c'],
            'expected_type': 'Flush'
        },
        {
            'name': 'Straight (Ace High)',
            'hole': ['As', 'Kd'],
            'community': ['Qh', 'Jc', 'Ts', '7h', '2c'],
            'expected_type': 'Straight'
        },
        {
            'name': 'Three Aces',
            'hole': ['As', 'Ad'],
            'community': ['Ah', 'Kh', 'Qh', '7s', '2c'],
            'expected_type': 'Three of a Kind'
        },
        {
            'name': 'Two Pair (Aces and Kings)',
            'hole': ['As', 'Ad'],
            'community': ['Kh', 'Kd', 'Qh', '7s', '2c'],
            'expected_type': 'Two Pair'
        },
        {
            'name': 'Pair of Aces',
            'hole': ['As', 'Ad'],
            'community': ['Kh', 'Qd', 'Jh', '7s', '2c'],
            'expected_type': 'One Pair'
        },
        {
            'name': 'High Card (Ace)',
            'hole': ['As', 'Kd'],
            'community': ['Qh', 'Jc', '9s', '7h', '2c'],
            'expected_type': 'High Card'
        }
    ]
    
    results = []
    
    for hand in test_hands:
        rank, percentile = encoder.evaluate_hand(hand['hole'], hand['community'])
        
        # Get actual hand type using eval7 directly
        eval7_cards = []
        for card in hand['hole'] + hand['community']:
            try:
                eval7_cards.append(eval7.Card(card))
            except:
                continue
        
        if len(eval7_cards) >= 5:
            eval7_rank = eval7.evaluate(eval7_cards)
            hand_class = eval7.handtype(eval7_rank)
        else:
            eval7_rank = 7462
            hand_class = "Invalid"
        
        results.append({
            'name': hand['name'],
            'rank': rank,
            'percentile': percentile,
            'expected_type': hand['expected_type'],
            'actual_type': hand_class,
            'eval7_rank': eval7_rank
        })
        
        print(f"{hand['name']:<25} | Rank: {rank:4d} | {percentile:3d}% | Expected: {hand['expected_type']:<15} | Actual: {hand_class}")
    
    return results

def test_four_of_a_kind_rankings():
    """Specifically test four of a kind rankings"""
    encoder = CardEncoder()
    
    print_header("FOUR OF A KIND DETAILED ANALYSIS")
    
    four_kinds = [
        (['As', 'Ad'], ['Ah', 'Ac', 'Kh'], 'Four Aces (with King)'),
        (['Ks', 'Kd'], ['Kh', 'Kc', 'Ah'], 'Four Kings (with Ace)'),
        (['Ks', 'Kd'], ['Kh', 'Kc', 'Qh'], 'Four Kings (with Queen)'),
        (['Qs', 'Qd'], ['Qh', 'Qc', 'Ah'], 'Four Queens (with Ace)'),
        (['2s', '2d'], ['2h', '2c', 'Ah'], 'Four Twos (with Ace)'),
        (['2s', '2d'], ['2h', '2c', '3h'], 'Four Twos (with Three)')
    ]
    
    print("Analyzing four of a kind hands:")
    print(f"{'Hand':<25} | {'Rank':<6} | {'%ile':<4} | {'Cards'}")
    print("-" * 70)
    
    for hole, community, description in four_kinds:
        rank, percentile = encoder.evaluate_hand(hole, community)
        all_cards = hole + community
        print(f"{description:<25} | {rank:<6} | {percentile:<4} | {all_cards}")

def test_eval7_direct():
    """Test eval7 directly to understand the ranking system"""
    print_header("DIRECT EVAL7 TESTING")
    
    # Test the four kings hand directly
    cards = [eval7.Card('Ks'), eval7.Card('Kd'), eval7.Card('Kh'), eval7.Card('Kc'), eval7.Card('Qh')]
    rank = eval7.evaluate(cards)
    hand_type = eval7.handtype(rank)
    
    print(f"Four Kings direct eval7 test:")
    print(f"Cards: {[str(c) for c in cards]}")
    print(f"Raw eval7 rank: {rank}")
    print(f"Hand type: {hand_type}")
    # print(f"Rank class: {eval7.rank_class(rank)}")  # Not available in this eval7 version
    
    # Test different four of a kinds
    print(f"\nComparing different four of a kinds:")
    
    test_cases = [
        ([eval7.Card('As'), eval7.Card('Ad'), eval7.Card('Ah'), eval7.Card('Ac'), eval7.Card('Kh')], "Four Aces"),
        ([eval7.Card('Ks'), eval7.Card('Kd'), eval7.Card('Kh'), eval7.Card('Kc'), eval7.Card('Ah')], "Four Kings + Ace"),
        ([eval7.Card('Ks'), eval7.Card('Kd'), eval7.Card('Kh'), eval7.Card('Kc'), eval7.Card('Qh')], "Four Kings + Queen"),
        ([eval7.Card('2s'), eval7.Card('2d'), eval7.Card('2h'), eval7.Card('2c'), eval7.Card('Ah')], "Four Twos"),
    ]
    
    for cards, description in test_cases:
        rank = eval7.evaluate(cards)
        print(f"{description:<20}: rank {rank}")
    
    # Show the eval7 ranking system
    print(f"\neval7 Ranking System:")
    print(f"- Lower numbers are BETTER hands")
    print(f"- Royal Flush: rank 1")
    print(f"- High Card: rank 7462 (worst)")
    print(f"- Four of a Kind range: approximately 11-166")

def analyze_ranking_distribution():
    """Analyze the distribution of hand rankings"""
    print_header("RANKING DISTRIBUTION ANALYSIS")
    
    # Show some reference points
    reference_hands = [
        ([eval7.Card('As'), eval7.Card('Ks'), eval7.Card('Qs'), eval7.Card('Js'), eval7.Card('Ts')], "Royal Flush"),
        ([eval7.Card('9s'), eval7.Card('8s'), eval7.Card('7s'), eval7.Card('6s'), eval7.Card('5s')], "Straight Flush"),
        ([eval7.Card('As'), eval7.Card('Ad'), eval7.Card('Ah'), eval7.Card('Ac'), eval7.Card('Kh')], "Four Aces"),
        ([eval7.Card('2s'), eval7.Card('2d'), eval7.Card('2h'), eval7.Card('2c'), eval7.Card('3h')], "Four Twos"),
        ([eval7.Card('As'), eval7.Card('Ad'), eval7.Card('Ah'), eval7.Card('Ks'), eval7.Card('Kd')], "Aces full of Kings"),
        ([eval7.Card('As'), eval7.Card('Ks'), eval7.Card('Qs'), eval7.Card('Js'), eval7.Card('9s')], "Ace-high flush"),
        ([eval7.Card('As'), eval7.Card('Kd'), eval7.Card('Qh'), eval7.Card('Jc'), eval7.Card('Ts')], "Broadway straight"),
    ]
    
    print(f"{'Hand Type':<20} | {'Rank':<6} | {'Percentile'}")
    print("-" * 45)
    
    for cards, description in reference_hands:
        rank = eval7.evaluate(cards)
        # Calculate percentile (lower rank = higher percentile)
        percentile = int((7462 - rank) / 7462 * 100)
        print(f"{description:<20} | {rank:<6} | {percentile}%")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE EVAL7 VALIDATION")
    print("   Verifying hand evaluation accuracy and rankings")
    
    test_hand_rankings()
    test_four_of_a_kind_rankings() 
    test_eval7_direct()
    analyze_ranking_distribution()
    
    print(f"\n{'='*60}")
    print("📊 ANALYSIS COMPLETE")
    print("   Check if four kings ranking makes sense based on eval7 system")
    print(f"{'='*60}")