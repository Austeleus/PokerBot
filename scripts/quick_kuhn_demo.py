#!/usr/bin/env python3
"""
Quick Kuhn Poker Demo

A shorter script that demonstrates key concepts without user interaction.
Run this first to get familiar with the environment basics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from poker_ai.envs.kuhn.kuhn_poker import KuhnPokerEnv, Action, get_action_name


def demo_basic_game():
    """Demonstrate a basic game flow"""
    print("🎯 DEMO 1: Basic Game Flow")
    print("="*50)
    
    env = KuhnPokerEnv(seed=42)
    print(f"Initial state: {env}")
    print(f"Player 0 has: {env.player_cards[0]}, Player 1 has: {env.player_cards[1]}")
    
    step = 1
    while not env.is_terminal_state:
        state = env.get_state()
        legal_actions = state['legal_actions']
        action = random.choice(legal_actions)
        action_name = get_action_name(action, state['history'])
        
        print(f"\nStep {step}: Player {state['current_player']} {action_name}s")
        env.step(action)
        print(f"New state: {env}")
        step += 1
    
    print(f"\n🏁 Game over! Payoffs: P0={env.get_payoff(0)}, P1={env.get_payoff(1)}")


def demo_all_terminal_patterns():
    """Show all 5 possible ways a game can end"""
    print("\n\n🎯 DEMO 2: All Terminal Patterns")
    print("="*50)
    
    patterns = {
        "CC": "Both players check",
        "BC": "Player 0 bets, Player 1 calls",
        "BF": "Player 0 bets, Player 1 folds", 
        "CBF": "Player 0 checks, Player 1 bets, Player 0 folds",
        "CBC": "Player 0 checks, Player 1 bets, Player 0 calls"
    }
    
    # Use Queen vs King for demonstration
    for pattern, description in patterns.items():
        env = KuhnPokerEnv()
        env.set_cards(2, 3)  # Queen vs King
        
        # Simulate the pattern
        for action_char in pattern:
            if action_char == 'C':
                if env.history.endswith('B'):
                    action = Action.CHECK_CALL.value  # Call
                    action_name = "Call"
                else:
                    action = Action.CHECK_CALL.value  # Check  
                    action_name = "Check"
            elif action_char == 'B':
                action = Action.BET_FOLD.value  # Bet
                action_name = "Bet"
            elif action_char == 'F':
                action = Action.BET_FOLD.value  # Fold
                action_name = "Fold"
            
            print(f"  Player {env.current_player} {action_name}s")
            env.step(action)
        
        payoffs = [env.get_payoff(0), env.get_payoff(1)]
        print(f"📋 {pattern}: {description}")
        print(f"   Result: P0={payoffs[0]:+2d}, P1={payoffs[1]:+2d} (Queen vs King)")
        print()


def demo_information_sets():
    """Show how information sets work"""
    print("\n🎯 DEMO 3: Information Sets")
    print("="*50)
    
    env = KuhnPokerEnv()
    env.set_cards(1, 3)  # Jack vs King
    
    print("Information sets track what each player knows:")
    print(f"Initial - P0 info set: '{env.get_info_set(0)}' (Jack, no history)")
    print(f"Initial - P1 info set: '{env.get_info_set(1)}' (King, no history)")
    
    # P0 checks
    env.step(Action.CHECK_CALL.value)
    print(f"\nAfter P0 checks:")
    print(f"P0 info set: '{env.get_info_set(0)}' (Jack, after checking)")
    print(f"P1 info set: '{env.get_info_set(1)}' (King, opponent checked)")
    
    # P1 bets
    env.step(Action.BET_FOLD.value)
    print(f"\nAfter P1 bets:")
    print(f"P0 info set: '{env.get_info_set(0)}' (Jack, opponent checked then bet)")
    print(f"P1 info set: '{env.get_info_set(1)}' (King, after check-bet)")
    
    print(f"\nFormat: 'card/history' where:")
    print(f"- card: 1=Jack, 2=Queen, 3=King")
    print(f"- history: C=check/call, B=bet, F=fold")


def demo_payoff_structure():
    """Demonstrate the payoff structure"""
    print("\n\n🎯 DEMO 4: Payoff Structure")
    print("="*50)
    
    print("Kuhn poker payoffs depend on pot size and showdown:")
    print("- Small pot (2 chips): ±1 payoff")
    print("- Big pot (4 chips): ±2 payoff")
    print("- Folds: Always ±1 payoff")
    print()
    
    scenarios = [
        (1, 3, "CC", "Jack vs King, both check → small pot showdown"),
        (1, 3, "BC", "Jack vs King, bet-call → big pot showdown"),
        (1, 3, "BF", "Jack bets, King folds → Jack wins by fold"),
        (3, 1, "CBF", "King checks, Jack bets, King folds → Jack wins by fold"),
        (3, 1, "CBC", "King checks, Jack bets, King calls → big pot showdown")
    ]
    
    for p0_card, p1_card, pattern, description in scenarios:
        env = KuhnPokerEnv()
        env.set_cards(p0_card, p1_card)
        
        # Simulate pattern
        for action_char in pattern:
            if action_char == 'C':
                action = Action.CHECK_CALL.value
            elif action_char == 'B':
                action = Action.BET_FOLD.value
            elif action_char == 'F':
                action = Action.BET_FOLD.value
            env.step(action)
        
        payoffs = [env.get_payoff(0), env.get_payoff(1)]
        print(f"📊 {description}")
        print(f"   Payoffs: P0={payoffs[0]:+2d}, P1={payoffs[1]:+2d}")
        print()


def demo_strategy_implications():
    """Show why strategy matters in Kuhn poker"""
    print("\n🎯 DEMO 5: Why Strategy Matters")
    print("="*50)
    
    print("Even in this simple game, strategy is crucial:")
    print()
    
    # Show how different cards should play differently
    card_names = {1: "Jack (worst)", 2: "Queen (middle)", 3: "King (best)"}
    
    print("💡 Strategic insights:")
    print("- King (3): Almost always bet/call (you usually win)")
    print("- Queen (2): Mixed strategy (depends on opponent)")  
    print("- Jack (1): Mostly check/fold, sometimes bluff")
    print()
    
    print("Example: What should Jack do facing a bet?")
    
    # Jack facing a bet in different situations
    situations = [
        ("J/B", "Jack, opponent bet first → usually fold"),
        ("J/CB", "Jack, after check-bet → usually fold, sometimes call")
    ]
    
    for info_set, explanation in situations:
        print(f"📋 Info set '{info_set}': {explanation}")
    
    print()
    print("🧠 The MCCFR algorithm will learn these optimal strategies!")
    print("   It discovers the right mix of betting, calling, and folding.")


def demo_game_tree_size():
    """Show the manageable size of Kuhn poker"""
    print("\n\n🎯 DEMO 6: Game Tree Complexity")
    print("="*50)
    
    print("Why Kuhn poker is perfect for learning CFR:")
    print()
    
    # Count information sets
    info_sets = set()
    for p0_card in [1, 2, 3]:
        for p1_card in [1, 2, 3]:
            if p0_card != p1_card:
                histories = ["", "C", "B", "CB"]  # Non-terminal histories
                for history in histories:
                    current_player = len(history) % 2
                    info_set = f"{[p0_card, p1_card][current_player]}/{history}"
                    info_sets.add(info_set)
    
    print(f"📊 Total information sets: {len(info_sets)}")
    print(f"📊 Actions per info set: 2 (check/call or bet/fold)")
    print(f"📊 Total game outcomes: 5 (CC, BC, BF, CBF, CBC)")
    print(f"📊 Card combinations: 6 (3×2 permutations)")
    print()
    print("🎯 Small enough to understand completely!")
    print("🎯 Large enough to demonstrate CFR principles!")
    print("🎯 Perfect stepping stone to complex poker!")


def main():
    """Run all demonstrations"""
    print("🃏 KUHN POKER ENVIRONMENT QUICK DEMO")
    print("This script demonstrates key concepts to help you understand the environment.")
    print()
    
    demo_basic_game()
    demo_all_terminal_patterns() 
    demo_information_sets()
    demo_payoff_structure()
    demo_strategy_implications()
    demo_game_tree_size()
    
    print("\n" + "="*60)
    print("🎉 Demo complete! Key takeaways:")
    print("✅ Kuhn poker has simple rules but strategic depth")
    print("✅ Information sets capture what players know")
    print("✅ Payoffs depend on pot size and showdown")
    print("✅ Perfect testbed for MCCFR algorithm")
    print()
    print("💡 Next step: Run 'python scripts/interactive_kuhn_poker.py'")
    print("   for hands-on exploration!")


if __name__ == "__main__":
    main()