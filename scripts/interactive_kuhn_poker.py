#!/usr/bin/env python3
"""
Interactive Kuhn Poker Environment Explorer

This script lets you:
1. Play manual games against random opponent
2. Explore all possible game scenarios
3. Analyze payoff structures
4. Test specific card combinations
5. Understand information sets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from poker_ai.envs.kuhn.kuhn_poker import KuhnPokerEnv, Action, get_action_name


def print_game_state(env, show_both_cards=False):
    """Print current game state in a readable format"""
    state = env.get_state()
    
    print(f"\n{'='*50}")
    print(f"🃏 GAME STATE")
    print(f"{'='*50}")
    
    if show_both_cards:
        card_names = {1: "Jack", 2: "Queen", 3: "King"}
        print(f"Player 0 has: {card_names[state['cards'][0]]} ({state['cards'][0]})")
        print(f"Player 1 has: {card_names[state['cards'][1]]} ({state['cards'][1]})")
    else:
        # Only show current player's card in real game
        if not env.is_terminal_state:
            card_names = {1: "Jack", 2: "Queen", 3: "King"}
            current_card = state['cards'][state['current_player']]
            print(f"Your card: {card_names[current_card]} ({current_card})")
    
    print(f"Action history: '{state['history']}' (C=check/call, B=bet, F=fold)")
    print(f"Pot size: {state['pot']} chips")
    
    if env.is_terminal_state:
        print(f"🏁 GAME OVER!")
        print(f"Final payoffs: P0={env.get_payoff(0)}, P1={env.get_payoff(1)}")
        
        # Explain the outcome
        history = state['history']
        if history == "CC":
            print("📝 Both players checked - showdown with small pot (2 chips)")
        elif history == "BC":
            print("📝 Bet and call - showdown with big pot (4 chips)")
        elif history == "BF":
            print("📝 Player 0 bet, Player 1 folded - Player 0 wins")
        elif history == "CBF":
            print("📝 Player 0 checked, Player 1 bet, Player 0 folded - Player 1 wins")
        elif history == "CBC":
            print("📝 Check, bet, call - showdown with big pot (4 chips)")
    else:
        print(f"Current player: {state['current_player']}")
        print(f"Information set: {env.get_info_set(state['current_player'])}")


def get_user_action(env):
    """Get action from user input"""
    state = env.get_state()
    legal_actions = state['legal_actions']
    
    print(f"\n🎯 Available actions for Player {state['current_player']}:")
    
    for i, action in enumerate(legal_actions):
        action_name = get_action_name(action, state['history'])
        print(f"  {i}: {action_name}")
    
    while True:
        try:
            choice = input(f"\nChoose action (0-{len(legal_actions)-1}): ").strip()
            choice_idx = int(choice)
            
            if 0 <= choice_idx < len(legal_actions):
                return legal_actions[choice_idx]
            else:
                print(f"❌ Invalid choice. Enter 0-{len(legal_actions)-1}")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Enter a number.")
        except EOFError:
            print("\n👋 Exiting...")
            return None


def play_manual_game():
    """Play a manual game where user controls one player"""
    print("\n🎮 MANUAL GAME MODE")
    print("You are Player 0. Player 1 will play randomly.")
    
    env = KuhnPokerEnv()
    print_game_state(env, show_both_cards=False)
    
    while not env.is_terminal_state:
        state = env.get_state()
        
        if state['current_player'] == 0:
            # Human player
            action = get_user_action(env)
            if action is None:  # User quit
                return
            
            action_name = get_action_name(action, state['history'])
            print(f"\n🧑 You chose: {action_name}")
            
        else:
            # Random opponent
            legal_actions = state['legal_actions']
            action = random.choice(legal_actions)
            action_name = get_action_name(action, state['history'])
            print(f"\n🤖 Opponent chose: {action_name}")
        
        env.step(action)
        print_game_state(env, show_both_cards=False)
    
    # Reveal both cards at the end
    print(f"\n🃏 CARD REVEAL:")
    card_names = {1: "Jack", 2: "Queen", 3: "King"}
    cards = env.get_state()['cards']
    print(f"You had: {card_names[cards[0]]} ({cards[0]})")
    print(f"Opponent had: {card_names[cards[1]]} ({cards[1]})")


def explore_all_scenarios():
    """Show all possible game outcomes for each card combination"""
    print("\n📊 COMPLETE SCENARIO ANALYSIS")
    print("Exploring all card combinations and game paths...")
    
    card_names = {1: "Jack", 2: "Queen", 3: "King"}
    terminal_patterns = ["CC", "BC", "BF", "CBF", "CBC"]
    pattern_descriptions = {
        "CC": "Both check",
        "BC": "Bet-call", 
        "BF": "Bet-fold",
        "CBF": "Check-bet-fold",
        "CBC": "Check-bet-call"
    }
    
    for p0_card in [1, 2, 3]:
        for p1_card in [1, 2, 3]:
            if p0_card != p1_card:
                print(f"\n{'='*60}")
                print(f"🃏 P0: {card_names[p0_card]} vs P1: {card_names[p1_card]}")
                print(f"{'='*60}")
                
                for pattern in terminal_patterns:
                    env = KuhnPokerEnv()
                    env.set_cards(p0_card, p1_card)
                    
                    # Simulate the pattern
                    for action_char in pattern:
                        if action_char == 'C':
                            if env.history.endswith('B'):
                                action = Action.CHECK_CALL.value  # Call
                            else:
                                action = Action.CHECK_CALL.value  # Check
                        elif action_char == 'B':
                            action = Action.BET_FOLD.value  # Bet
                        elif action_char == 'F':
                            action = Action.BET_FOLD.value  # Fold
                        
                        env.step(action)
                    
                    payoffs = [env.get_payoff(0), env.get_payoff(1)]
                    print(f"  {pattern} ({pattern_descriptions[pattern]:15s}): P0={payoffs[0]:+2d}, P1={payoffs[1]:+2d}")


def analyze_information_sets():
    """Analyze all possible information sets in the game"""
    print("\n🧠 INFORMATION SET ANALYSIS")
    print("All possible information sets players can encounter:")
    
    card_names = {1: "Jack", 2: "Queen", 3: "King"}
    
    # Collect all possible info sets
    info_sets = {0: set(), 1: set()}
    
    # Generate all possible game states
    for p0_card in [1, 2, 3]:
        for p1_card in [1, 2, 3]:
            if p0_card != p1_card:
                # All possible histories where each player might have to act
                histories = ["", "C", "B", "CB"]
                
                for history in histories:
                    env = KuhnPokerEnv()
                    env.set_cards(p0_card, p1_card)
                    env.history = history
                    
                    # Determine whose turn it would be
                    if len(history) % 2 == 0:
                        current_player = 0
                    else:
                        current_player = 1
                    
                    # Check if this is a valid non-terminal state
                    if not env._is_terminal_history(history):
                        info_set = env.get_info_set(current_player)
                        info_sets[current_player].add(info_set)
    
    for player in [0, 1]:
        print(f"\n📋 Player {player} Information Sets:")
        sorted_info_sets = sorted(list(info_sets[player]))
        
        for info_set in sorted_info_sets:
            card, history = info_set.split('/')
            card_name = card_names[int(card)]
            
            if history == "":
                context = "First to act"
            elif history == "C":
                context = "Opponent checked" if player == 1 else "After checking"
            elif history == "B":
                context = "Opponent bet" if player == 1 else "After betting"
            elif history == "CB":
                context = "Checked, opponent bet" if player == 0 else "After check-bet"
            
            print(f"  {info_set:6s} - {card_name} ({context})")


def payoff_analysis():
    """Analyze expected payoffs for different strategies"""
    print("\n💰 PAYOFF ANALYSIS")
    print("Expected payoffs for always playing the same action:")
    
    strategies = {
        "Always Check/Call": Action.CHECK_CALL.value,
        "Always Bet/Fold": Action.BET_FOLD.value
    }
    
    for strategy_name, action in strategies.items():
        total_payoff = 0
        game_count = 0
        
        print(f"\n📈 {strategy_name} Strategy:")
        
        # Test against all possible opponents and card combinations
        for p0_card in [1, 2, 3]:
            for p1_card in [1, 2, 3]:
                if p0_card != p1_card:
                    for opponent_action in [Action.CHECK_CALL.value, Action.BET_FOLD.value]:
                        env = KuhnPokerEnv()
                        env.set_cards(p0_card, p1_card)
                        
                        # Play the game
                        while not env.is_terminal_state:
                            if env.current_player == 0:
                                env.step(action)
                            else:
                                env.step(opponent_action)
                        
                        payoff = env.get_payoff(0)
                        total_payoff += payoff
                        game_count += 1
        
        avg_payoff = total_payoff / game_count if game_count > 0 else 0
        print(f"  Average payoff: {avg_payoff:.4f}")
        print(f"  Total games: {game_count}")


def test_specific_scenario():
    """Let user test a specific card combination and see all outcomes"""
    print("\n🔬 SPECIFIC SCENARIO TESTER")
    
    card_names = {1: "Jack", 2: "Queen", 3: "King"}
    
    print("Choose cards for both players:")
    print("1 = Jack, 2 = Queen, 3 = King")
    
    try:
        p0_card = int(input("Player 0 card (1-3): "))
        p1_card = int(input("Player 1 card (1-3): "))
        
        if p0_card not in [1, 2, 3] or p1_card not in [1, 2, 3]:
            print("❌ Invalid cards. Use 1, 2, or 3.")
            return
        
        if p0_card == p1_card:
            print("❌ Players cannot have the same card.")
            return
        
        print(f"\n🃏 Testing: P0 has {card_names[p0_card]}, P1 has {card_names[p1_card]}")
        
        # Show all possible outcomes
        terminal_patterns = ["CC", "BC", "BF", "CBF", "CBC"]
        pattern_descriptions = {
            "CC": "Both players check",
            "BC": "P0 bets, P1 calls", 
            "BF": "P0 bets, P1 folds",
            "CBF": "P0 checks, P1 bets, P0 folds",
            "CBC": "P0 checks, P1 bets, P0 calls"
        }
        
        print(f"\n📊 All possible outcomes:")
        for pattern in terminal_patterns:
            env = KuhnPokerEnv()
            env.set_cards(p0_card, p1_card)
            
            # Simulate the pattern
            for action_char in pattern:
                if action_char == 'C':
                    if env.history.endswith('B'):
                        action = Action.CHECK_CALL.value  # Call
                    else:
                        action = Action.CHECK_CALL.value  # Check
                elif action_char == 'B':
                    action = Action.BET_FOLD.value  # Bet
                elif action_char == 'F':
                    action = Action.BET_FOLD.value  # Fold
                
                env.step(action)
            
            payoffs = [env.get_payoff(0), env.get_payoff(1)]
            print(f"  {pattern}: {pattern_descriptions[pattern]:25s} → P0={payoffs[0]:+2d}, P1={payoffs[1]:+2d}")
        
    except (ValueError, KeyboardInterrupt, EOFError):
        print("❌ Invalid input or interrupted.")


def main_menu():
    """Main interactive menu"""
    while True:
        print(f"\n{'='*60}")
        print(f"🎯 KUHN POKER ENVIRONMENT EXPLORER")
        print(f"{'='*60}")
        print("1. 🎮 Play manual game (vs random opponent)")
        print("2. 📊 Explore all scenarios")
        print("3. 🧠 Analyze information sets") 
        print("4. 💰 Payoff analysis")
        print("5. 🔬 Test specific scenario")
        print("6. 👋 Exit")
        
        try:
            choice = input("\nChoose option (1-6): ").strip()
            
            if choice == "1":
                play_manual_game()
            elif choice == "2":
                explore_all_scenarios()
            elif choice == "3":
                analyze_information_sets()
            elif choice == "4":
                payoff_analysis()
            elif choice == "5":
                test_specific_scenario()
            elif choice == "6":
                print("👋 Thanks for exploring Kuhn Poker!")
                break
            else:
                print("❌ Invalid choice. Enter 1-6.")
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break


if __name__ == "__main__":
    print("🃏 Welcome to the Kuhn Poker Environment Explorer!")
    print("This tool helps you understand the game mechanics before implementing MCCFR.")
    main_menu()