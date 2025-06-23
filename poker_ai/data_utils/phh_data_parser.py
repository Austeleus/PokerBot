import pandas as pd
import ast
from pathlib import Path
import itertools

# Action mapping for a discrete action space.
ACTION_MAP = {"f": 0, "cc": 1}  # 0: FOLD, 1: CHECK/CALL


def get_raise_category(raise_amount, pot_size_before_action):
    """Categorizes a raise into discrete buckets."""
    if pot_size_before_action == 0:
        return 2

    ratio = raise_amount / pot_size_before_action
    if ratio <= 0.75:
        return 2  # Raise Small (<= 0.75x pot)
    elif ratio <= 1.25:
        return 3  # Raise Medium (0.75x - 1.25x pot)
    else:
        return 4  # Raise Large (> 1.25x pot)


def parse_hand_data(hand_data: dict) -> list:
    """
    Parses a single hand dictionary into a list of state-action records.
    (This function is correct and remains unchanged)
    """
    decision_points = []

    # --- 1. Initial Game State Setup ---
    num_players = len(hand_data['players'])
    stacks = list(hand_data['starting_stacks'])
    player_names = list(hand_data['players'])
    hole_cards = {i: () for i in range(num_players)}

    # --- 2. Simulate Pre-Flop Actions (Blinds/Antes) ---
    pot_size = sum(hand_data['antes'])
    for i in range(num_players):
        stacks[i] -= hand_data['antes'][i]

    small_blind, big_blind = hand_data['blinds_or_straddles'][:2]
    sb_player_idx, bb_player_idx = 0, 1

    stacks[sb_player_idx] -= small_blind
    stacks[bb_player_idx] -= big_blind
    pot_size += small_blind + big_blind

    bets_in_round = {i: 0 for i in range(num_players)}
    bets_in_round[sb_player_idx] = small_blind
    bets_in_round[bb_player_idx] = big_blind

    current_bet_level = big_blind
    community_cards = []
    round_action_history = []

    # --- 3. Process Action Sequence ---
    for action_str in hand_data['actions']:
        parts = action_str.split()

        if parts[0] == 'd':
            if parts[1] == 'dh':
                player_idx = int(parts[2][1:]) - 1
                cards_str = parts[3]
                hole_cards[player_idx] = (cards_str[:2], cards_str[2:])
            elif parts[1] == 'db':
                current_bet_level = 0
                bets_in_round = {i: 0 for i in range(num_players)}
                round_action_history = []
                cards_str = "".join(parts[2:])
                for i in range(0, len(cards_str), 2):
                    community_cards.append(cards_str[i:i + 2])
            continue

        player_idx = int(parts[0][1:]) - 1
        action_code = parts[1]

        state = {
            "hand_id": hand_data['hand'], "player_id": player_idx,
            "player_name": player_names[player_idx], "position": player_idx,
            "hole_card_1": hole_cards[player_idx][0] if hole_cards[player_idx] else None,
            "hole_card_2": hole_cards[player_idx][1] if hole_cards[player_idx] else None,
            "community_card_1": community_cards[0] if len(community_cards) > 0 else None,
            "community_card_2": community_cards[1] if len(community_cards) > 1 else None,
            "community_card_3": community_cards[2] if len(community_cards) > 2 else None,
            "community_card_4": community_cards[3] if len(community_cards) > 3 else None,
            "community_card_5": community_cards[4] if len(community_cards) > 4 else None,
            "pot_size_before_action": pot_size,
            "amount_to_call": current_bet_level - bets_in_round[player_idx],
            "action_history_in_round": " ".join(round_action_history)
        }
        for i in range(num_players):
            state[f'p{i + 1}_stack'] = stacks[i]

        action_label = -1
        if action_code == 'f':
            action_label = ACTION_MAP['f']
        elif action_code == 'cc':
            amount_to_call = current_bet_level - bets_in_round[player_idx]
            if amount_to_call > 0:
                call_amount = min(amount_to_call, stacks[player_idx])
                stacks[player_idx] -= call_amount
                bets_in_round[player_idx] += call_amount
                pot_size += call_amount
            action_label = ACTION_MAP['cc']
        elif action_code == 'cbr':
            total_bet_amount = int(parts[2])
            amount_already_in_round = bets_in_round[player_idx]
            raise_amount = total_bet_amount - current_bet_level
            action_label = get_raise_category(raise_amount, state['pot_size_before_action'])

            amount_to_add_to_pot = total_bet_amount - amount_already_in_round
            stacks[player_idx] -= amount_to_add_to_pot
            pot_size += amount_to_add_to_pot
            bets_in_round[player_idx] = total_bet_amount
            current_bet_level = total_bet_amount

        state['action_taken'] = action_label
        state['action_str'] = action_str
        decision_points.append(state)

        round_action_history.append(f"p{player_idx + 1}_{action_code}")

    return decision_points


def _parse_hand_block(hand_block_lines: list) -> list:
    """
    Parses a list of strings representing a single hand block.
    """
    hand_text = "\n".join(hand_block_lines)
    hand_data = {}
    try:
        for line in hand_text.strip().split('\n'):
            if '=' in line:
                key, value_str = line.split('=', 1)

                # --- THIS IS THE FIX ---
                # Replace lowercase 'true'/'false' with Python's uppercase equivalents
                processed_value_str = value_str.strip().replace('true', 'True').replace('false', 'False')

                hand_data[key.strip()] = ast.literal_eval(processed_value_str)

        if not hand_data:
            return []

        return parse_hand_data(hand_data)
    except Exception as e:
        print(f"---Skipping a block due to parsing error: {e}---")
        print(f"Problematic Text:\n{hand_text[:300]}...\n")
        return []


def parse_phh_file(file_path: str or Path) -> list:
    """
    Reads a PHH file line-by-line and parses all hands within it.
    (This function is correct and remains unchanged)
    """
    all_decision_points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_hand_lines = []
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if stripped_line.startswith('variant ='):
                if current_hand_lines:
                    hand_decision_points = _parse_hand_block(current_hand_lines)
                    all_decision_points.extend(hand_decision_points)

                current_hand_lines = [stripped_line]
            else:
                current_hand_lines.append(stripped_line)

        if current_hand_lines:
            hand_decision_points = _parse_hand_block(current_hand_lines)
            all_decision_points.extend(hand_decision_points)

    return all_decision_points