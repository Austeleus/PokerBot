import os
import re
import pandas as pd
from tqdm import tqdm
import json
import ast  # Use Abstract Syntax Tree to safely evaluate literals

# --- Constants and Mappings ---

# Card and player constants
RANKS = '23456789TJQKA'
SUITS = 'shdc'
CARDS = [r + s for r in RANKS for s in SUITS]
CARD_TO_INDEX = {card: i for i, card in enumerate(CARDS)}
NUM_CARDS = len(CARDS)
MAX_PLAYERS = 6


# --- Encoding Functions ---

def encode_card(card_str):
    """Converts a card string (e.g., 'As', 'Td') into a 52-element one-hot encoded vector."""
    vector = [0] * NUM_CARDS
    if card_str and card_str in CARD_TO_INDEX:
        vector[CARD_TO_INDEX[card_str]] = 1
    return vector


# --- New Parsing Logic for Machine-Readable .phh Format ---

def parse_phh_file_content(content):
    """
    Parses the string content of a .phh file into a Python dictionary.
    Example line: "blinds_or_straddles = [50, 100, 0, 0, 0, 0]"
    """
    data = {}
    for line in content.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                # Safely evaluate the string as a Python literal
                data[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Handle plain strings that aren't literals (like variant = 'NT')
                data[key] = value
    return data


def parse_hand(file_path, hero_name="Pluribus"):
    """
    Parses a single machine-readable .phh file and extracts state-action pairs for the hero.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    hand_data = parse_phh_file_content(content)

    # --- Initial Game State Setup ---
    try:
        player_names = hand_data['players']
        if hero_name not in player_names:
            return []

        num_players = len(player_names)
        starting_stacks = hand_data['starting_stacks']
        small_blind, big_blind = hand_data['blinds_or_straddles'][:2]

        hero_player_index = player_names.index(hero_name)
    except KeyError:
        return []  # Skip malformed files

    # --- State Tracking Variables ---
    current_stacks = list(starting_stacks)
    pot_size = 0
    community_cards = []
    hole_cards = [[] for _ in range(num_players)]
    player_bets = [0] * num_players  # Total bet by each player in the hand
    round_bets = [0] * num_players  # Bet by each player in the current round
    players_in_hand = [True] * num_players

    # Position calculation (0=SB, 1=BB, etc. relative to button)
    # The dataset description implies a fixed button for simplicity in some cases.
    # We will assume button rotates based on hand number for a more general approach.
    button_pos = hand_data.get('hand', 0) % num_players
    positions = {i: ((i - button_pos - 1 + num_players) % num_players) for i in range(num_players)}

    decision_points = []

    # --- Simulate Hand by Processing Actions ---
    for action_str in hand_data['actions']:
        parts = action_str.split()

        # --- Handle Dealing Actions ---
        if parts[0] == 'd':  # Deal action
            if parts[1] == 'dh':  # Deal Hole Cards
                p_idx = int(parts[2][1:]) - 1
                hole_cards[p_idx] = list(re.findall('..', parts[3]))
            elif parts[1] == 'db':  # Deal Board Cards
                community_cards.extend(list(re.findall('..', parts[2])))
            continue

        # --- Handle Player Actions ---
        p_idx = int(parts[0][1:]) - 1
        action_code = parts[1]

        # This is a decision point for the hero
        if p_idx == hero_player_index:
            # --- CAPTURE STATE ---
            # 1. Hole Cards
            hero_hole_cards = hole_cards[hero_player_index]
            hole_cards_encoded = encode_card(hero_hole_cards[0]) + encode_card(hero_hole_cards[1])

            # 2. Community Cards (padded to 5)
            community_cards_encoded = [0] * (5 * NUM_CARDS)
            for i, card in enumerate(community_cards):
                offset = i * NUM_CARDS
                community_cards_encoded[offset:offset + NUM_CARDS] = encode_card(card)

            # 3. Stacks, Pot, and Bet to Call (normalized by big blind)
            pot_size_norm = pot_size / big_blind
            stacks_norm = [s / big_blind if players_in_hand[i] else 0 for i, s in enumerate(current_stacks)]

            current_round_max_bet = max(round_bets) if round_bets else 0
            bet_to_call = current_round_max_bet - round_bets[p_idx]
            bet_to_call_norm = bet_to_call / big_blind

            # 4. Position
            hero_pos_encoded = [0] * MAX_PLAYERS
            hero_pos_encoded[positions[hero_player_index]] = 1

            # Assemble state vector
            state_vector = (
                    hole_cards_encoded +
                    community_cards_encoded +
                    stacks_norm +
                    [pot_size_norm, bet_to_call_norm] +
                    hero_pos_encoded
            )

            # --- CAPTURE ACTION ---
            # Simplified action mapping: fold, check/call, bet/raise
            if action_code == 'f':
                action_label = 0
                action_name = 'fold'
            elif action_code == 'cc':
                action_label = 1
                action_name = 'check_call'
            elif action_code == 'cbr':
                action_label = 2
                action_name = 'bet_raise'
            else:
                continue  # Skip unknown actions

            raise_amount = int(parts[2]) if len(parts) > 2 else 0

            decision_points.append({
                "hand_id": hand_data.get('hand', 'N/A'),
                "state_vector": json.dumps(state_vector),
                "action_label": action_label,
                "action_name": action_name,
                "raise_amount_norm": raise_amount / big_blind
            })

        # --- UPDATE STATE after every player action ---
        if action_code == 'f':
            players_in_hand[p_idx] = False

        elif action_code == 'cc':  # check/call
            current_round_max_bet = max(round_bets) if round_bets else 0
            amount_to_call = current_round_max_bet - round_bets[p_idx]

            current_stacks[p_idx] -= amount_to_call
            pot_size += amount_to_call
            round_bets[p_idx] += amount_to_call

        elif action_code == 'cbr':  # call/bet/raise
            bet_amount = int(parts[2])
            amount_to_pot = bet_amount - round_bets[p_idx]

            current_stacks[p_idx] -= amount_to_pot
            pot_size += amount_to_pot
            round_bets[p_idx] += amount_to_pot

    return decision_points


def create_dataset(root_dir, output_csv, hero_name="Pluribus"):
    """Walks a directory of hand histories, parses them, and saves the aggregated data."""
    all_decisions = []
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found at '{root_dir}'")
        return

    filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if
                 f.endswith('.phh')]
    print(f"Found {len(filepaths)} hand history files.")

    for file_path in tqdm(filepaths, desc="Parsing hands"):
        try:
            hand_decisions = parse_hand(file_path, hero_name)
            all_decisions.extend(hand_decisions)
        except Exception as e:
            # print(f"Skipping {os.path.basename(file_path)} due to error: {e}")
            continue

    df = pd.DataFrame(all_decisions)

    if df.empty:
        print("\nParsing complete, but no decision points were found.")
        print(f"Please check that the directory '{root_dir}' contains valid .phh files for the hero '{hero_name}'.")
        return

    df.dropna(subset=['action_label'], inplace=True)
    df['action_label'] = df['action_label'].astype(int)

    print(f"\nSuccessfully parsed {len(df)} decision points.")
    print("Action distribution:")
    print(df['action_name'].value_counts(normalize=True))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")


if __name__ == '__main__':
    try:
        script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(script_path))
    except NameError:
        project_root = os.getcwd()

    data_directory = os.path.join(project_root, 'data', 'hand_histories', 'pluribus')
    output_file = os.path.join(project_root, 'data', 'pluribus_dataset.csv')

    print(f"Looking for hand histories in: {data_directory}")
    print(f"Will save dataset to: {output_file}")

    create_dataset(data_directory, output_file, hero_name="Pluribus")