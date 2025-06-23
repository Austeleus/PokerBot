from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

# Add the project root to the Python path
# This allows us to import from the poker_bot package
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from poker_ai.data_utils.phh_data_parser import parse_phh_file


def main():
    """
    Parses all .phh files found recursively in the raw data directory
    and saves the combined results to a single CSV file.
    """
    raw_data_dir = project_root / 'data' / 'hand_histories' / 'pluribus'
    processed_data_dir = project_root / 'data' / 'processed'

    # Ensure the output directory exists
    processed_data_dir.mkdir(exist_ok=True)

    # --- THIS IS THE MODIFIED LINE ---
    # Use glob with '**/' to search recursively through all subdirectories for .phh files.
    print(f"Recursively searching for .phh files in: {raw_data_dir}")
    phh_files = list(raw_data_dir.glob('**/*.phh'))

    if not phh_files:
        print(f"Error: No .phh files found in {raw_data_dir} or its subdirectories.")
        print("Please ensure your Pluribus hand history folders are placed there.")
        return

    print(f"Found {len(phh_files)} .phh files to parse.")

    all_decision_points = []

    for file_path in tqdm(phh_files, desc="Parsing PHH files"):
        decisions = parse_phh_file(file_path)
        all_decision_points.extend(decisions)

    if not all_decision_points:
        print("Error: No decision points were parsed. Check the format of your .phh files.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_decision_points)

    # Save to CSV
    output_path = processed_data_dir / 'state_action_pairs.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSuccessfully parsed a total of {len(df)} decision points.")
    print(f"Data saved to {output_path}")
    print("\nSample of the final data:")
    # Use to_markdown if you have tabulate installed, otherwise use standard print
    try:
        print(df.head().to_markdown(index=False))
    except ImportError:
        print(df.head())


if __name__ == '__main__':
    main()