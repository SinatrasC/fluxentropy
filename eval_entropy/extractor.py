import json
import pandas as pd
import os
import argparse

def process_entropy_file(entropy_file, results_file, output_csv, required_fields=None):
    # Debug: Check if files exist
    print(f"Checking if files exist...")
    print(f"Entropy file ({entropy_file}): {os.path.exists(entropy_file)}")
    print(f"Results file ({results_file}): {os.path.exists(results_file)}")

    if not os.path.exists(entropy_file):
        raise FileNotFoundError(f"Entropy file not found at {entropy_file}")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found at {results_file}")

    # Define minimum required fields if not specified
    if required_fields is None:
        required_fields = ['index', 'entropy', 'text']

    # Step 1: Read the entropy dataset
    print("Loading entropy dataset...")
    with open(entropy_file, 'r', encoding='utf-8') as f:
        entropy_data = json.load(f)
    
    # More flexible dataset extraction
    if isinstance(entropy_data, dict):
        for key in ['sorted_dataset', 'dataset_analysis', 'dataset']:
            if key in entropy_data:
                dataset = entropy_data[key]
                break
        else:
            dataset = entropy_data
    else:
        dataset = entropy_data

    # Validate required fields
    sample_entry = dataset[0] if dataset else {}
    print("Available fields in dataset:", list(sample_entry.keys()))
    missing_fields = [field for field in required_fields if field not in sample_entry]
    if missing_fields:
        raise ValueError(f"Missing required fields in dataset: {missing_fields}")

    print(f"Loaded {len(dataset)} entries from entropy dataset.")

    # Step 2: Read the results CSV into a DataFrame
    print("Loading results CSV...")
    results_df = pd.read_csv(results_file)
    print(f"Loaded {len(results_df)} rows from results CSV.")

    # Ensure the column exists
    prompt_column = 'Prompt with Correct Choice'
    correct_answer_column = 'Correctly Answered'

    if prompt_column not in results_df.columns:
        raise ValueError(f"Column '{prompt_column}' not found in results CSV.")
    if correct_answer_column not in results_df.columns:
        raise ValueError(f"Column '{correct_answer_column}' not found in results CSV.")

    results_df[prompt_column] = results_df[prompt_column].astype(str)

    # Prepare a list to store the output data
    output_data = []

    # Step 3: Match entries and associate 'Correctly Answered'
    print("Matching entropy entries with results CSV...")
    matches_found = 0
    for entry in dataset:
        index = entry.get('index')
        entropy = entry.get('entropy')
        varentropy = entry.get('varentropy')
        text = entry.get('text', '').strip()

        if not text:
            print(f"Skipping entry with index {index} due to empty text.")
            correctly_answered = None
        else:
            # Try different matching strategies
            matched_rows = results_df[results_df[prompt_column] == text]
            
            if matched_rows.empty:
                matched_rows = results_df[results_df[prompt_column].str.contains(text, regex=False, na=False)]
            
            if matched_rows.empty:
                matched_rows = results_df[results_df[prompt_column].str.lower().str.contains(text.lower(), regex=False, na=False)]

            if not matched_rows.empty:
                matched_row = matched_rows.iloc[0]
                correctly_answered = matched_row[correct_answer_column]
                matches_found += 1
                print(f"Match found for index {index}. Correctly Answered: {correctly_answered}")
            else:
                correctly_answered = None
                print(f"No match found for index {index}.")

        # More flexible output entry creation
        output_entry = {k: entry.get(k) for k in entry.keys()}
        output_entry['Correctly Answered'] = correctly_answered
        output_data.append(output_entry)

    print(f"Total matches found: {matches_found} out of {len(dataset)}")

    # Step 5: Write the output data to a CSV file
    print(f"Writing output data to {output_csv}...")
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Data has been extracted and saved to {output_csv}.")

def main():
    parser = argparse.ArgumentParser(description='Process entropy files and match with results.')
    parser.add_argument('--entropy-file', required=True,
                       help='Path to the entropy JSON file')
    parser.add_argument('--results-file', required=True,
                       help='Path to the results CSV file')
    parser.add_argument('--output-csv', required=True,
                       help='Path for the output CSV file')
    parser.add_argument('--required-fields', nargs='+', default=['index', 'entropy', 'text'],
                       help='Required fields in the entropy dataset')
    
    args = parser.parse_args()
    
    process_entropy_file(
        args.entropy_file,
        args.results_file,
        args.output_csv,
        required_fields=args.required_fields
    )

if __name__ == "__main__":
    main()
