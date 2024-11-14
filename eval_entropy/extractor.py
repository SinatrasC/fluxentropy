import json
import pandas as pd
import os

# Paths to the data files
entropy_file = os.path.join('data', 'entropy', 'hellaswag_pertoken_entropy_sorted.json')
results_file = os.path.join('data', 'samples', 'Enhanced_Results_with_Correct_Choice_in_Prompts.csv')
output_csv = 'entropy_text_correctly_answered.csv'

# Debug: Check if files exist
print(f"Checking if files exist...")
print(f"Entropy file ({entropy_file}): {os.path.exists(entropy_file)}")
print(f"Results file ({results_file}): {os.path.exists(results_file)}")

if not os.path.exists(entropy_file):
    raise FileNotFoundError(f"Entropy file not found at {entropy_file}")
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Results file not found at {results_file}")

# Step 1: Read the entropy dataset
print("Loading entropy dataset...")
with open(entropy_file, 'r', encoding='utf-8') as f:
    entropy_data = json.load(f)
print(f"Loaded {len(entropy_data['sorted_dataset'])} entries from entropy dataset.")

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
for entry in entropy_data['sorted_dataset']:
    index = entry.get('index')
    entropy = entry.get('entropy')
    text = entry.get('text', '').strip()  # Remove leading/trailing whitespace

    if not text:
        print(f"Skipping entry with index {index} due to empty text.")
        correctly_answered = None
    else:
        # Attempt to find a matching prompt
        # Adjust the matching strategy as per data characteristics
        # For example, if 'text' is a part of 'Prompt with Correct Choice', use that
        # Otherwise, consider alternative matching strategies

        # Debug: Print the text being matched
        print(f"Matching text for index {index}: '{text}'")

        # Here, assuming that 'text' should be a substring in the 'Prompt with Correct Choice'
        # However, based on your sample data, this might not be the case
        # Therefore, consider alternative matching strategies

        # Example alternative: Exact match
        matched_rows = results_df[results_df[prompt_column] == text]

        # If no exact match, try substring
        if matched_rows.empty:
            matched_rows = results_df[results_df[prompt_column].str.contains(text, regex=False, na=False)]

        # If still no match, try more flexible matching (e.g., case-insensitive)
        if matched_rows.empty:
            matched_rows = results_df[results_df[prompt_column].str.lower().str.contains(text.lower(), regex=False, na=False)]

        # If still no match, consider partial or fuzzy matching
        # (Requires additional libraries like fuzzywuzzy or RapidFuzz)

        if not matched_rows.empty:
            # If multiple matches, take the first one
            matched_row = matched_rows.iloc[0]
            correctly_answered = matched_row[correct_answer_column]
            matches_found += 1
            print(f"Match found for index {index}. Correctly Answered: {correctly_answered}")
        else:
            # No match found
            correctly_answered = None
            print(f"No match found for index {index}.")

    # Append the data to the output list
    output_data.append({
        'index': index,
        'entropy': entropy,
        'text': text,
        'Correctly Answered': correctly_answered
    })

print(f"Total matches found: {matches_found} out of {len(entropy_data['sorted_dataset'])}")

# Step 5: Write the output data to a CSV file
print(f"Writing output data to {output_csv}...")
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv, index=False)
print(f"Data has been extracted and saved to {output_csv}.")
