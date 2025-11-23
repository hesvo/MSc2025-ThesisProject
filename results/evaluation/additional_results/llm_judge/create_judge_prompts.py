import os
import csv
import sys

# --- Configuration ---
TEMPLATE_FILE_PATH = "judge_template.txt"
CSV_FILE_PATH = "judge_data_base.csv" # Set to "benchmark_dataset_subset.csv" for benchmark subset
BASE_OUTPUT_DIR = "judge_prompts"

def load_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        sys.exit(1)

def create_prompts():

    # Prompt template
    print(f"Loading prompt template from '{TEMPLATE_FILE_PATH}'...")
    prompt_template = load_text_file(TEMPLATE_FILE_PATH)

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to the '{BASE_OUTPUT_DIR}' directory.")


    # Process input CSV file
    print(f"Reading data from '{CSV_FILE_PATH}'...")
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for i, row in enumerate(reader):
                try:
                    category = row['category']
                    rank = row['rank']
                    language = row['language']
                    file_id = row['id']
                    reference_code = row['reference']
                    generated_code = row['generated_rci' if 'generated_rci' in row else 'generated']
                                        
                    filled_prompt = prompt_template

                    filled_prompt = filled_prompt.replace("<reference code>", reference_code)
                    filled_prompt = filled_prompt.replace("<generated code>", generated_code)

                    # Save the final prompt using name='id'
                    output_filename = f"{category}_rank-{rank}_prob-{file_id}.txt"
                    output_filepath = os.path.join(BASE_OUTPUT_DIR, output_filename)
                    
                    with open(output_filepath, 'w', encoding='utf-8') as out_file:
                        out_file.write(filled_prompt)
                        
                except KeyError as e:
                    print(f"  -> ERROR: Row {i+2} is missing a required column: {e}. Skipping row.")
                except Exception as e:
                    print(f"  -> ERROR: An unexpected error occurred on row {i+2}: {e}. Skipping row.")

    except FileNotFoundError:
        print(f"ERROR: CSV file not found '{CSV_FILE_PATH}'.")
        sys.exit(1)

    print(f"\nPrompt generation complete. Prompts are located in '{BASE_OUTPUT_DIR}'.")

def main():
    create_prompts()

if __name__ == "__main__":
    main()