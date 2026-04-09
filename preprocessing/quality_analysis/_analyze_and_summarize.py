import json
import os
import time
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# ---
# Gemini API call
# ---
def analyze_content_with_gemini(file_content, prompt_template, api_key):
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    client = genai.Client(api_key=api_key)

    try:
        prompt = prompt_template.format(file_content=file_content) + "\n\nPlease provide the output in a valid JSON format."
        response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)

        # Clean the response to ensure it's valid JSON
        clean_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(clean_response)
    except Exception as e:
        print(f"\nError analyzing content: {e}")
        return {
            "semantic_noise_score": 0,
            "speaker_attribution_score": 0,
            "document_atomicity_score": 0,
            "textual_fidelity_score": 0,
            "suggested_actions": ["Error analyzing content."],
            "notes": str(e)
        }

# ---
# Main analysis script
# ---
def run_analysis(root_dir, analysis_file, output_file, api_key):
    BATCH_SIZE = 10

    # Load existing results if the output file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            analysis_results = json.load(f)
    else:
        analysis_results = {}

    # Create a set of already processed files for quick lookup
    processed_files = set()
    for category in analysis_results:
        if isinstance(analysis_results[category], list):
            for item in analysis_results[category]:
                processed_files.add(item['file_path'])

    with open(analysis_file, 'r') as f:
        file_data = json.load(f)

    total_files = sum(len(files) for files in file_data.values())
    print(f"Found {total_files} total files. {len(processed_files)} already processed.")

    files_to_process = []
    for category, files in file_data.items():
        if category not in analysis_results:
            analysis_results[category] = []
        for file_info in files:
            if file_info["file_path"] not in processed_files:
                files_to_process.append((category, file_info))

    processed_in_session = 0
    for i, (category, file_info) in enumerate(files_to_process):
        relative_path = file_info["file_path"]
        print(f"[{len(processed_files) + i + 1}/{total_files}] Analyzing: {relative_path}")

        file_path = Path(root_dir) / relative_path
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}")
            content = ""

        prompt = transcript_prompt if category == "01_Audio_Transcripts" else other_document_prompt
        
        scores = analyze_content_with_gemini(content, prompt, api_key)
        
        file_info.update(scores)
        analysis_results[category].append(file_info)
        processed_in_session += 1

        # Save progress every BATCH_SIZE files
        if processed_in_session % BATCH_SIZE == 0:
            print(f"\n--- Saving progress ({processed_in_session} files processed in this session) ---")
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            print("--- Progress saved ---\n")

        time.sleep(1)

    # Final save for any remaining files
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    ROOT_DIR = "/Users/aban/drive/Projects/PeatLearn/raw_data"
    ANALYSIS_FILE = "/Users/aban/drive/Projects/PeatLearn/file_analysis.json"
    OUTPUT_FILE = "/Users/aban/drive/Projects/PeatLearn/file_analysis_with_scores.json"

    transcript_prompt = '''
You are an expert in data quality analysis. I will provide you with the content of a file, and you will analyze it based on the following criteria: semantic noise, speaker attribution, document atomicity, and textual fidelity. You will then provide a JSON object with the following information:

*   `semantic_noise_score`: A rating from 1 to 10, where 1 is very low noise and 10 is very high noise.
*   `speaker_attribution_score`: A rating from 1 to 10, where 1 is very clear attribution and 10 is very unclear attribution.
*   `document_atomicity_score`: A rating from 1 to 10, where 1 is a single, self-contained document and 10 is a document with many distinct parts.
*   `textual_fidelity_score`: A rating from 1 to 10, where 1 is very high fidelity and 10 is very low fidelity.
*   `suggested_actions`: A list of suggested cleaning actions.
*   `notes`: Any other relevant notes.

Here is the file content:

{file_content}
'''

    other_document_prompt = '''
You are an expert in data quality analysis. I will provide you with the content of a file, and you will analyze it based on the following criteria: semantic noise, document atomicity, and textual fidelity. You will then provide a JSON object with the following information:

*   `semantic_noise_score`: A rating from 1 to 10, where 1 is very low noise and 10 is very high noise.
*   `document_atomicity_score`: A rating from 1 to 10, where 1 is a single, self-contained document and 10 is a document with many distinct parts.
*   `textual_fidelity_score`: A rating from 1 to 10, where 1 is very high fidelity and 10 is very low fidelity.
*   `suggested_actions`: A list of suggested cleaning actions.
*   `notes`: Any other relevant notes.

Here is the file content:

{file_content}
'''
    
    run_analysis(ROOT_DIR, ANALYSIS_FILE, OUTPUT_FILE, api_key)
