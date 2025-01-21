import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from common_fx import process_file_content, generate_content

# Initialize the model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def load_existing_data(file_path):
    """Load existing embeddings from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading file: {e}")
    return {}

def embed_text(text):
    """Generate embeddings for a given text using LEGAL-BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    outputs = model(**inputs)
    # Pooling: Take the mean across the sequence length (dim=1)
    pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return pooled_embedding  # Shape: [embedding_size]



def embed_sections(case_summary):
    """Generate embeddings for each section in the case summary."""
    embeddings = {}
    sections = case_summary.split('###')  # Split the summary into sections
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n', 1)
            title = lines[0].strip()  # Extract the section title
            content = lines[1].strip() if len(lines) > 1 else ""
            embeddings[title] = embed_text(content)  # 1D array for each section
    return embeddings


def save_embeddings(file_path, case_id, section_embeddings):
    """Save or update section embeddings for a case in the JSON file."""
    data = load_existing_data(file_path)
    serializable_embeddings = {k: v.tolist() for k, v in section_embeddings.items()}
    data[case_id] = serializable_embeddings
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2, separators=(',', ': '), ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")

def find_section_similarities(case_embeddings, target_case_embeddings):
    """Compute cosine similarity for each common section between two cases."""
    section_similarities = {}
    common_sections = set(case_embeddings.keys()).intersection(target_case_embeddings.keys())
    for section in common_sections:
        case_section_embedding = case_embeddings[section]
        target_section_embedding = target_case_embeddings[section]
        similarity = cosine_similarity([case_section_embedding], [target_section_embedding])[0][0]
        section_similarities[section] = similarity
    return section_similarities

def find_most_similar_cases(case_id, target_case_embeddings, existing_embeddings, top_n=3):
    """Find the top N most similar cases to the given case based on cosine similarity."""
    similarities = []
    
    # Convert target embeddings to numpy array and ensure it's 2D
    target_embeddings = np.array(list(target_case_embeddings.values()))
    target_avg_embedding = np.mean(target_embeddings, axis=0)
    
    for other_case_id, other_case_embeddings in existing_embeddings.items():
        if other_case_id != case_id:
            # Convert other case embeddings to numpy array and ensure it's 2D
            other_embeddings = np.array(list(other_case_embeddings.values()))
            other_avg_embedding = np.mean(other_embeddings, axis=0)
            
            # Reshape embeddings to 2D arrays (needed for cosine_similarity)
            target_reshaped = target_avg_embedding.reshape(1, -1)
            other_reshaped = other_avg_embedding.reshape(1, -1)
            
            # Calculate similarity
            similarity = cosine_similarity(target_reshaped, other_reshaped)[0][0]
            similarities.append((other_case_id, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def summary(file_path, case_id, output):
    """Generate and save embeddings for a case summary."""
    section_embeddings = embed_sections(output)
    save_embeddings(file_path, case_id, section_embeddings)

def similar(file_path, case_id):
    """Find and log the most similar cases to a given case."""
    existing_embeddings = load_existing_data(file_path)
    if case_id not in existing_embeddings:
        raise ValueError(f"Case {case_id} does not have an embedding. Summarize the case first.")
    
    target_case_embeddings = existing_embeddings[case_id]
    top_similar_cases = find_most_similar_cases(case_id, target_case_embeddings, existing_embeddings)
    
    # Print results for logging
    print("Top similar cases:")
    for similar_case, similarity_score in top_similar_cases:
        print(f"Case {similar_case}: Similarity {similarity_score:.4f}")

        prompt = f"Compare these two cases based on the situations and contexts they describe, not just the exact wording or phrasing.\nCase 1: {case_id}\n{process_file_content(case_id, False)}\nCase 2: {similar_case}\n{process_file_content(similar_case, False)}\n\nTasks:\n1. In one paragraph, explain how the two cases are similar, focusing on the underlying situations, challenges, and dynamics.\n2. In another paragraph, explain how the two cases are different, highlighting key distinctions in context or circumstances.\n3. Provide an integer score between 1 and 10 to rate their similarity, where:\n\t- 1 means the cases are completely unrelated.\n\t- 10 means the cases are almost identical in terms of situations and context."
        generate_content(prompt)
    
    # Return the results
    return top_similar_cases

if __name__ == "__main__":
    try:
        # Capture the arguments passed to the script
        case_id = sys.argv[1]
        output = sys.argv[2]
        str_button = sys.argv[3]

        # Validate inputs
        if not case_id:
            raise ValueError("case_id cannot be empty (embed.py).")
        if "not contain notes in Client 360" in output:
            raise ValueError(f"There is no notes or detailed information about case {case_id} and cannot do anything with this information(embed.py).")

        main(case_id, output, str_button)
    except Exception as e:
        logger.error(f"An error occurred (embed.py): {e}")

def main(case_id, output, str_button):
    # Log inputs
    # log_inputs(case_id, output, str_button)

    file_path = "C:\\Users\\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\Work\\embeddings.json"

    if str_button == "summary":
        summary(file_path, case_id, output)
    elif str_button == "similar":
        top_similar = similar(file_path, case_id)
        return top_similar
