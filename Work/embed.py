import os
import sys
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from common_fx import process_file_content as process_file_content

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

# Set up logging
log_file = "script.log"


# Function to log the header with current time
def log_header():
    current_time = datetime.now().strftime("%H:%M")  # Get current time in hh:mm format
    logger.info(f"\n========== Log Session Started at {current_time} ==========\n")


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s"  # Only log the message without timestamp and level by default
)
logger = logging.getLogger()

# Log the header only once at the start of the log session
log_header()

def log_inputs(case_id, output, str_button):
    """Log info about the case being processed."""
    logger.info(f"Processing FSFNPersonID+case: {case_id}")
    logger.info(f"Processing with button: {str_button}")

    """
    # Split text into paragraphs
    paragraphs = output.split("\n")

    # Log first bit of each paragraph
    for i, paragraph in enumerate(paragraphs):
        truncated_paragraph = paragraph[:50]
        logger.info(f"{i+1}.\t{truncated_paragraph}")
    """


def load_existing_data(file_path):
    """
    Check if the file exists and load existing data.
    Returns contents/dictionary in file or empty dict {}
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading file: {e}")
    return {}

def embed_sections(case_summary):
    """
    Generate embeddings for each section in the case summary.
    Returns dictionary containing the section titles as keys and their corresponding embeddings
    """
    embeddings = {}
    sections = case_summary.split('###')  # Split the string by headers
    for section in sections:
        if section.strip():  # Skip empty sections
            lines = section.strip().split('\n', 1)
            title = lines[0].strip()  # Extract the title
            content = lines[1].strip() if len(lines) > 1 else ""  # Extract the content
            if content.startswith("-") or content.startswith("1."):  # Check if it's a list
                items = [item.strip() for item in content.split('\n') if item.strip()]
                embeddings[title] = np.mean([model.encode(item) for item in items], axis=0)
            else:  # Process as a paragraph
                embeddings[title] = model.encode(content)
    return embeddings

def save_embeddings(file_path, case_id, section_embeddings):
    """Save or update the section embeddings for a case in the JSON file."""
    data = load_existing_data(file_path)
    serializable_embeddings = {k: v.tolist() for k, v in section_embeddings.items()}

    # Add or update embeddings for the case
    data[case_id] = serializable_embeddings

    # Log if updating or adding a case
    if case_id in data:
        logger.info(f"Updating case: {case_id}")
    else:
        logger.info(f"Adding case: {case_id}")

    # Save updated data to the file
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2, separators=(',', ': '), ensure_ascii=False)
        logger.info(f"Embeddings for case {case_id} saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")

def embedding_exist(existing_embeddings, case_id):
    """
    Check if a given case_id exists in the existing embeddings.
    Returns (case_id, embedding) if found, otherwise None.
    """
    if case_id in existing_embeddings:
        return case_id, existing_embeddings[case_id]
    logger.warning(f"{case_id} does not have a summary embedding. Summarize the case before clicking 'sandra'")
    return None

def find_section_similarities(case_id, target_case_id, case_embeddings, target_case_embeddings):
    """
    Compares the embeddings of each common section between the case_id and target_case_id.
    Returns a dictionary where keys are section titles and values are cosine similarity scores.
    Also returns sections that were not compared.
    """
    section_similarities = {}
    not_compared_sections = []

    # Find common sections
    common_sections = set(case_embeddings.keys()).intersection(target_case_embeddings.keys())
    
    # Compare sections
    for section in common_sections:
        case_section_embedding = case_embeddings[section]
        target_case_section_embedding = target_case_embeddings[section]
        
        # Compute cosine similarity for the section
        similarity = cosine_similarity([case_section_embedding], [target_case_section_embedding])[0][0]
        section_similarities[section] = similarity

    # Find sections that were not compared
    not_compared_sections = list(set(case_embeddings.keys()).union(target_case_embeddings.keys()) - set(common_sections))

    return section_similarities, not_compared_sections

def find_most_similar_cases(case_id, target_case_embeddings, existing_embeddings, top_n=3):
    """
    Find the top N most similar cases to the given case based on cosine similarity.
    """
    similarities = []
    
    for other_case_id, other_case_embeddings in existing_embeddings.items():
        if other_case_id != case_id:  # Skip the case itself
            # Compute similarity for the entire case (you can modify this to compute section-level similarity)
            case_embedding = np.mean(list(target_case_embeddings.values()), axis=0)
            other_case_embedding = np.mean(list(other_case_embeddings.values()), axis=0)
            similarity = cosine_similarity([case_embedding], [other_case_embedding])[0][0]
            similarities.append((other_case_id, similarity))
    
    # Sort by similarity and return the top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def summary(file_path, case_id, output):
    # Generate embeddings
    section_embeddings = embed_sections(output)
    # Save embeddings to file
    save_embeddings(file_path, case_id, section_embeddings)

def sandra(file_path, case_id):
    # Load existing embeddings from the JSON file
    existing_embeddings = load_existing_data(file_path)
            
    # Check if the case_id exists in the existing embeddings
    if case_id not in existing_embeddings:
        raise ValueError(f"Case {case_id} does not have an embedding. You must first summarize the case before proceeding.")
            
    # Get the embedding for the current case
    target_case_embeddings = existing_embeddings[case_id]

    # Find the top 3 most similar cases
    top_similar_cases = find_most_similar_cases(case_id, target_case_embeddings, existing_embeddings)
    
    logger.info(f"Top 3 most similar cases to {case_id}:")
    logger.info(process_file_content(case_id, short=True))
    for similar_case, similarity_score in top_similar_cases:
        logger.info(f"\tCase {similar_case} Similarity: {similarity_score:.4f}")

    # Now compare the common sections between the current case and the most similar cases
    for similar_case, _ in top_similar_cases:
        similar_case_embeddings = existing_embeddings[similar_case]
        
        section_similarities, not_compared_sections = find_section_similarities(case_id, similar_case, target_case_embeddings, similar_case_embeddings)
        
        logger.info(f"\nSection similarities for case {similar_case}:")
        logger.info(process_file_content(similar_case, short=True))
        for section, similarity in section_similarities.items():
            logger.info(f"\tSection: {section}, Similarity: {similarity:.4f}")
                
        if not_compared_sections:
            logger.info("\tSections that were not compared:")
            for section in not_compared_sections:
                logger.info(f"\t\t{section}")
    
    return [item[0] for item in top_similar_cases]

if __name__ == "__main__":
    try:
        # Capture the arguments passed to the script
        case_id = sys.argv[1]
        output = sys.argv[2]
        str_button = sys.argv[3]

        # Validate inputs
        if not case_id:
            raise ValueError("case_id cannot be empty.")
        if "not contain notes in Client 360" in output:
            raise ValueError(f"There is no notes or detailed information about case {case_id} and cannot do anything with this information.")

        main(case_id, output, str_button)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main(case_id, output, str_button):
        # Log inputs
        log_inputs(case_id, output, str_button)

        file_path = "C:\\Users\\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\Work\\embeddings.json"

        if str_button == "summary":
            summary(file_path, case_id, output)
        elif str_button == "sandra":
            top_similar = sandra(file_path, case_id)
            return top_similar

