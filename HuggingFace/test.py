import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CaseSummary:
    def __init__(self, case_number, summary, embedding=None):
        self.case_number = case_number
        self.summary = summary
        self.embedding = embedding if embedding is not None else []

    def to_dict(self):
        return {
            "CaseNumber": self.case_number,
            "Summary": self.summary,
            "Embedding": self.embedding,
        }

    @staticmethod
    def from_dict(data):
        return CaseSummary(
            case_number=data["CaseNumber"],
            summary=data["Summary"],
            embedding=data["Embedding"],
        )


class TextEmbedding:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_model(self, texts):
        self.vectorizer.fit(texts)

    def get_text_embedding(self, text):
        embedding = self.vectorizer.transform([text]).toarray()
        return embedding.flatten().tolist()

    def save_embeddings_to_file(self, file_path, case_summaries):
        existing_summaries = []

        # Load existing summaries if file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing_summaries = [
                    CaseSummary.from_dict(item)
                    for item in json.load(f)
                ]

        # Update or append new summaries
        for new_case in case_summaries:
            existing_case = next(
                (cs for cs in existing_summaries if cs.case_number == new_case.case_number),
                None,
            )
            if existing_case:
                existing_case.summary = new_case.summary
                existing_case.embedding = new_case.embedding
            else:
                existing_summaries.append(new_case)

        # Save updated summaries to file
        with open(file_path, "w") as f:
            json.dump([cs.to_dict() for cs in existing_summaries], f, indent=4)

    def get_top_similar_cases(self, query_summary, file_path, n=3):
        # Generate embedding for query
        query_embedding = self.get_text_embedding(query_summary)

        # Load existing summaries
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        with open(file_path, "r") as f:
            existing_summaries = [
                CaseSummary.from_dict(item)
                for item in json.load(f)
            ]

        # Compute similarities
        similarities = []
        for case in existing_summaries:
            if case.embedding:
                similarity = cosine_similarity(
                    [query_embedding], [case.embedding]
                )[0, 0]
                similarities.append((case, similarity))

        # Return top N similar cases
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]


def embed_from_csv_and_save(embedding_model, csv_file_path, json_file_path):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    case_summaries = []
    for _, row in df.iterrows():
        case_number = row["CaseNumber"]
        summary = row["Summary"]
        embedding = embedding_model.get_text_embedding(summary)
        case_summaries.append(CaseSummary(case_number, summary, embedding))

    # Save embeddings to file
    embedding_model.save_embeddings_to_file(json_file_path, case_summaries)


if __name__ == "__main__":
    embedding_model = TextEmbedding()

    # Fit the embedding model with some dummy data
    dummy_data = [
        "James Lewis was charged with murder. He allegedly killed his business partner over a financial dispute. Forensic evidence and witness statements were critical in building the case. Lewis claimed self-defense, but the evidence suggested premeditation. He was found guilty and given a life sentence.",
        "Emily Clark faced charges of human trafficking. She was implicated in a ring that exploited vulnerable individuals for labor. Testimonies from victims and intercepted communications were pivotal.",
    ]
    embedding_model.fit_model(dummy_data)

    # Define file paths
    json_file_path = "CaseSummaries.json"
    csv_file_path = "C:\\Users\\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\CSharpMethod\\CaseSummaries.csv"

    # Generate embeddings from CSV and save
    embed_from_csv_and_save(embedding_model, csv_file_path, json_file_path)

    # Query for similar cases
    query_summary = (
        "Public Health Advisors working in federal government positions focus on promoting the well-being of the population, particularly in response to health crises such as pandemics, natural disasters, or environmental hazards. They develop and implement health policies, coordinate with local health agencies, and track disease outbreaks to protect communities. Public Health Advisors often work with the Centers for Disease Control and Prevention (CDC) or other federal health agencies. The position demands expertise in epidemiology, health promotion, and crisis management. These professionals are also responsible for educating the public about health risks and preventative measures to improve overall community health outcomes."
    )
    num_similar = 3

    try:
        top_cases_with_similarity = embedding_model.get_top_similar_cases(
            query_summary, json_file_path, num_similar
        )

        print(f"Top {num_similar} Similar Cases:")
        for case, similarity in top_cases_with_similarity:
            print(f"Case {case.case_number}: {case.summary[:100]}")
            print(f"Similarity Score: {similarity:.4f}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
