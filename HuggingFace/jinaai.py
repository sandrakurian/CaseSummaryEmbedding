# https://huggingface.co/jinaai/jina-embeddings-v2-base-en

from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

texts = [
    "Follow the white rabbit.",  # English
    "Siga al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# Tokenize and encode the text
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # Assuming embeddings are taken from the last hidden state or pooled output
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling as an example

# Normalize embeddings for cosine similarity
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

# Compute similarities (dot product is equivalent to cosine similarity here)
similarity_matrix = torch.matmul(embeddings, embeddings.T)

# Example: similarity between the first and second text
similarity = similarity_matrix[0, 1].item()
print(f"Similarity between the first and second text: {similarity:.4f}")
