# Use a pipeline as a high-level helper
from transformers import pipeline
import torch


pipe = pipeline("feature-extraction", model="pooruss/xlm-roberta-large-qp-similarity")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("pooruss/xlm-roberta-large-qp-similarity")
model = AutoModel.from_pretrained("pooruss/xlm-roberta-large-qp-similarity")


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

# Loop through all pairs of texts and print similarities
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = similarity_matrix[i, j].item()
        print(f"Similarity between text {i + 1} and text {j + 1}: {similarity:.4f}")
