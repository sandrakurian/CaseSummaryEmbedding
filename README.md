# Text Embedding and Similarity Search

This project is part of a larger initiative to store legal case summaries and retrieve similar cases based on their content. The goal of this project is to leverage text embedding techniques to convert case summaries into numerical representations and use similarity search to find the most relevant cases.

The system allows you to:

- **Store Case Summaries**: Save case summaries along with their corresponding embeddings.
- **Retrieve Similar Cases**: Perform similarity searches to find the most similar case summaries based on a query.

## Features

- **Text Embedding**: Converts case summaries into numerical embeddings using ML.NET.
- **Similarity Search**: Finds the most similar case summaries using cosine similarity.
- **Case Data Storage**: Stores and manages case summaries and their embeddings in a JSON file.

## Requirements

- .NET 6 or higher
- Microsoft.ML
- Newtonsoft.Json
- CsvHelper

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sandrakurian/CaseSummaryEmbedding.git
   ```

2. Navigate to the project folder:

   ```bash
   cd CaseSummaryEmbedding
   ```

3. Install the required dependencies:

   ```bash
   dotnet restore
   ```

## Usage

### Embedding Text

To generate embeddings for a case summary, create an instance of the `TextEmbedding` class and call the `GetTextEmbedding` method with the case summary text.

Example:

```csharp
var textEmbedder = new TextEmbedding();
var embedding = textEmbedder.GetTextEmbedding("Case summary text here.");
```

### Save Case Summaries with Embeddings

To store case summaries with embeddings in a JSON file, use the `SaveEmbeddingsToFile` method.

Example:

```csharp
textEmbedder.SaveEmbeddingsToFile("CaseSummaries.json", caseSummaries);
```

### Get Similar Cases

To find the most similar case summaries to a query, use the `GetTopSimilarCases` method. This method returns the top `n` most similar cases based on cosine similarity.

Example:

```csharp
var topCases = textEmbedder.GetTopSimilarCases(querySummary, "CaseSummaries.json", 3);
foreach (var caseSimilarity in topCases)
{
    Console.WriteLine($"Case {caseSimilarity.Case.CaseNumber}: {caseSimilarity.Case.Summary}");
    Console.WriteLine($"Similarity: {caseSimilarity.Similarity}");
}
```

### CSV Integration

To load case summaries from a CSV file and generate embeddings, use the `EmbedFromCsvAndSave` method.

Example:

```csharp
EmbedFromCsvAndSave(textEmbedder, "CaseSummaries.csv", "CaseSummaries.json");
```

## Structure

- **CaseSummary**: Represents a case summary with a case number, the summary text, and its embedding.
- **TextEmbedding**: Handles the process of generating embeddings using ML.NET and cosine similarity for comparing case summaries.
- **Program**: The entry point of the application, where data is processed and the similarity search is performed.
