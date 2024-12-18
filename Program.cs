using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;

public class CaseSummary
{
    public int CaseNumber { get; set; }
    public string Summary { get; set; } = string.Empty; 
    public List<float> Embedding { get; set; } = new List<float>();
}

public class CaseRecord
{
    public int CaseNumber { get; set; }
    public string Summary { get; set; } = string.Empty;
}


public class TextEmbedding
{
    public class InputData
    {
        public string Text { get; set; } = string.Empty;
        public string? CaseNumber { get; set; }
    }

    public class OutputData
    {
        [VectorType(128)]
        public float[] Embedding { get; set; } = Array.Empty<float>();
    }

    private readonly MLContext _mlContext;
    private readonly ITransformer _model;

    public TextEmbedding()
    {
        _mlContext = new MLContext();

        var pipeline = _mlContext.Transforms.Text.FeaturizeText("Embedding", nameof(InputData.Text));

        // Dummy data for initial training (the first two cases)
        var dummyData = new List<InputData>
        {
            new InputData { Text = "James Lewis was charged with murder. He allegedly killed his business partner over a financial dispute. Forensic evidence and witness statements were critical in building the case. Lewis claimed self-defense, but the evidence suggested premeditation. He was found guilty and given a life sentence.", CaseNumber = "Case-001" },
            new InputData { Text = "Emily Clark faced charges of human trafficking. She was implicated in a ring that exploited vulnerable individuals for labor. Testimonies from victims and intercepted communications were pivotal.", CaseNumber = "Case-002" }
        };

        var trainingData = _mlContext.Data.LoadFromEnumerable(dummyData);
        _model = pipeline.Fit(trainingData);
    }

    public List<float> GetTextEmbedding(string inputText)
    {
        var inputData = new InputData { Text = inputText, CaseNumber = null };
        var inputDataView = _mlContext.Data.LoadFromEnumerable(new[] { inputData });

        var transformedData = _model.Transform(inputDataView);
        var outputData = _mlContext.Data.CreateEnumerable<OutputData>(transformedData, reuseRowObject: false);

        return outputData.FirstOrDefault()?.Embedding.ToList() ?? new List<float>();
    }

    public void SaveEmbeddingsToFile(string filePath, List<CaseSummary> caseSummaries)
    {
        List<CaseSummary> existingSummaries = new();

        if (File.Exists(filePath))
        {
            var json = File.ReadAllText(filePath);
            existingSummaries = JsonConvert.DeserializeObject<List<CaseSummary>>(json) ?? new List<CaseSummary>();
        }

        foreach (var newCase in caseSummaries)
        {
            var existingCase = existingSummaries.Find(c => c.CaseNumber == newCase.CaseNumber);
            if (existingCase != null)
            {
                existingCase.Summary = newCase.Summary;
                existingCase.Embedding = newCase.Embedding;
            }
            else
            {
                existingSummaries.Add(newCase);
            }
        }

        var updatedJson = JsonConvert.SerializeObject(existingSummaries, Formatting.Indented);
        File.WriteAllText(filePath, updatedJson);
    }

    public List<(CaseSummary Case, double Similarity)> GetTopSimilarCases(string summary, string filePath, int n)
    {
        var queryEmbedding = GetTextEmbedding(summary);
        if (!queryEmbedding.Any()) throw new InvalidOperationException("Failed to generate embedding.");

        List<CaseSummary> existingSummaries = new();
        if (File.Exists(filePath))
        {
            try
            {
                var json = File.ReadAllText(filePath);
                existingSummaries = JsonConvert.DeserializeObject<List<CaseSummary>>(json) ?? new List<CaseSummary>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading JSON: {ex.Message}");
            }
        }

        return existingSummaries
            .Where(cs => cs.Embedding != null)
            .Select(cs => (cs, CosineSimilarity(queryEmbedding, cs.Embedding)))
            .OrderByDescending(x => x.Item2)
            .Take(n)
            .ToList();
    }

    private double CosineSimilarity(List<float> vectorA, List<float> vectorB)
    {
        if (vectorA.Count != vectorB.Count) throw new ArgumentException("Vectors must be the same length.");

        double dotProduct = 0.0;
        double magnitudeA = 0.0;
        double magnitudeB = 0.0;

        for (int i = 0; i < vectorA.Count; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            magnitudeA += Math.Pow(vectorA[i], 2);
            magnitudeB += Math.Pow(vectorB[i], 2);
        }

        magnitudeA = Math.Sqrt(magnitudeA);
        magnitudeB = Math.Sqrt(magnitudeB);

        return magnitudeA > 0 && magnitudeB > 0 ? dotProduct / (magnitudeA * magnitudeB) : 0.0;
    }
}

class Program
{
    static void Main(string[] args)
    {
        var textEmbedder = new TextEmbedding();
        string filePath = "CaseSummaries.json";
        string csvFilePath = "CaseSummaries.csv"; // Path to the CSV file

        // Load data from the CSV and embed the cases
        EmbedFromCsvAndSave(textEmbedder, csvFilePath, filePath);

        // Test the function with a query
        string querySummary = "The term \"pinnacle\" refers to the highest point or peak of something, both literally and metaphorically. In a physical sense, it often describes the topmost part of a mountain or a structure, such as a building or tower, symbolizing achievement, excellence, or prominence. For example, reaching the pinnacle of a mountain is not only a strenuous physical accomplishment but also signifies arriving at a remarkable height that few may achieve. Metaphorically, \"pinnacle\" is frequently used to denote the culmination of success or the apex of a person's career or life journey, such as an artist reaching the pinnacle of their creative work or a leader attaining the pinnacle of their influence and power. In literature and discussions about personal growth or ambition, describing a moment or position as the pinnacle highlights its significance and the hard work necessary to reach such heights. The use of \"pinnacle\" evokes feelings of inspiration and aspiration, suggesting that while the journey to greatness may be challenging, the rewards of perseverance can lead to remarkable and fulfilling outcomes. Overall, the concept of the pinnacle represents both a literal and figurative aspiration for greatness, making it a powerful symbol in various contexts.";
        int numSimilar = 3;

        try
        {
            var topCasesWithSimilarity = textEmbedder.GetTopSimilarCases(querySummary, filePath, numSimilar);

            Console.WriteLine($"Top {numSimilar} Similar Cases");
            foreach (var pair in topCasesWithSimilarity)
            {
                var caseSummary = pair.Case;
                var similarity = pair.Similarity;

                Console.WriteLine($"Case {caseSummary.CaseNumber}: {caseSummary.Summary}");
                Console.WriteLine($"Similarity Score: {similarity:F4}");
                Console.WriteLine();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    public static void EmbedFromCsvAndSave(TextEmbedding textEmbedder, string csvFilePath, string filePath)
    {
        var caseSummaries = new List<CaseSummary>();

        // Open the CSV file and read it
        using (var reader = new StreamReader(csvFilePath))
        using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            // Ensure quotes are properly handled around the Summary field
            Quote = '"', 
            BadDataFound = null // Ignore bad data instead of throwing exceptions
        }))
        {
            // Read records from the CSV
            var records = csv.GetRecords<CaseRecord>().ToList();

            foreach (var record in records)
            {
                var embedding = textEmbedder.GetTextEmbedding(record.Summary);
                caseSummaries.Add(new CaseSummary
                {
                    CaseNumber = record.CaseNumber,
                    Summary = record.Summary,
                    Embedding = embedding
                });
            }
        }

        // Now save the embeddings to file
        textEmbedder.SaveEmbeddingsToFile(filePath, caseSummaries);
    }

}
