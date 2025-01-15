# Case Prediction Writeup  

please look in the [Work folder](/Work/) to find relavent informaton

## Table of Contents
1. [Objective](#objective)
2. [Overview of Process](#overview-of-process)
3. [Embedding Process Overview](#embedding-process-overview)
   - [Preprocessing Case Summaries](#1-preprocessing-case-summaries)
   - [Generating Embeddings](#2-generating-embeddings)
   - [Storing Embeddings](#3-storing-embeddings)
   - [Similarity Analysis](#4-similarity-analysis)
   - [Handling User Requests](#5-handling-user-requests)
4. [Embedding Case Summaries -vs- Embedding the Full Case](#embedding-case-summaries--vs--embedding-the-full-case)
5. [Prediction Process Overview](#prediction-process-overview)
   - [Preprocessing the Input Data](#1-preprocessing-the-input-data)
   - [Prompt Construction](#2-prompt-construction)
   - [API](#3-api)
6. [Reporting](#reporting)
   - [Case Processing Status](#1-case-processing-status)
   - [Similar Case Search](#2-similar-case-search)
   - [Analysis and Prediction](#4-analysis-and-prediction)

---

## Objective  
Currently, case predictions rely solely on the specifics of the ongoing case, often limiting the scope and accuracy of these forecasts. With the proposed implementation, predictions will become significantly more informed and reliable. By referencing and analyzing patterns from a database of similar past cases, this system will enable more educated and data-driven predictions. This shift from isolated assessments to pattern-based forecasting will provide decision-makers with deeper insights.  

---

## Overview of Process  
The prediction process will leverage both the details of the main case and patterns derived from similar cases to provide more accurate and informed predictions. Here’s how the process will work:  

1. **Embedding Case Summaries**:  
   - The system will transform case summaries into numerical representations (embeddings). These embeddings capture the essential characteristics and details of each case in a format suitable for computational analysis.  

2. **Finding Similar Cases**:  
   - The database of case embeddings will be queried to identify the top three cases most similar to the main case.  
   - Similarity will be determined using algorithms such as cosine similarity, which measures the closeness of the embeddings.  

3. **Prediction Generation**:  
   - After identifying the top three similar cases, the system will analyze their outcomes and extract patterns.  
   - These patterns will serve as a foundation for predicting the likely outcome of the main case.  

---

## Embedding Process Overview
The overall goal of the embedding process is to convert textual case summaries into machine-readable numerical representations. This transformation enables us to analyze the semantic similarity between different cases, making it easier to compare and manage them effectively. The process relies heavily on natural language processing (NLP) techniques, particularly the use of pretrained transformer models for generating embeddings.

### 1. **Preprocessing Case Summaries**
The preprocessing phase involves extracting and preparing the content from the case summary for embedding generation:

- **Section Identification**: The `embed_sections()` function takes the full case summary as input. The text is divided into distinct sections based on the delimiter `###` (which is used to indicate headers). This allows the case summary to be structured into manageable parts, each corresponding to a specific section.
- **Title and Content Extraction**: Each section is then split into two parts: the **title** (the first line of the section) and the **content** (the remaining text or list items). The title is used as the key for the embeddings dictionary, while the content is passed to the transformer model for embedding generation.

### 2. **Generating Embeddings**
After preprocessing, the next step is to generate numerical representations (embeddings) for each section's content:

- **Model Selection**: The code uses a pretrained transformer model, specifically the `all-mpnet-base-v2` model from SentenceTransformers. This model is well-suited for generating dense vector representations that capture the semantic meaning of text.
- **Content Handling**: There are two primary types of content to handle:
  - **Paragraphs**: If the section contains regular paragraphs (i.e., not structured as a list), the entire content is passed through the model to generate a single embedding.
  - **Lists**: If the content is structured as a list (e.g., starts with `-` or `1.`), each list item is individually encoded, and the mean of these embeddings is used to represent the entire section. This approach ensures that the semantic meaning of each list item is captured and averaged for a cohesive representation of the entire list.

This process is carried out in the `embed_sections()` function, which returns a dictionary where each key is a section title, and its value is the corresponding embedding.

### 3. **Storing Embeddings**
Once the embeddings are generated, the next task is to store them in a structured format for easy retrieval:

- **Embeddings Storage**: The embeddings are saved in a JSON file (`embeddings.json`). The `save_embeddings()` function loads any existing data from the file and updates it with the newly generated embeddings. Each case is identified by its `case_id`, and the embeddings for each section are stored as a list of floating-point numbers.
- **File Handling**: If the case's embeddings are new, they are added to the JSON file. If the case already exists, its embeddings are updated. This is done in a way that maintains a persistent record of embeddings across multiple cases, ensuring that future similarity comparisons can be efficiently handled.

### 4. **Similarity Analysis**
Once embeddings are stored, the system can analyze similarities between cases:

- **Section Similarity**: The `find_section_similarities()` function compares the embeddings of corresponding sections from two cases. This comparison is done using **cosine similarity**, a metric that measures the angle between two vectors. Higher cosine similarity values indicate that the sections are more semantically similar.
- **Overall Similarity**: In the `find_most_similar_cases()` function, the system calculates the similarity between the entire case (by averaging the embeddings of its sections). It then identifies the top N most similar cases based on these overall similarities. This allows for efficient case retrieval and comparison.

### 5. **Handling User Requests**
The system is designed to respond to two primary user actions:

- **Summary**: When the user selects the "summary" button, the system generates embeddings for the case summary and stores them in the JSON file. This allows the system to later compare the case with others.
- **Similarity Search ("Sandra")**: When the user selects the "sandra" button, the system loads existing embeddings, calculates similarities between the current case and all other cases, and identifies the most similar cases. It then compares sections of the current case with those of the similar cases and reports the similarities for further analysis.

---

## Embedding Case Summaries -vs- Embedding the Full Case  

Embedding case summaries strikes a balance between efficiency and accuracy. It reduces computational demands, ensures the focus remains on the critical aspects of the case, and provides meaningful and actionable insights in a scalable manner.  

1. **Focused Information Extraction/Noise Reduction**:  
   - Summaries distill the most relevant information about a case, such as the context, primary events, and outcomes. Embedding these summaries avoids processing extraneous details and focuses on the core content critical for prediction and similarity analysis.  
   - Embedding the entire case might include irrelevant or redundant details, such as procedural text or legal formalities, which can dilute the quality of the embeddings.  

2. **Computational Efficiency**:  
   - Embedding summaries requires less time, enabling faster predictions and quicker updates to the database when new cases are added.  
   - The final embedding of a case summary takes less storage than a full case embedding.  

3. **Improved Similarity Analysis**:  
   - Summaries standardize the information being compared, ensuring that similarity analysis focuses on the most important aspects of cases rather than being influenced by variations in case length or structure.  

---

## Prediction Process Overview

The prediction process revolves around using a combination of case summaries, NLP models, and an external API to generate predictions about the future outcomes of a given case, based on its similarity to other related cases. This process utilizes machine learning and reasoning techniques to analyze, predict, and justify potential outcomes.

### 1. **Preprocessing the Input Data**
The prediction process starts with the **preprocessing of input data**, which includes the main case and its similar cases:

- **Main Case**: The main case (referred to as `main_case`) is the case for which predictions are to be made. Its detailed summary is extracted using the `process_file_content(main_case, short=False)` function.
- **Similar Cases**: A list of similar cases (`similar_cases`) is also provided. Each of these similar cases is processed in a similar manner to the main case. The summaries of these similar cases are combined into a string (`similar_cases_str`) for use in the prompt.

### 2. **Prompt Construction**
Once the case summaries are prepared, the next step is to create the **prompt** that will be used to communicate with the external API:

**Prompt Structure**: The prompt is carefully structured to guide the model in generating predictions. It consists of three primary sections:
1. **Analysis**: This section requests a comprehensive analysis of the main case and its possible future outcomes, backed by logical reasoning.
2. **Predictions**: This section requests specific future predictions in bullet-point format, detailing possible outcomes based on the case's information.
3. **Referrals**: In this section, the model is asked to explain how the similar cases influenced the analysis and predictions. The model should compare the similarities and differences between the main case and the similar cases to justify the predictions made.

  The prompt combines the **main case summary** and the **similar case summaries** into the appropriate format, ensuring the model has all the necessary context to generate an informed response.

### 3. **API**
The core of the prediction process is the interaction with an external API to generate the required analysis and predictions:

- **API Setup**: The code uses Google's `generative language` API (specifically the Gemini model) to generate the predictions. The API key is included in the URL, and the model is invoked using a POST request.

- **Handling the Response**: If the response status is successful (`status_code == 200`), the content generated by the API is extracted from the `response_json`. The relevant content (predictions and analysis) is stored in the `text` field.

- **Error Handling**: If the API request fails, an error message is indicates the status code and error details for troubleshooting.

---

## Reporting  

The `script.log` file contains a detailed log of the processing steps, outputs, and analysis results related to various case files being processed. Based on the provided excerpt, here's a breakdown of the contents:

### 1. **Case Processing Status**

- **Processing Specific Cases**: The log tracks the processing of specific cases, including which button or section of the case is being worked on. For example:

- **Embeddings Saved**: The log also indicates that embeddings (vector representations) for various case files have been successfully saved.

### 2. **Similar Case Search**
The log shows the most similar cases identified for a given case.

**Case Breakdown by Sections**: The log includes a detailed breakdown of the sections of the similar cases and their respective similarity scores to the main case. This helps identify which aspects of the similar cases match most closely with the case being analyzed.


### 4. **Analysis and Prediction**
- **Case Analysis**: 
  - The log contains a detailed analysis of the main case, including insights into the challenges faced by the family, such as:
    - Domestic violence, mental health struggles, financial instability, and legal complexities.
  - The analysis provides an overview of how these issues impact the family's well-being and how interventions might be necessary to address them.

- **Predictions**: The log outlines possible future outcomes for the case based on the analysis. 

- **Referrals**
  - The log mentions that comparing intervention effectiveness and long-term outcomes in these similar cases would greatly enhance the accuracy of the predictions.
  - The log highlights that the similarity scores between the cases suggest a high degree of similarity, which would help improve the prediction if more details were available from the similar cases.