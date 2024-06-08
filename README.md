# Text Summarization using LLMs on Medical Reports

#### Abstract
This project focuses on summarizing medical reports using natural language processing (NLP) techniques, specifically employing large language models (LLMs). The aim is to facilitate healthcare professionals in quickly extracting relevant information from extensive medical data.Medical reports are often lengthy and dense with technical information, making them difficult to navigate. The project tests both extractive and abstractive summarization models to determine which is most effective for summarizing various types of medical documents.

#### Introduction
The rapid growth of medical documents necessitates efficient methods for processing and summarizing these texts. Text summarization (ATS) can significantly aid clinicians by providing concise summaries of verbose clinical documents, thus enhancing their decision-making process.

#### Related Work
* Transformer Models: Training transformers from scratch for summarizing long documents without pre-training.
* Biomedical Data Summarization: Different metrics and application areas have been explored, highlighting the challenges and the use of transformer-based methodologies.
* Universal Summarization Frameworks: The need for frameworks capable of handling diverse clinical notes and integration into decision-support systems.

#### Data Collection and Pre-Processing:
* Dataset: Scraped from [mtsamples.com](https://www.mtsamples.com), consisting of over 5000 sample medical reports.
* Cleaning: Removal of extraneous data and ensuring data quality.

#### Models Used:
* Frequency Based Model Scoring: Uses word frequency to score sentences.
* Text Rank: A graph-based ranking algorithm similar to PageRank.
* Bert Extractive Summarizer: Uses BERT for sentence embeddings and K-means clustering.
* Bart Summarizer: A fine-tuned version of the BART model.
* BigBirdPegasus: A transformer model capable of handling long sequences.

#### Results:
* Models Compared: Frequency Based Model Scoring, Text Rank, Bert Extractive Summarizer, Bart Summarizer, and BigBirdPegasus.
* Performance Metrics: ROUGE scores for precision and recall.
* Findings: The fine-tuned BART model performed the best, providing the most comprehensive summaries.
