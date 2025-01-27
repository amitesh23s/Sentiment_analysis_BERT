# Twitter Sentiment Analysis

This project focuses on determining the sentiment of tweets by classifying them into one of four predefined categories: positive, negative, neutral, and irrelevant. By leveraging the BERT transformer model (bert-base-uncased) and fine-tuning it for sentiment classification, this project showcases the effectiveness of transformer-based models in understanding and analyzing textual data from social media.


## Problem Statement
Social media platforms like Twitter are rich sources of unstructured text data, making them ideal for sentiment analysis. The goal of this project is to classify tweets into one of the following sentiment categories:

 - Positive: Tweets expressing a positive or favorable sentiment.
 - Negative: Tweets with a negative tone or sentiment.
 - Neutral: Tweets that are neither strongly positive nor strongly negative.
 - Irrelevant: Tweets unrelated to the context or objective of sentiment analysis.
This classification can help brands and organizations monitor public sentiment, gain actionable insights, and make informed decisions based on user feedback.



## Project Workflow
- ### Load Dataset
    The first step involves loading the dataset containing tweets and their corresponding sentiment labels. The dataset is structured as a CSV file and is imported into a pandas DataFrame for easy manipulation and exploration. This allows us to analyze the data and perform necessary preprocessing steps.

- ### Data Preprocessing
  - In this step, we Remove missing values (None) from the dataset to ensure data quality. Clean the text by handling special characters, hashtags, and mentions to make the data suitable for tokenization. Combine preprocessing steps to standardize the dataset for training. This ensures that the input data is consistent, clean, and ready for further processing.

- ### Assign Classification Labels
  - Each tweet is assigned one of four predefined classification labels: 0 for Positive, 1 for Neutral, 2 for Negative, 3 for Irrelevant. These numeric labels allow us to convert the sentiment categories into a format that the model can interpret.

- ### Split Data
  - The dataset is split into training and testing subsets using scikit-learn's model_selection tools. The split ensures that the training and testing data are stratified across the sentiment labels to maintain balance in the data. This step is critical to evaluate the model's generalization capability.

- ### Tokenization
  - Tokenization is a crucial step where the text data is converted into numerical format. Using the BERT tokenizer from Hugging Face’s transformers library, each tweet is tokenized into tokens (subwords) and padded or truncated to a uniform length. This step creates arrays of input IDs and attention masks that are compatible with transformer models.

- ### Model Fine-Tuning
    The bert-base-uncased model, a pre-trained transformer-based language model, is fine-tuned for this multi-class classification task. The fine-tuning process involves:

    - Feeding tokenized tweet data into the model.
    - Training the model to learn the relationships between input text and corresponding sentiment labels.
    - Utilizing Hugging Face’s Trainer API to streamline the training process.

- ### Evaluation
    The trained model is evaluated on the test set using the Pearson correlation coefficient (r) to measure the alignment between predicted and actual sentiment labels. The Pearson coefficient provides insights into the strength of the linear relationship between the predicted and actual values: +1: Perfect positive correlation. 0: No correlation. -1: Perfect negative correlation.
    The evaluation showed that the Pearson coefficient improved after each epoch, highlighting the model's increasing ability to learn meaningful patterns in the data.

- ### Validation
    Finally, the model is tested on a validation set to assess its real-world performance. The validation accuracy reached 80%, demonstrating the model's effectiveness in accurately classifying tweets into sentiment categories.

![Data preprocessing (3)](https://github.com/user-attachments/assets/8741d2ea-2729-4bc7-8b9f-a9c4adfb001e)


## Skills used
This project required expertise in the following tools and techniques:

 - pandas: For data loading, cleaning, and preprocessing, including handling missing values and assigning labels.
 - transformers: For leveraging Hugging Face's tools, such as:
    - BertTokenizer: Tokenizing tweets into input IDs and attention masks.
    - BertForSequenceClassification: Fine-tuning the pre-trained BERT model for sentiment classification.
    - Trainer and TrainingArguments: Simplifying the training and evaluation pipeline.
- numpy: For numerical computations, such as handling arrays during evaluation.
- scikit-learn: For splitting the dataset into training and testing subsets using stratified sampling, ensuring balanced sentiment labels in each set.


## Results
- Validation Accuracy: 80%
- Pearson Correlation Coefficient: The coefficient improved after each epoch, demonstrating the model's ability to better align predictions with true sentiment labels.
    
The results indicate that the fine-tuned BERT model is highly effective for sentiment analysis of tweets, making it a reliable tool for real-world applications.
