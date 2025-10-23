# Conversational AI Interface and Intent Classification System

A functional Natural Language Processing (NLP) chatbot designed to accurately understand and respond to user queries by implementing a machine learning-based **Intent Classification** model and deploying it through a user-friendly web interface.

---

## Project Title & Short Description

**Title:** Conversational AI Interface and Intent Classification System

**Description:** This project implements a classification model (**Logistic Regression**) utilizing **TF-IDF features** to map user text input to a predefined intent (e.g., 'greeting', 'thanks'). The entire application is packaged and deployed as an interactive web chat interface using the **Streamlit** framework.

---

## Problem Statement / Goal

The primary goal is to build an effective and responsive **Rule-Based Chatbot** system that automates the initial phase of human-computer interaction. This involves:
1.  **Intent Recognition**: Accurately classifying natural language input into known categories using a machine learning model.
2.  **Contextual Response**: Providing a relevant and randomized response to maintain a natural conversational flow.
3.  **Deployment**: Packaging the system into a user-friendly, accessible web application using Streamlit.

---

## Tech Stack / Tools Used

The solution is built entirely in Python, integrating popular libraries for NLP, machine learning, and web development:

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Interface** | Streamlit | Deploying the interactive, front-end web application |
| **Modeling** | Scikit-learn | Core library for the classification model (Logistic Regression) |
| **Feature Extraction**| TfidfVectorizer | Converting raw text patterns into numerical features |
| **NLP** | NLTK | Tokenization and text processing utilities (`punkt` download) |

---

## Approach / Methodology

1.  **Intent Definition**: A comprehensive set of intents (`greeting`, `goodbye`, `about`, etc.), associated `patterns`, and randomized `responses` are hardcoded into the system.
2.  **Text Vectorization**: The training `patterns` are transformed into numerical vectors using the **TF-IDF Vectorizer**, emphasizing words that are unique and descriptive of a particular intent.
3.  **Model Training**: A **Logistic Regression** classifier is trained on the TF-IDF feature matrix to learn the optimal weights for mapping text to the correct intent tag.
4.  **Inference**: When a user inputs text, the text is first vectorized using the fitted TF-IDF model, then the classifier predicts the intent tag.
5.  **Response Generation**: Based on the predicted tag, the system selects and returns a random response from the pre-defined list for that intent, ensuring variety in conversation.

---

## Results / Key Findings

* The system successfully demonstrates a robust, production-ready pipeline for **Intent Recognition** using classical machine learning, providing high accuracy with minimal computational overhead.
* The use of **Streamlit** enables rapid deployment of the ML model, transforming a core script into a fully interactive web application immediately.
* The system is scalable; adding new functionality requires only defining a new intent and retraining the Logistic Regression model.

---

## Topic Tags

Chatbot ConversationalAI NLP IntentClassification LogisticRegression Streamlit Tfidf Python Scikit-learn

---

## How to Run the Project

### 1. Install Requirements and NLTK Data

Install all necessary packages using the provided `requirements.txt` file and download the required NLTK data:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
