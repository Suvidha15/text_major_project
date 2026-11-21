
# InnerVoice – AI-Based Mental Health Detection and Text Classification

## About the Project

InnerVoice is an AI-based web application that analyzes user-provided text to identify patterns associated with different mental health states. It uses Machine Learning and Natural Language Processing (NLP) techniques to classify text into four categories: Non-Stress, Stress, Depression, and Suicidal.

The application also provides confidence scores, visual summaries, and Explainable AI (XAI) insights to help users understand which words influenced the prediction.

## Features

- **Text Classification:** Classifies text into four mental health categories.
- **Sentence-Level Analysis:** Splits paragraphs into individual sentences and analyzes them separately.
- **Confidence Score:** Displays the prediction confidence for each sentence.
- **Explainable AI (XAI):** Uses LIME to identify words contributing to the model's predictions.
- **Visual Analytics:** Displays category distribution using a pie chart.
- **Mental Health Insights:** Provides conclusions and supportive suggestions based on the detected category.
- **Voice Output:** Provides voice-based conclusions.
- **Crisis Support:** Displays additional support information when concerning patterns are detected.

## Technologies Used

- Python
- Flask
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- LIME
- NumPy
- HTML, CSS and JavaScript

## Classification Categories

| Category | Description |
|---|---|
| Non-Stress | Text reflecting a calm or positive state |
| Stress | Text indicating stress-related patterns |
| Depression | Text associated with emotional low points |
| Suicidal | Text containing potentially concerning patterns |

## Project Structure

```text
text_major_project/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

*The structure above is illustrative; adjust it to match your actual folders and files.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Suvidha15/text_major_project.git
cd text_major_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file, install the main dependencies:

```bash
pip install flask numpy scikit-learn lime
```

### 5. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## Explainable AI

InnerVoice integrates LIME (Local Interpretable Model-Agnostic Explanations) to provide word-level explanations for individual predictions. It highlights influential words and indicates whether their contribution supports or opposes the predicted category.

## Disclaimer

InnerVoice is an academic project developed for educational and research purposes. Its predictions are based on text patterns and should not be considered a medical diagnosis or a substitute for professional mental health assessment.

If someone is experiencing suicidal thoughts or immediate danger, they should seek help from a qualified mental health professional or local emergency services.

## Repository

GitHub: https://github.com/Suvidha15/text_major_project
