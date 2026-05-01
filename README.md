# CyberShield URL Scanner

A college-level cybersecurity project that classifies URLs as **Safe**,
**Suspicious**, or **Phishing** using:

1. **Rule-based heuristics** — flags URLs containing `@` or with more than 3 dots.
2. **Machine Learning** — `MultinomialNB` (Naive Bayes) trained on a built-in
   dataset of safe and phishing URLs, vectorized with `CountVectorizer`.

## Project Structure

```
phishing_streamlit/
├── app.py            # Streamlit UI + integration logic
├── model.py          # CountVectorizer + Multinomial Naive Bayes model
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. (Recommended) virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Streamlit will open the app in your browser (default: http://localhost:8501).

## How it works

- The form sends the URL to `analyze()` in `app.py`.
- Rule-based layer first:
  - URL contains `@` or has more than 3 dots → **Suspicious** (yellow).
- Otherwise, the trained model in `model.py` predicts:
  - **Safe** (green) or **Phishing** (red), with confidence score.
- Empty / invalid input is handled with inline error messages.
