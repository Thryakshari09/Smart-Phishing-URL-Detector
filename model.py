"""
model.py
--------
ML model for the Smart Phishing URL Detector.

- CountVectorizer turns each URL into a bag-of-tokens feature vector.
- MultinomialNB (Naive Bayes) is trained on a small built-in dataset
  containing both safe and phishing URLs.
- `predict_url(url)` returns ("Safe"|"Phishing", confidence in 0..1).
"""

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# label: 0 = Safe, 1 = Phishing
SAMPLE_URLS = [
    # Safe / legitimate
    ("https://www.google.com", 0),
    ("https://github.com/login", 0),
    ("https://www.wikipedia.org/wiki/Phishing", 0),
    ("https://stackoverflow.com/questions/tagged/python", 0),
    ("https://www.amazon.com/gp/cart/view.html", 0),
    ("https://mail.google.com/mail/u/0/", 0),
    ("https://www.microsoft.com/en-us/microsoft-365", 0),
    ("https://www.apple.com/shop/buy-iphone", 0),
    ("https://www.netflix.com/browse", 0),
    ("https://www.linkedin.com/feed/", 0),
    ("https://www.paypal.com/signin", 0),
    ("https://twitter.com/home", 0),
    ("https://www.bbc.com/news", 0),
    ("https://openai.com/research", 0),
    ("https://www.dropbox.com/home", 0),
    # Phishing-looking
    ("http://paypal-login-secure-update.com/verify", 1),
    ("http://192.168.1.10/login.php?account=verify", 1),
    ("http://apple-id-locked-confirm.support-login.ru", 1),
    ("http://secure-update-amaz0n.com/account/verify", 1),
    ("http://login-microsoft365-confirm.tk/auth", 1),
    ("http://bank-of-america.verify-account.online/login", 1),
    ("http://free-gift-card-walmart.win/claim", 1),
    ("http://netflix-billing-update.account-secure.cf", 1),
    ("http://dropbox-shared-document-view.gq/login", 1),
    ("http://chase-online-banking-alert.security-check.ml", 1),
    ("http://verify-paypal.com.account.login.confirm.tk", 1),
    ("http://facebook-security-help-center.support-team.ga", 1),
    ("http://google-docs-shared-file.review-doc.cf/open", 1),
    ("http://signin.ebay.com.update-info.win/login", 1),
    ("http://instagram-verify-badge.account-help.tk", 1),
]


def _tokenize(url: str):
    """Split a URL into word-like tokens."""
    return re.findall(r"[A-Za-z0-9]+", url.lower())


# ---- Train once on import ----
_urls   = [u for u, _ in SAMPLE_URLS]
_labels = [y for _, y in SAMPLE_URLS]

vectorizer = CountVectorizer(tokenizer=_tokenize, token_pattern=None)
X = vectorizer.fit_transform(_urls)

classifier = MultinomialNB()
classifier.fit(X, _labels)


def predict_url(url: str):
    """Return (label, confidence) where label is 'Safe' or 'Phishing'."""
    features = vectorizer.transform([url])
    pred  = int(classifier.predict(features)[0])
    proba = classifier.predict_proba(features)[0]
    label = "Phishing" if pred == 1 else "Safe"
    return label, float(proba[pred])
