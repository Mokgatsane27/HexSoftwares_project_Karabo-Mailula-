 HexSoftwares_SentimentAnalysis

## Project Title
**Sentiment Analysis of Customer Reviews**

## Description
This machine learning project classifies customer reviews as **Positive** or **Negative** using natural language processing and the Naive Bayes classifier. It helps businesses understand customer feedback at scale.

## Technologies Used
- Python
- pandas, numpy, scikit-learn
- NLTK (for dataset and text preprocessing)
- matplotlib, seaborn

---

## 1. Data Loading

```python
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

reviews = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# 2.Text Preprocessing
'''Python 
import pandas as pd
df = pd.DataFrame({'review': reviews, 'sentiment': labels})

# 3. Feature Extraction
'''Python 
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# 4. Train-Test Split
''' Python 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 5. Train the Naive Bayes Model
'''Python 
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Model Evaluation
'''Python 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

