# app.py

# === INSTALL REQUIRED LIBRARIES (Uncomment if needed) ===
# !pip install streamlit nltk seaborn matplotlib scikit-learn wordcloud vaderSentiment

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === DOWNLOAD NLTK RESOURCES ===
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Decoding Emotions Through Sentiment Analysis")
st.markdown("This app uses Machine Learning and VADER to analyze emotions in social media texts.")

# === SAMPLE DATA ===
data = {
    "text": [
        "I am so happy today!",
        "This is the worst day ever.",
        "I'm scared of what will happen next.",
        "I love spending time with my friends.",
        "Everything is falling apart.",
        "I'm feeling excited and joyful!",
        "Why is this happening to me?",
        "That movie was hilarious and fun!",
        "I can't stop crying.",
        "He really made my day!"
    ],
    "emotion": [
        "joy", "anger", "fear", "joy", "sadness",
        "joy", "fear", "joy", "sadness", "joy"
    ]
}
df = pd.DataFrame(data)

# === CLEANING FUNCTION ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# === EMOTION DISTRIBUTION ===
st.subheader("Emotion Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='pastel', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# === WORD CLOUDS ===
st.subheader("Word Clouds by Emotion")
emotions = df['emotion'].unique()
for emotion in emotions:
    text = " ".join(df[df['emotion'] == emotion]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

# === MODEL TRAINING ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === CLASSIFICATION REPORT ===
st.subheader("Model Performance")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="YlGnBu", ax=ax_cm)
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(fig_cm)

# === VADER ANALYSIS ===
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

def vader_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['vader_score'].apply(vader_sentiment)

# === VADER RESULTS ===
st.subheader("VADER Sentiment Results")
st.dataframe(df[['text', 'vader_sentiment']])

# === USER INPUT FOR LIVE PREDICTION ===
st.subheader("Try Your Own Text")
user_input = st.text_area("Enter a sentence to analyze:", "")

if user_input:
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    model_pred = clf.predict(vectorized)[0]
    vader_score = analyzer.polarity_scores(cleaned)['compound']
    vader_label = vader_sentiment(vader_score)

    st.markdown(f"**ML-Predicted Emotion:** {model_pred}")
    st.markdown(f"**VADER Sentiment:** {vader_label} (score: {vader_score:.2f})")
