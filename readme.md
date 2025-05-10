# Emotion and Sentiment Analysis

This project performs **emotion classification** using a machine learning model and **sentiment analysis** using VADER (Valence Aware Dictionary and sEntiment Reasoner). It demonstrates text preprocessing, data visualization, model training, evaluation, and comparison of emotion classification with lexicon-based sentiment prediction.

## 📊 Features

- Emotion distribution visualization using Seaborn
- Word cloud generation for each emotion
- Text preprocessing with NLTK
- TF-IDF vectorization
- Logistic Regression classifier for emotion prediction
- Evaluation using confusion matrix and classification report
- Sentiment analysis with VADER

## 📁 Project Structure

```
.
├── emotion_sentiment_analysis.py
├── README.md
├── requirements.txt
└── sample_outputs/
    ├── emotion_distribution.png
    ├── confusion_matrix.png
    └── vader_sample_output.png
```

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-sentiment-analysis.git
cd emotion-sentiment-analysis
```

### 2. Install dependencies

Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python emotion_sentiment_analysis.py
```

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
vaderSentiment
```

## 📈 Sample Output

- Emotion distribution bar chart
- Word clouds for each emotion
- Confusion matrix for model evaluation
- VADER sentiment classifications

## 📚 Data

The project uses a small, hardcoded dataset of 10 labeled text samples. You can replace it with your own dataset in CSV or JSON format.

## ✅ To-Do

- [ ] Add support for external datasets
- [ ] Improve model performance with more data
- [ ] Add Streamlit interface
- [ ] Evaluate other models (e.g., SVM, Naive Bayes)

## 📝 License

This project is licensed under the MIT License.

---

**Feel free to fork or contribute!*
