# Sentiment Analysis using Naive Bayes

An educational project implementing a Naive Bayes classifier from scratch for movie review sentiment analysis.

## 📋 Overview

This project demonstrates how to build a text classification system using the Naive Bayes algorithm. The classifier learns to predict whether movie reviews are positive or negative by analyzing word frequencies in the training data.

**Key Concepts:**
- Text preprocessing and tokenization
- Naive Bayes probability calculations
- Laplace smoothing for handling unseen words
- Sentiment classification

## 📊 Dataset

The dataset is from [IMDb dataset sentiment analysis in CSV format on Kaggle](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format).

**Dataset Statistics:**
- **Training set:** 40,000 movie reviews (50% positive, 50% negative)
- **Test set:** 5,000 movie reviews (50% positive, 50% negative)
- **Labels:** 0 = negative, 1 = positive

### Dataset Exploration

The notebook includes comprehensive dataset exploration with actual results:

- **Class Distribution:** ✓ Perfectly balanced (50% negative, 50% positive in both sets)
- **Missing Values:** ✓ No missing values detected in either dataset
- **Text Length Statistics:**
  - **Training set:** Mean: 231.34 words | Median: 173 words | Range: 4-2,470 words
  - **Test set:** Mean: 231.92 words | Median: 173 words | Range: 10-2,108 words
- **Sample Reviews:** Real examples from both classes displayed to understand the data
- **Data Quality:** ✓ Dataset is clean and ready for processing

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. **Dataset Download:** The dataset is automatically downloaded from Kaggle using `kagglehub` when you run the notebook. No manual download required! The dataset will be cached locally for reuse.

   The notebook will download the dataset to: `~/.cache/kagglehub/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format/`

4. NLTK stopwords are also downloaded automatically when you run the notebook

## 🚀 Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook Naive_bayes.ipynb
```

The notebook is organized into the following sections:
1. **Imports** - Loading required libraries
2. **Dataset Download** - Automatic download from Kaggle using kagglehub
3. **Data Check** - Exploring the dataset structure and statistics
4. **Dataset Exploration** - Analyzing class balance, text length distribution, and sample reviews
5. **Data Split** - Separating features and labels
6. **Process Text** - Text cleaning and preprocessing
7. **Count Words** - Building word frequency dictionaries
8. **Train NB** - Training the Naive Bayes classifier
9. **Test NB** - Evaluating model performance
10. **Results** - Model accuracy and custom predictions

## 🔍 Text Preprocessing Steps

The preprocessing pipeline includes:

1. **Remove URLs** - Strip hyperlinks from text
2. **Remove special characters** - Clean hashtags and @ mentions
3. **Tokenization** - Split text into individual words
4. **Remove punctuation** - Strip punctuation marks
5. **Filter non-alphabetic tokens** - Keep only words
6. **Remove stop words** - Filter common words (the, is, at, etc.)
7. **Stemming** - Reduce words to their root form (e.g., "running" → "run")

## 🎯 Model Training Details

**Vocabulary Size:** 89,511 unique words

**Training Process:**
1. Build frequency dictionary mapping (word, sentiment) → count
2. Calculate log prior: log(P(positive) / P(negative)) = -0.0019
3. Calculate log likelihood for each word using Laplace smoothing
4. Training time: ~2-10 minutes depending on hardware

The near-zero log prior (-0.0019) confirms the dataset is perfectly balanced between positive and negative reviews.

## 📈 Model Performance

The Naive Bayes classifier achieves:

**Accuracy: 86.06%** on the test dataset

### Example Predictions

```python
# Positive review
"I would say that this was the most awe... wait for it.. some move of all the times!"
→ Prediction: Positive (score: 0.4373)

# Negative review
"Hate this film. Waste of time:((("
→ Prediction: Negative (score: -2.5588)

# Simple phrase
"good movie"
→ Prediction: Negative (score: -0.2928)  # Interesting case!
```

**Note:** The "good movie" example shows that the model relies on overall context. Simple phrases may not have enough information for accurate classification. The word "good" appears frequently in both positive AND negative reviews (11,826 times in negative vs 11,939 times in positive), making it a weak indicator on its own.

## 🧮 How Naive Bayes Works

The classifier calculates the probability of a review being positive or negative using:

1. **Prior Probability:** Base rate of positive vs negative reviews
2. **Likelihood:** How likely each word appears in positive vs negative reviews
3. **Prediction Score:** 
   ```
   score = log(P(positive)/P(negative)) + Σ log(P(word|positive)/P(word|negative))
   ```
   - If score > 0: Predict positive
   - If score < 0: Predict negative

**Laplace Smoothing** is applied (adding 1 to all counts) to handle words that weren't seen during training.

## 📚 Learning Outcomes

This project teaches:

- Text preprocessing techniques for NLP
- Implementing machine learning algorithms from scratch
- Understanding probabilistic classification
- Evaluating model performance
- Working with real-world text datasets

## 🔮 Future Improvements

Potential enhancements for learning:

- **N-grams:** Include bigrams and trigrams (word pairs/triplets) for better context
- **TF-IDF weighting:** Give more importance to rare, informative words
- **Smoothing techniques:** Experiment with different smoothing methods
- **Feature engineering:** Add features like review length, punctuation count
- **Model comparison:** Compare with Logistic Regression, SVM, or neural networks
- **Cross-validation:** Implement k-fold cross-validation for more robust evaluation
- **Confusion matrix:** Analyze false positives and false negatives
- **Word importance:** Identify which words most strongly indicate sentiment

## 📝 Files

- `Naive_bayes.ipynb` - Main Jupyter notebook with implementation
- `requirements.txt` - Python package dependencies
- `README.md` - This file

**Note:** The dataset files (`Train.csv` and `Test.csv`) are automatically downloaded when you run the notebook using `kagglehub`. They will be cached in `~/.cache/kagglehub/` for future use.

## 📖 References

- [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format)
- [Naive Bayes Classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [NLTK Documentation](https://www.nltk.org/)

## 📄 License

This project is open-sourced under the MIT License. Feel free to use it for learning and educational purposes.

## 🤝 Contributing

This is an educational project. Feel free to fork, experiment, and learn from it!

---

**Happy Learning! 🎓**

