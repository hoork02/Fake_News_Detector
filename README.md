# 📰 Fake News Detector

A machine learning-based application to detect and classify news articles as **REAL** or **FAKE** using text classification techniques.

---

## 🎯 Problem Statement

Misinformation spreads at unprecedented speed across digital platforms, eroding public trust and enabling uninformed decision-making. Users lack an automated, scalable mechanism to verify news authenticity in real-time—manual fact-checking is time-consuming and inaccessible at scale.

## 💡 Solution

Built a production-ready NLP + ML pipeline that:
- **Preprocesses** news content with advanced text cleaning (URL removal, stemming, stopword removal)
- **Vectorizes** text into numerical features using TF-IDF (5000 features)
- **Trains multiple classifiers** (Logistic Regression & Naive Bayes) to identify the best-performing model
- **Deploys** as an interactive Streamlit web app for real-time classification with confidence scoring

## 📊 Impact

- **Real-Time Classification**: Users can instantly verify news articles without manual research
- **High Accuracy**: Compares multiple ML models and selects the best performer
- **Confidence Scoring**: Provides prediction confidence percentages for informed decision-making
- **Scalable & Accessible**: Web interface enables non-technical users to combat misinformation at scale

---

## ✨ Features

- **Text Preprocessing**: Advanced text cleaning with URL removal, punctuation handling, stopword removal, and stemming
- **Multiple ML Models**: Supports Logistic Regression and Naive Bayes classifiers
- **TF-IDF Vectorization**: Converts text to numerical features for model training
- **Web Interface**: Interactive Streamlit app for easy news classification
- **Confidence Scoring**: Provides confidence percentages for predictions
- **Model Persistence**: Trained models and vectorizers are saved for quick inference

## 📋 Project Structure

```
Fake_News_Detector/
├── app.py                    # Streamlit web application
├── train.py                  # Model training pipeline
├── preprocess.py             # Text preprocessing utilities
├── models/                   # Directory for saved models
│   ├── fake_news_model.pkl   # Trained classification model
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hoork02/Fake_News_Detector.git
cd Fake_News_Detector
```

2. Install required dependencies:
```bash
pip install pandas scikit-learn joblib nltk seaborn matplotlib streamlit
```

3. Download NLTK data (required for stopwords):
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Dataset

The project expects a `News.csv` file with the following columns:
- `title`: Article title
- `text`: Article content
- `class`: Label (0 for Fake, 1 for Real)

Place your dataset in the project root directory before training.

## 📖 Usage

### Training the Model

To train and evaluate the models:

```bash
python train.py
```

This script will:
1. Load and preprocess the news dataset
2. Split data into training and testing sets (80/20 split)
3. Train multiple classifiers (Logistic Regression, Naive Bayes)
4. Display accuracy and classification reports
5. Save the best performing model and vectorizer to the `models/` directory
6. Generate and display a confusion matrix

### Running the Web Application

To launch the interactive Streamlit app:

```bash
streamlit run app.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Enter any news content in the text area
3. Click "Detect" to classify the news as REAL or FAKE
4. View the prediction with confidence score

## 🔧 How It Works

### Text Preprocessing (`preprocess.py`)
- Converts text to lowercase
- Removes URLs
- Removes punctuation and numbers
- Tokenizes text into words
- Removes English stopwords
- Applies Porter Stemming for word normalization

### Model Training (`train.py`)
- Combines article titles and text
- Applies TF-IDF vectorization (max 5000 features)
- Trains multiple classifiers
- Selects the best model based on accuracy
- Saves model and vectorizer for inference

### Prediction (`app.py`)
- Preprocesses user input
- Vectorizes text using saved TF-IDF vectorizer
- Makes prediction using the saved model
- Calculates confidence score
- Displays results with color-coded output

## 📊 Model Performance

The application trains and compares multiple models:

| Model | Performance |
|-------|-------------|
| Logistic Regression | High accuracy, good generalization |
| Naive Bayes | Fast training, reasonable accuracy |

The best performing model is automatically saved and used for predictions.

## 🔐 Key Technologies

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **nltk**: Natural language processing
- **joblib**: Model serialization
- **streamlit**: Web application framework
- **matplotlib/seaborn**: Data visualization

## 🚨 Important Notes

- The `News.csv` dataset is required for training
- Trained models must be generated before running the app
- The model's accuracy depends on training data quality and size
- Confidence scores are calculated using the decision function or default fallback

## 🐛 Troubleshooting

**Model not found error**: Ensure you've run `train.py` first to generate the model files

**NLTK stopwords error**: Run `python -c "import nltk; nltk.download('stopwords')"`

**Empty input warning**: The app requires non-empty text input for classification

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

Created by [hoork02](https://github.com/hoork02)

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for any improvements.

---

**Note**: This detector is a machine learning model and should be used as a tool to assist in identifying potential misinformation, not as a definitive truth authority. Always verify information through multiple trusted sources.
