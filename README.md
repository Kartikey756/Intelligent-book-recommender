# Intelligent Library Book Recommendation System

A machine learning-based book recommendation prototype trained on the Book-Crossing dataset. The application compares three classification algorithms—**Random Forest**, **SVM (`LinearSVC`)**, and **Decision Tree**—to predict reader preference probabilities and serve recommendations through a responsive Flask web interface.

## Features

- **Personalized Recommendations:** Adjust reader age, rating strictness (1–10), classifier choice, and optional title/author keywords to generate the top 6 book matches.
- **Single-Book 3-Model Comparison:** Select any book from the catalog to run live inference across Random Forest, LinearSVC, and Decision Tree side by side.
- **Model Evaluation Benchmark:** Interactive Chart.js bar chart and metrics table comparing Accuracy, Precision, Recall, and F1-Score on a 20% held-out test split.
- **Reading List:** Bookmark recommended books locally in the browser during a session.

## Model Summary

All classifiers were trained on engineered user and catalog features (`Age`, `User_Avg_Rating`, `User_Activity`, `Book_Avg_Rating`, `Book_Popularity`, `Year-Of-Publication`, and demographic encodings) using a leak-free train/test split:

- **Random Forest (Best Performing):** Achieved **78.75% F1-Score** and **70.72% Accuracy** on the held-out test set.
- **SVM (`LinearSVC`):** Evaluated with `StandardScaler` preprocessing and sigmoid probability calibration.
- **Decision Tree:** Baseline interpretable tree classifier for direct performance comparison.

## Project Structure

```text
├── app.py                  # Flask server and prediction API routes
├── rf_book_model.pkl       # Trained Random Forest model
├── svm_book_model.pkl      # Trained LinearSVC model
├── dt_book_model.pkl       # Trained Decision Tree model
├── scaler.pkl              # Feature scaler for LinearSVC
├── feature_cols.pkl        # Saved feature schema
├── book_catalog.csv        # Processed book catalog and aggregated ratings
├── model_comparison.csv    # Test-set evaluation metrics
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main web interface
└── static/
    └── style.css           # Responsive styling
```

## Setup and Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/intelligent-book-recommender.git](https://github.com/your-username/intelligent-book-recommender.git)
   cd intelligent-book-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://127.0.0.1:5000`.

## Tech Stack

- **Backend & ML:** Python, Flask, scikit-learn, pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js