from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load saved models, scaler, and data
rf_model = joblib.load('rf_book_model.pkl')
dt_model = joblib.load('dt_book_model.pkl')
svm_model = joblib.load('svm_book_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

comparison_df = pd.read_csv('model_comparison.csv')
book_catalog = pd.read_csv('book_catalog.csv')

# Extract available country options dynamically from feature_cols
country_cols = [col for col in feature_cols if col.startswith('Country_')]
countries = sorted([col.replace('Country_', '') for col in country_cols]) + ['other']

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_model = 'Random Forest'
    age = 20
    user_avg = 8.0

    if request.method == 'POST':
        age = float(request.form['age'])
        user_avg = float(request.form['user_avg'])
        selected_model = request.form['model']

        # Build feature matrix for all 3,331 books
        temp = book_catalog.copy()
        temp['Age'] = age
        temp['User_Activity'] = 20
        temp['User_Avg_Rating'] = user_avg

        # Set country columns to 0 (neutral baseline) in the background
        for col in country_cols:
            temp[col] = 0

        X_input = temp[feature_cols]

        if selected_model == 'Random Forest':
            scores = rf_model.predict_proba(X_input)[:, 1]
        elif selected_model == 'Decision Tree':
            scores = dt_model.predict_proba(X_input)[:, 1]
        else:
            decision_scores = svm_model.decision_function(scaler.transform(X_input))
            scores = 1 / (1 + np.exp(-decision_scores))

        temp['Match_Score'] = np.round(scores * 100, 1)

        top_books = temp.sort_values(
            by=['Match_Score', 'Book_Popularity'], ascending=False
        ).head(5)

        recommendations = top_books[[
            'Book-Title', 'Book-Author', 'Year-Of-Publication',
            'Book_Avg_Rating', 'Match_Score'
        ]].to_dict(orient='records')

    return render_template(
        'index.html',
        comparison=comparison_df.to_dict(orient='records'),
        recommendations=recommendations,
        selected_model=selected_model,
        age=int(age),
        user_avg=user_avg
    )

if __name__ == '__main__':
    app.run(debug=True)