from flask import Flask, render_template, request, jsonify
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

country_cols = [col for col in feature_cols if col.startswith('Country_')]

# Top 250 popular titles for the searchable 3-model comparison dropdown
popular_titles = (
    book_catalog.sort_values(by='Book_Popularity', ascending=False)['Book-Title']
    .dropna()
    .unique()[:250]
    .tolist()
)


def prepare_features(df_slice, age, user_avg):
    temp = df_slice.copy()
    temp['Age'] = float(age)
    temp['User_Activity'] = 20
    temp['User_Avg_Rating'] = float(user_avg)
    for col in country_cols:
        temp[col] = 0
    return temp[feature_cols]


def score_model(model_name, X_input):
    if model_name == 'Random Forest':
        return rf_model.predict_proba(X_input)[:, 1]
    elif model_name == 'Decision Tree':
        return dt_model.predict_proba(X_input)[:, 1]
    else:
        decision_scores = svm_model.decision_function(scaler.transform(X_input))
        return 1 / (1 + np.exp(-decision_scores))


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_model = 'Random Forest'
    age = 21
    user_avg = 8.0
    keyword = ''
    era = 'all'
    searched = False

    if request.method == 'POST':
        searched = True
        age = float(request.form['age'])
        user_avg = float(request.form['user_avg'])
        selected_model = request.form['model']
        keyword = request.form.get('keyword', '').strip()
        era = request.form.get('era', 'all')

        temp = book_catalog.copy()

        if keyword:
            title_match = temp['Book-Title'].fillna('').str.contains(keyword, case=False, na=False)
            author_match = temp['Book-Author'].fillna('').str.contains(keyword, case=False, na=False)
            temp = temp[title_match | author_match]

        if era == 'classic':
            temp = temp[temp['Year-Of-Publication'] < 1990]
        elif era == '90s':
            temp = temp[(temp['Year-Of-Publication'] >= 1990) & (temp['Year-Of-Publication'] < 2000)]
        elif era == 'modern':
            temp = temp[temp['Year-Of-Publication'] >= 2000]

        if not temp.empty:
            X_input = prepare_features(temp, age, user_avg)
            scores = score_model(selected_model, X_input)
            temp['Match_Score'] = np.round(scores * 100, 1)

            top_books = temp.sort_values(
                by=['Match_Score', 'Book_Popularity'], ascending=False
            ).head(6)

            recommendations = top_books[[
                'ISBN', 'Book-Title', 'Book-Author', 'Publisher',
                'Year-Of-Publication', 'Book_Avg_Rating', 'Match_Score'
            ]].to_dict(orient='records')

    return render_template(
        'index.html',
        comparison=comparison_df.to_dict(orient='records'),
        recommendations=recommendations,
        popular_titles=popular_titles,
        selected_model=selected_model,
        age=int(age),
        user_avg=user_avg,
        keyword=keyword,
        era=era,
        searched=searched,
        total_books=len(book_catalog)
    )


@app.route('/api/compare_book', methods=['POST'])
def compare_book():
    data = request.get_json()
    title = data.get('title', '').strip()
    age = float(data.get('age', 21))
    user_avg = float(data.get('user_avg', 8.0))

    match = book_catalog[
        book_catalog['Book-Title'].fillna('').str.lower() == title.lower()
    ]
    if match.empty:
        match = book_catalog[
            book_catalog['Book-Title'].fillna('').str.contains(title, case=False, na=False)
        ]

    if match.empty:
        return jsonify({'error': 'Book not found in catalog. Pick one from the suggestions.'}), 404

    book_row = match.iloc[[0]]
    X_input = prepare_features(book_row, age, user_avg)

    results = []
    for name in ['Random Forest', 'SVM (LinearSVC)', 'Decision Tree']:
        prob = float(score_model(name, X_input)[0])
        score_pct = round(prob * 100, 1)
        results.append({
            'model': name,
            'score': score_pct,
            'verdict': 'Recommended' if score_pct >= 50 else 'Skip'
        })

    row = book_row.iloc[0]
    return jsonify({
        'title': row['Book-Title'],
        'author': row['Book-Author'],
        'year': int(row['Year-Of-Publication']),
        'avg_rating': round(float(row['Book_Avg_Rating']), 2),
        'results': results
    })


if __name__ == '__main__':
    app.run(debug=True)