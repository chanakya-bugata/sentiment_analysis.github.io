from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the saved model and vectorizer
model_path = 'models/best_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
else:
    print("Error: Model or vectorizer file not found. Please make sure to train the model first.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        sentiment = predict_sentiment(user_input)
        return render_template('index.html', sentiment=sentiment, user_input=user_input)
    else:
        return render_template('index.html', sentiment=None, user_input=None)

def predict_sentiment(text):
    text_vector = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
