from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('kmeans_model.pkl')

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    desc = data.get('description', '')
    clean = preprocess(desc)
    vec = vectorizer.transform([clean])
    cluster = int(model.predict(vec)[0])
    return jsonify({'cluster': cluster})

if __name__ == '__main__':
    app.run(debug=True)