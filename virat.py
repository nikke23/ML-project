import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Read the data
df = pd.read_csv('netflix_titles.csv')
df = df[['title', 'description']].dropna().reset_index(drop=True)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Preprocess descriptions
df['clean_description'] = df['description'].apply(preprocess)

# Vectorize the cleaned descriptions
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_description'])

# Clustering
k = 5  # Number of clusters
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)
df['cluster'] = model.labels_

def show_cluster_keywords(model, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    for i, center in enumerate(model.cluster_centers_):
        top_terms = center.argsort()[-n_terms:][::-1]
        print(f"\nCluster {i} Keywords: {', '.join(terms[j] for j in top_terms)}")

show_cluster_keywords(model, vectorizer)

for i in range(k):
    print(f"\n=== Cluster {i} ===")
    print(df[df['cluster'] == i][['title', 'description']].head(3))

# Dimensionality reduction for visualization
reduced = PCA(n_components=2).fit_transform(X.toarray())
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df['cluster'], cmap='rainbow', alpha=0.6)
plt.title('Movie Description Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'kmeans_model.pkl')
