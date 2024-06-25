import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(newsgroups.data)
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
for i in range(k):
    print(f"Cluster {i + 1}:")
    top_terms = [terms[ind] for ind in order_centroids[i, :5]]
    print(top_terms)
