from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('twitter.txt', 'r', encoding='utf-8') as f:
    tweets = [line.strip() for line in f if line.strip()]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

tweets_cleaned = [preprocess(tweet) for tweet in tweets]

vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(tweets_cleaned)

for k in [2, 3, 4]:
    print(f'\n===== K = {k} =====')
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(k):
        print(f"\n--- 类别 {i} 的关键词 ---")
        keywords = [terms[ind] for ind in order_centroids[i, :10]]
        print(", ".join(keywords))