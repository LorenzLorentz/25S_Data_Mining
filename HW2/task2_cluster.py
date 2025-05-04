from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from string import punctuation

def preprocess(text:str, stop_words:set) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def tf_idf_vectorize(texts:list[str]) -> tuple:
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts), vectorizer.get_feature_names_out()

def k_means(k:int, vectors, terms) -> None:
    print(f"k={k}")
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vectors)
    centers = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    for i in range(k):
        print(f"cluster {i}'s keywords")
        keywords = [terms[ind] for ind in centers[i, :10]]
        print(", ".join(keywords))
    print("")

def main():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    with open('twitter.txt', 'r') as f:
        tweets = [line.strip() for line in f if line.strip()]
    
    tweets = [preprocess(tweet, stop_words) for tweet in tweets]

    tweets_vectorized, terms = tf_idf_vectorize(tweets)

    for k in [2, 3, 4]:
        k_means(k, tweets_vectorized, terms)

if __name__ == "__main__":
    main()