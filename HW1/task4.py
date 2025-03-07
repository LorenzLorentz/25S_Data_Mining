import os
import sys
import math
import numpy as np
from collections import Counter, defaultdict

sys.stdout = open(f'task4.txt', 'w')

# 1. 文档的表示: 构造词典和计算TF-IDF向量
def build_vocab_and_tfidf_vectors(corpus_dir):
    doc_count = 0
    term_doc_freq = defaultdict(int)
    term_freqs = []
    
    # 读取所有文档，计算词频
    for filename in os.listdir(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            words = file.read().split()
            doc_count += 1
            term_freq = Counter(words)
            term_freqs.append(term_freq)
            for word in term_freq:
                term_doc_freq[word] += 1
    
    vocab = {word: idx for idx, word in enumerate(term_doc_freq.keys())}
    
    # 计算TF-IDF向量
    tfidf_vectors = []
    for term_freq in term_freqs:
        tfidf_vector = np.zeros(len(vocab))
        for word, freq in term_freq.items():
            if word in vocab:
                tf = freq / sum(term_freq.values())
                idf = math.log(doc_count / (1 + term_doc_freq[word]))
                tfidf_vector[vocab[word]] = tf * idf
        tfidf_vectors.append(tfidf_vector)
    
    return vocab, np.array(tfidf_vectors)

# 2. 计算词语共现矩阵
def build_cooccurrence_matrix(corpus_dir, vocab):
    cooc_matrix = np.zeros((len(vocab), len(vocab)))
    
    for filename in os.listdir(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            words = file.read().split()
            unique_words = set(words)
            for word1 in unique_words:
                if word1 in vocab:
                    idx1 = vocab[word1]
                    for word2 in unique_words:
                        if word2 in vocab:
                            idx2 = vocab[word2]
                            cooc_matrix[idx1][idx2] += 1
    
    return cooc_matrix

# 计算欧式距离和余弦相似度
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 3. 找到最相似的文档
def find_similar_docs(tfidf_vectors, doc_idx, top_n=5):
    distances = []
    similarities = []
    target_vector = tfidf_vectors[doc_idx]
    
    for i, vector in enumerate(tfidf_vectors):
        if i != doc_idx:
            distances.append((i, euclidean_distance(target_vector, vector)))
            similarities.append((i, cosine_similarity(target_vector, vector)))
    
    distances.sort(key=lambda x: x[1])
    similarities.sort(key=lambda x: -x[1])
    
    return distances[:top_n], similarities[:top_n]

# 4. 找到最相似的词语
def find_similar_words(cooc_matrix, vocab, word, top_n=5):
    if word not in vocab:
        return [], []
    
    word_idx = vocab[word]
    target_vector = cooc_matrix[word_idx]
    
    distances = []
    similarities = []
    
    for i, vector in enumerate(cooc_matrix):
        if i != word_idx:
            distances.append((i, euclidean_distance(target_vector, vector)))
            similarities.append((i, cosine_similarity(target_vector, vector)))
    
    distances.sort(key=lambda x: x[1])
    similarities.sort(key=lambda x: -x[1])
    
    inv_vocab = {idx: word for word, idx in vocab.items()}
    
    return [(inv_vocab[i], dist) for i, dist in distances[:top_n]], [(inv_vocab[i], sim) for i, sim in similarities[:top_n]]

# 示例用法
corpus_dir = 'nyt_corp0/'
vocab, tfidf_vectors = build_vocab_and_tfidf_vectors(corpus_dir)
cooc_matrix = build_cooccurrence_matrix(corpus_dir, vocab)

# 文档相似度分析
distances, similarities = find_similar_docs(tfidf_vectors, doc_idx=0)
print('欧式距离最近的5篇文档:', distances)
print('余弦相似度最高的5篇文档:', similarities)

# 词语相似度分析
distances, similarities = find_similar_words(cooc_matrix, vocab, word='news')
print('欧式距离最近的5个词:', distances)
print('余弦相似度最高的5个词:', similarities)