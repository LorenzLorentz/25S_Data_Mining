import os
import sys
import math
import numpy as np
from collections import Counter, defaultdict

sys.stdout=open(f'task4.txt', 'w')

def build_vocab_tifdif(dir):
    doc_cnt=0
    word_freqs={}
    doc_freqs=[]
    
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        with open(path, 'r') as file:
            doc_cnt+=1
            words=file.read().split()
            doc_freqs.append(Counter(words))                
            for word in set(words):
                if word in word_freqs:
                    word_freqs[word]+=1
                else:
                    word_freqs[word]=1
    
    vocab={word: idx for idx, word in enumerate(word_freqs.keys())}

    tfidf_vecs=[]
    for freqs in doc_freqs:
        tfidf_vec=np.zeros(len(vocab))
        for word, freq in freqs.items():
            tf=freq/sum(freqs.values())
            idf=math.log(doc_cnt/(1+word_freqs[word]))
            tfidf_vec[vocab[word]]=tf*idf
        tfidf_vecs.append(tfidf_vec)
    
    return vocab, np.array(tfidf_vecs)

def build_cooc_matrix(dir, vocab):
    cooc_matrix = np.zeros((len(vocab), len(vocab)))
    
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        with open(path, 'r') as file:
            words=set(file.read().split())
            for word1 in words:
                idx1=vocab[word1]
                for word2 in words:
                    idx2=vocab[word2]
                    cooc_matrix[idx1][idx2]+=1
    
    return cooc_matrix

def distance(vec1, vec2):
    return np.linalg.norm(vec1-vec2)

def cos_similar(vec1, vec2):
    inner=np.dot(vec1, vec2)
    len1=np.linalg.norm(vec1)
    len2=np.linalg.norm(vec2)
    return inner/(len1*len2+1e-8)

def inner_distance(vec1, vec2):
    return np.dot(vec1, vec2)
    # return np.linalg.norm(vec1/np.linalg.norm(vec1)-vec2/np.linalg.norm(vec2))

def similar_docs(tfidf_vecs, doc, top_n=5):
    dists=[]
    coss=[]
    inner_dists=[]
    target=tfidf_vecs[doc]
    
    for i, vec in enumerate(tfidf_vecs):
        if i!=doc:
            dists.append((i, distance(target, vec)))
            coss.append((i, cos_similar(target, vec)))
            inner_dists.append((i, inner_distance(target, vec)))
    
    dists.sort(key=lambda x: x[1])
    coss.sort(key=lambda x: -x[1])
    inner_dists.sort(key=lambda x: x[1])
    
    return dists[:top_n], coss[:top_n], inner_dists[:top_n]

def similar_words(cooc_matrix, vocab, word, top_n=5):
    if word not in vocab:
        return [], []
    
    target=cooc_matrix[vocab[word]]
    
    dists=[]
    coss=[]
    inner_dists=[]
    
    for i, vec in enumerate(cooc_matrix):
        if i != vocab[word]:
            dists.append((i, distance(target, vec)))
            coss.append((i, cos_similar(target, vec)))
            inner_dists.append((i, inner_distance(target, vec)))
    
    dists.sort(key=lambda x: x[1])
    coss.sort(key=lambda x: -x[1])
    inner_dists.sort(key=lambda x: x[1])
    
    inv_vocab = {idx: word for word, idx in vocab.items()}
    
    return [(inv_vocab[i], dist) for i, dist in dists[:top_n]], [(inv_vocab[i], cos) for i, cos in coss[:top_n]], [(inv_vocab[i], inner_dist) for i, inner_dist in inner_dists[:top_n]]

dir='nyt_corp0/'
vocab, tfidf_vecs=build_vocab_tifdif(dir)
cooc_matrix=build_cooc_matrix(dir, vocab)

doc=0
dists, coss, inner_dists=similar_docs(tfidf_vecs, doc=doc)
print(f'与文档{doc}欧式距离最近的5篇文档:', dists)
print(f'与文档{doc}余弦相似度最高的5篇文档:', coss)
print(f'与文档{doc}内积距离最近的5篇文档:', inner_dists)

word='news'
dists, coss, inner_dists=similar_words(cooc_matrix, vocab, word=word)
print(f'与单词{word}欧式距离最近的5个词:', dists)
print(f'与单词{word}余弦相似度最高的5个词:', coss)
print(f'与文档{word}内积距离最近的5个词:', inner_dists)