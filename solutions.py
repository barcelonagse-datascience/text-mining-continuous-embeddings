
#######################################
# Exercise #1
#######################################

import re
import spacy
import seaborn as sns
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

stemmer = SnowballStemmer("english")
not_alphanumeric_or_space = re.compile(r'[^(\w|\s|\d)]')
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc)
    words = [t.lemma_ for t in nlp(doc) if t.lemma_ != '-PRON-']
    return ' '.join(words).lower()

vectorizer = TfidfVectorizer(min_df=2,
                             max_df=.8,
                             preprocessor=preprocess,
                             stop_words='english',
                             use_idf=False,
                             norm=False)

v = vectorizer.fit_transform(docs)
v = np.asarray(v.todense())

labels, _ = zip(*sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]))

def pca_loadings(v):
    pca = TruncatedSVD(2)
    pca.fit(v)
    return pca.components_.T

def plot_words(vecs, labels):
    df = pd.DataFrame(vecs).assign(labels = labels)
    ax = sns.scatterplot(x=0, y=1, data=df)
    for _, r in df.iterrows():
        ax.text(r[0] + 0.01 , r[1], r.labels)

word_vecs = pca_loadings(v)
plot_words(word_vecs, labels)


#######################################
# Exercise #2
#######################################


import torch
import torch.nn.functional as F

def train(V, y, components, epochs):
    try:
        V = V.toarray()
    except AttributeError:
        pass

    Y = torch.tensor(y, dtype=torch.float32)
    X = torch.tensor(V, dtype=torch.float32)

    C = torch.tensor(components, dtype=torch.float32, requires_grad=True)
    beta = torch.randn((components.shape[0], 1), requires_grad=True)

    opt = torch.optim.Adam([C, beta], lr=0.01)
    criterion = torch.nn.BCELoss()

    for i in range(epochs):
        L = torch.mm(X, C.T)
        out = torch.mm(L, beta)
        p = torch.sigmoid(out)
        loss = criterion(p, Y)

        if i % 20 == 0:
            print(loss)

        loss.backward()
        opt.step()
        opt.zero_grad()

    return C.detach().numpy()


learned_vecs = train(v, y, word_vecs.T, 200)
plot_words(learned_vecs.T, labels)


#######################################
# Exercise #3
#######################################

def plot_heatmap(d, labels):
    d = pd.DataFrame(d, index=labels, columns=labels)
    sns.heatmap(d)

plot_heatmap(v.T@v, labels)

plot_heatmap(word_vecs@word_vecs.T, labels)


#######################################
# Exercise #4
#######################################


def _weight(pos, i):
    try:
        return 1 / abs(pos - i)
    except ZeroDivisionError:
        return 0

def weights(pos, l):
    return [_weight(pos, i) for i in range(l)]

def weighted_cooccurence(docs, vocab):
    V = len(vocab)
    M = np.zeros((V, V))

    docs = [preprocess(doc) for doc in docs]
    docs = [[vocab.get(word) for word in doc.split()
             if vocab.get(word)]
            for doc in docs]

    for doc in docs:
        for idx, i in enumerate(doc):
            for w, j in zip(weights(idx, len(doc)), doc):
                M[i, j] += w
    return M


plot_heatmap(weighted_cooccurence(docs, vectorizer.vocabulary_), labels)


def glove_wannabe(docs, vocab, epochs):

    target = weighted_cooccurence(docs, vectorizer.vocabulary_)

    T = torch.tensor(target, dtype=torch.float32)
    W = torch.randn(word_vecs.shape, requires_grad=True)

    opt = torch.optim.Adam([W], lr=0.1)

    for i in range(epochs):
        attmpt = torch.mm(W, W.T)
        loss = torch.dist(attmpt, T)
        if i % 20 == 0:
            print(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    return W.detach().numpy()

learned_vecs = glove_wannabe(docs, vectorizer.vocabulary_, 200)
plot_words(learned_vecs, labels)
