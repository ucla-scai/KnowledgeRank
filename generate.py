import numpy as np
import pickle
import scipy.io as sio
import joblib
from scipy import sparse
import random
import os
import utility_common
import random
# ------------------------------------------------------------------------
def getLines(loc): return utility_common.getLines(loc)
def clean(lines): return utility_common.clean(lines)
def writeLines(loc, lines): return utility_common.writeLines(loc, lines)
def sample(arr): return utility_common.sample(arr)
def isNumber(s): return utility_common.isNumber(s)
def clean(s): return utility_common.clean(s)
def eps(): return utility_common.eps()
def sample(a): return utility_common.sample(a)
# ------------------------------------------------------------------------
def getVocab(file):
    id_word = {}
    word_id = {}
    id = 0
    lines = getLines(file)
    for line in lines:
        split = clean(line).split()
        for word in [clean(x) for x in split]:
            if word not in word_id:
                word_id[word] = id
                id_word[id] = word
                id += 1
    return id_word, word_id
# ------------------------------------------------------------------------
def generateData(files):
    mu = 5.0
    D = 2000
    V_avg = 500
    mesh = getLines(files['mesh'])
    vocabs = {}
    vocabs['mesh'] = getVocab(mesh)
    ks = getLines(files['ks_parents'])
    children = getLines(files['children'])
    parents = getLines(files['parents'])
    vocabs['parents'] = getVocab(parents)
    vocabs['ks'] = getVocab(ks)
    topic_beta = {}
    for line in ks:
        split = line.split()
        topic = split[0]
        beta = []
        for i in range(len(vocabs['ks'][0])):
            beta[i] = eps()
        for i in range(1, len(split), 2):
            word = split[i]
            count = int(split[i+1])
            id = vocabs['ks'][1][word]
            beta[id] = count
        for i in range(len(vocabs['ks'][0])):
            if beta[i] == eps(): continue
            beta[i] = pow(beta[i], mu)
        topic_beta[topic] = beta
    T = len(topic_beta)
    topic_id = { x:i for i,x in enumerate(topic_beta) }
    id_topic = { topic_id[x]:x for x in topic_id }
    alpha = [50.0/float(T) for _ in range(T)]
    input = []
    key = []
    V = len(vocabs['ks'][0])
    topic_phi = {}
    for t in topic_beta:
        beta = topic_beta[t]
        phi = np.random.dirichlet(beta)
        topic_phi[t] = phi
    for d in range(D):
        theta = np.random.dirichlet(alpha)
        n = np.random.poisson(float(V_avg))
        doc = []
        t = []
        for v in range(n):
            z = sample(theta)
            topic = id_topic[z]
            if random.randint(0,1) == 0:
                phi = topic_phi[topic]
                w = sample(phi)
                word = vocabs['ks'][0][w]
                doc.append(word)
                t.append(topic)
            else:
                for w in vocabs['ks'][0]:
                    id = w
                    if vocabs[ks][1][vocabs[ks][0][id]] in vocabs['parents'][1]: break
                child = vocabs['parents'][0][0] if random.randint(0,1) == 0 else vocabs['parents'][0][-1]
                line = children[child]
                split = line.split()
                token = split[random.randint(0, len(split)-1)]
                for i in range(10):
                    child = vocabs['parent'][0][0] if random.randint(0,1) == 0 else vocabs['parent'][0][-1]
                    line = children[child]
                    split = line.split()
                    token = split[random.randint(0, len(split)-1)]
                    if token in vocabs['ks'][1]: break
                doc.append(token)
                t.add(topic)
        input.add(''.join(doc))
        key.append(''.join(t))
    return input, key
