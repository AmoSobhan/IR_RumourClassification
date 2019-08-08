from __future__ import unicode_literals
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score
from random import shuffle
import csv
import hazm as hz
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def delete_Punc(str):
    punctuations = '''1234567890۱۲۳۴۵۶۷۸۹۰!()-[]{};:'"\,<>./?@#$%^&*_~/.><؟|؛\«:»{}[]()*،×٪٬!÷'''

    no_punct = ""
    for char in str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def textNormalizer(lousyCollection):
    docs = list()
    normalizer = hz.Normalizer()
    lemmatizer = hz.Lemmatizer()
    stemmer = hz.Stemmer()
    for i in range(len(lousyCollection)):
        normalized = normalizer.normalize(lousyCollection[i])
        docs.append(delete_Punc(normalized))
    for doc in docs:
        tokens = hz.word_tokenize(doc)
        for token in tokens:
            tokens[tokens.index(token)] = lemmatizer.lemmatize(stemmer.stem(token))
        docs[docs.index(doc)] = tokens
    return docs


def collectionToterms(docs):
    terms = list()
    prepositions = ['به', 'با', 'بر', 'در', 'که', 'و', 'طور', 'هر', 'تا', 'برای', 'هم', 'را', 'آن', 'این', 'از']
    for doc in docs:
        for token in doc:
            if (token not in terms) and (token not in prepositions):
                terms.append(token)
    print(len(terms))
    return terms, len(terms)


def set_df(docs, terms):
    df_table = [0] * len(terms)
    for term, i in zip(terms, range(len(terms))):
        for doc in docs:
            if term in doc:
                df_table[i] += 1
    return df_table


def calc_tf(doc, term):
    tf = 0
    for element in doc:
        if element == term:
            tf += 1
    return tf


def calc_tf_idf(tf, df, N):
    x = np.log10(1 + tf) * np.log10(N / df)
    return x


def set_tf_idf(docs, terms, df_table, N):
    tf_table = []  # len must be 30 and 1187
    tf_idf_table = []  # len must be 30 and 1187
    for doc in docs:
        tf = []
        tf_idf = []
        for term, i in zip(terms, range(len(terms))):
            t = calc_tf(doc, term)
            tf_idf.append(calc_tf_idf(t, df_table[i], N))
            tf.append(t)
        tf_table.append(tf)
        tf_idf_table.append(tf_idf)
    return tf_table, tf_idf_table


counter = 0
nonRumourPaper = list()
RumourPaper = list()
docValue = list()
id = list()
refNum = list()
with open('nonRumour.csv', newline='') as csvfile:
    nonRumour = csv.reader(csvfile)
    for row in nonRumour:
        nonRumourPaper.append(row)
with open('rumour.csv', newline='') as csvfile:
    Rumour = csv.reader(csvfile)
    for row in Rumour:
        RumourPaper.append(row)

lousyCollection = list()
pairedLousyCollection = list()
for i in range(len(RumourPaper)):
    if RumourPaper[i][16] == 'R':
        listPairCollection = list()
        tempdoc = RumourPaper[i][7] + RumourPaper[i][14]
        # lousyCollection.append(tempdoc)
        # docValue.append(-1)
        listPairCollection.append(tempdoc)
        listPairCollection.append(-1)
        pairedLousyCollection.append(listPairCollection)
        # id.append(RumourPaper[i][3])
        # refNum.append(RumourPaper[i][1])
        counter += 1
for i in range(len(nonRumourPaper)):
    if nonRumourPaper[i][16] == 'V':
        listPairCollection = list()
        tempdoc = nonRumourPaper[i][7] + nonRumourPaper[i][14]
        # lousyCollection.append(tempdoc)
        # docValue.append(1)
        listPairCollection.append(tempdoc)
        listPairCollection.append(1)
        pairedLousyCollection.append(listPairCollection)
        # id.append(nonRumourPaper[i][3])
        # refNum.append(nonRumourPaper[i][1])
        counter += 1
for i in range(len(RumourPaper)):
    if RumourPaper[i][16] == 'U':
        listPairCollection = list()
        tempdoc = RumourPaper[i][7] + RumourPaper[i][14]
        # lousyCollection.append(tempdoc)
        # docValue.append(-1)
        listPairCollection.append(tempdoc)
        listPairCollection.append(0)
        pairedLousyCollection.append(listPairCollection)
        # id.append(RumourPaper[i][3])
        # refNum.append(RumourPaper[i][1])
        counter += 1
print(counter)
#To get a beeter distribution of docs shuffleing inorder i used
#Shuffle notRandom
for i in range(21):
    docValue.append(pairedLousyCollection[i][1])
    docValue.append(pairedLousyCollection[41-i][1])
    lousyCollection.append(pairedLousyCollection[i][0])
    lousyCollection.append(pairedLousyCollection[41-i][0])
#ShRandom
# np.random.shuffle(pairedLousyCollection)
# for i in range(counter):
#     docValue.append(pairedLousyCollection[i][1])
#     lousyCollection.append(pairedLousyCollection[i][0])
trueCounter = counter
print(counter)
docs = textNormalizer(lousyCollection)
terms, lenterms = collectionToterms(docs)
tf_table, tf_idf_table = set_tf_idf(docs, terms, df_table=set_df(docs, terms), N=counter)
df_table = set_df(docs, terms)

trainLen = counter // 10
all_ypreds = 0
for i in range(10):
    trainData = [[0] * lenterms] * trainLen
    testDocValue = list()
    testData = [[0] * lenterms] * (counter - trainLen)
    trainDocValue = list()
    predicted = [[0] * lenterms]
    y_pred = list()
    if i != 9 and i != 0:
        j = 0
        while j != (i * trainLen):
            testData[j] = tf_idf_table[j]
            testDocValue.append(docValue[j])
            j += 1
        trainData = tf_idf_table[i * trainLen:(i + 1) * trainLen]
        trainDocValue = docValue[i * trainLen:(i + 1) * trainLen]
        j += trainLen
        while j != counter:
            testData[j - trainLen] = tf_idf_table[j]
            testDocValue.append(docValue[j])
            j += 1
    elif i == 9:
        trainData = tf_idf_table[i * trainLen:]
        trainDocValue = docValue[i * trainLen:]
        testData = tf_idf_table[:i * trainLen]
        testDocValue = docValue[:i * trainLen]
    else:
        trainData = tf_idf_table[:(i + 1) * trainLen]
        trainDocValue = docValue[:(i + 1) * trainLen]
        testData = tf_idf_table[(i + 1) * trainLen:]
        testDocValue = docValue[(i + 1) * trainLen:]

    kNearestNeighbors = KNeighborsClassifier(n_neighbors=4).fit(trainData, trainDocValue)
    for row in testData:
        predicted[0] = row
        y_pred.append(kNearestNeighbors.predict(predicted))
        # print(kNearestNeighbors.predict(predicted))
    # print("Each Fold_Score is : ")
    # print(accuracy_score(testDocValue, y_pred))
    all_ypreds += accuracy_score(testDocValue, y_pred)
print("\n"+ "The Average_Score is : ")
print(all_ypreds / 10)

