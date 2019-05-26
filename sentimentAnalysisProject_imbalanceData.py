# -*- coding: utf-8 -*-

from pythainlp.tokenize import word_tokenize

import re

import numpy as np

from nltk import FreqDist, precision, recall, f_measure, NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.classify import util

from sklearn.model_selection import KFold
import collections, itertools
import random, math


pos = [(line.strip(), 'pos') for line in open("pos.txt", 'r', encoding = 'utf-8')]
neg = [(line.strip(), 'neg') for line in open("neg.txt", 'r', encoding = 'utf-8')]

dataTrainPos = []
dataTrainNeg = []
dataTestPos = []
dataTestNeg = []

for i in pos:
    if random.randint(0,1) == 0 :
        if len(dataTestPos) < math.floor(len(pos)*0.3) :
            dataTestPos.append(i)
        else :
            dataTrainPos.append(i)
    else :
        if len(dataTrainPos) < math.floor(len(pos)*0.7) :
            dataTrainPos.append(i)
        else :
            dataTestPos.append(i)

for j in neg:
    if random.randint(0,1) == 0 :
        if len(dataTestNeg) < math.floor(len(neg)*0.3) :
            dataTestNeg.append(j)
        else :
            dataTrainNeg.append(j)
    else :
        if len(dataTrainNeg) < math.floor(len(neg)*0.7) :
            dataTrainNeg.append(j)
        else :
            dataTestNeg.append(j)


def split_words (sentence):
    return word_tokenize(sentence.lower(), engine="newmm")
sentences = [(split_words(sentence), sentiment) for (sentence, sentiment) in dataTrainPos + dataTrainNeg]
testSentences = [(split_words(sentence), sentiment) for (sentence, sentiment) in dataTestPos + dataTestNeg]

def get_words_in_sentences(sentences):
    all_words = []
    for (words, sentiment) in sentences:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    word_features = [word[0] for word in wordlist.most_common()]
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


features_data = np.array(sentences)
features_data_test = np.array(testSentences)

k_fold = KFold(n_splits=10, random_state=1992, shuffle=True)
word_features = None
accuracy_scores = []
accuracy_data_scores = []
for train_set, test_set in k_fold.split(features_data):
    word_features = get_word_features(get_words_in_sentences(features_data[train_set].tolist()))
    train_features = apply_features(extract_features, features_data[train_set].tolist())
    test_features = apply_features(extract_features, features_data[test_set].tolist())
    classifier = NaiveBayesClassifier.train(train_features)
    
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    testdata_features = apply_features(extract_features, features_data_test.tolist())
    refdatasets = collections.defaultdict(set)
    testdatasets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_features):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    for i, (feats, label) in enumerate(testdata_features):
        refdatasets[label].add(i)
        observed = classifier.classify(feats)
        testdatasets[observed].add(i)
    
    accuracy_scores.append(util.accuracy(classifier, test_features))
    accuracy_data_scores.append(util.accuracy(classifier, testdata_features))
    print('train: {} test: {}'.format(len(train_set), len(test_set)))
    print('=================== Results ===================')
    print('Accuracy {:f}'.format(accuracy_scores[-1]))
    print('            Positive     Negative')
    print('F1         [{:f}     {:f}]'.format(
        f_measure(refsets['pos'], testsets['pos']),
        f_measure(refsets['neg'], testsets['neg'])
    ))
    print('Precision  [{:f}     {:f}]'.format(
        precision(refsets['pos'], testsets['pos']),
        precision(refsets['neg'], testsets['neg'])
    ))
    print('Recall     [{:f}     {:f}]'.format(
        recall(refsets['pos'], testsets['pos']),
        recall(refsets['neg'], testsets['neg'])
    ))
    print('===============================================\n')
    print('testData: {}'.format(len(testSentences)))
    print('=================== Results ===================')
    print('Accuracy TestData {:f}'.format(accuracy_data_scores[-1]))
    print('F1         [{:f}     {:f}]'.format(
        f_measure(refdatasets['pos'], testdatasets['pos']),
        f_measure(refdatasets['neg'], testdatasets['neg'])
    ))
    print('Precision  [{:f}     {:f}]'.format(
        precision(refdatasets['pos'], testdatasets['pos']),
        precision(refdatasets['neg'], testdatasets['neg'])
    ))
    print('Recall     [{:f}     {:f}]'.format(
        recall(refdatasets['pos'], testdatasets['pos']),
        recall(refdatasets['neg'], testdatasets['neg'])
    ))
    print('===============================================\n')
    

accuracy_scores_sum = 0
for i in accuracy_scores:
    accuracy_scores_sum+=i

accuracy_data_scores_sum = 0
for i in accuracy_data_scores:
    accuracy_data_scores_sum+=i

print('\n===============================================\n')
print("average of accuracy scores : " , accuracy_scores_sum/10)
print("average of accuracy data scores : " , accuracy_data_scores_sum/10)
print('\n===============================================\n')
print

lsData = ["FutureForward.txt", "Democrat.txt", "Pprp.txt", "PheuThai.txt", "Bhumjaithai.txt"]
lsPos = []
lsNeg = []
lsAns = []

for dataPartyList in lsData :
    dataInDataPartyList = [line.strip() for line in open(dataPartyList, 'r', encoding = 'utf-8')]
    countPos = 0
    countNeg = 0
    for sentence in dataInDataPartyList :
        sentiment = classifier.classify(extract_features(split_words(sentence)))
        if (sentiment == "neg"):
            countNeg+=1
        elif (sentiment == "pos") :
            countPos+=1
    lsPos.append(countPos)
    lsNeg.append(countNeg)
    if (countPos > countNeg) :
        lsAns.append("positive")
    elif (countNeg > countPos) :
        lsAns.append("negative")
    else :
        lsAns.append("equal")


for index in range(len(lsData)) :
    print('{} : {} (pos = {:f} %,neg = {:f} %)'.format(
        lsData[index], 
        lsAns[index], 
        lsPos[index]*100/(lsPos[index]+lsNeg[index]), 
        lsNeg[index]*100/(lsPos[index]+lsNeg[index])
        )
        )