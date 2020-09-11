import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import random
from collections import defaultdict
from sklearn.model_selection import KFold
from statistics import mean
from sklearn import svm


# creating data set
def createDataSet(training_data, testing_data):
    stemmer = SnowballStemmer("english")
    featureDict = defaultdict(lambda: 0)
    for sentence in training_data:
        for word, _ in sentence:
            stem = stemmer.stem(word)
            index = word.find(stem)
            if index != -1:
                featureDict[word[:index]] = 0
                featureDict[word[index + len(stem):]] = 0

    for sentence in testing_data:
        for word, _ in sentence:
            stem = stemmer.stem(word)
            index = word.find(stem)
            if index != -1:
                featureDict[word[:index]] = 0
                featureDict[word[index + len(stem):]] = 0

    print("feature dictionary size: " + str(len(featureDict)))

    X_train = []
    Y_train = []
    wordDict = defaultdict()
    for sentence in training_data:
        for word, tag in sentence:
            if wordDict.get(word) is None:
                x_dict = featureDict.copy()
                stem = stemmer.stem(word)
                index = word.find(stem)
                if index != -1:
                    x_dict[word[:index]] = 1
                    x_dict[word[index + len(stem):]] = 1
                x = np.array(list(x_dict.values()))
                wordDict[word] = x
                X_train.append(x)
                Y_train.append(tag)
            else:
                x = wordDict[word]
                X_train.append(x)
                Y_train.append(tag)

    print("training data size: " + str(len(X_train)) + " " + str(len(Y_train)))

    X_test = []
    Y_test = []
    for sentence in testing_data:
        for word, tag in sentence:
            if wordDict.get(word) is None:
                x_dict = featureDict.copy()
                stem = stemmer.stem(word)
                index = word.find(stem)
                if index != -1:
                    x_dict[word[:index]] = 1
                    x_dict[word[index + len(stem):]] = 1
                x = np.array(list(x_dict.values()))
                wordDict[word] = x
                X_test.append(x)
                Y_test.append(tag)
            else:
                x = wordDict[word]
                X_test.append(x)
                Y_test.append(tag)

    print("testing data size: " + str(len(X_test)) + " " + str(len(Y_test)))
    return (np.array(X_train), np.array(Y_train)), \
           (np.array(X_test), np.array(Y_test))


if __name__ == '__main__':
    nltk.download('brown')  # Brown corpus
    nltk.download('universal_tagset')  # tag set

    corpusData = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'),
                          dtype=object)
    random.shuffle(corpusData)

    cv = KFold(n_splits=5)
    accuracies = list()
    for train_index, test_index in cv.split(corpusData):
        train_data, test_data = createDataSet(corpusData[train_index],
                                              corpusData[test_index])
        clf = svm.LinearSVC()
        clf.fit(train_data[0], train_data[1])
        accuracy = clf.score(test_data[0], test_data[1])
        accuracies.append(accuracy)

    print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)
