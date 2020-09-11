import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import random
from sklearn.model_selection import KFold
from statistics import mean
from sklearn import svm


def createDataSet(training_data, testing_data):
    stemmer = SnowballStemmer("english")
    SUFFIX = set()
    TAGS = set()
    for sentence in training_data:
        for word, tag in sentence:
            TAGS.add(tag)
            stem = stemmer.stem(word)
            index = word.find(stem)
            if index != -1:
                SUFFIX.add(word[index + len(stem):])
    for sentence in testing_data:
        for word, tag in sentence:
            TAGS.add(tag)
            stem = stemmer.stem(word)
            index = word.find(stem)
            if index != -1:
                SUFFIX.add(word[index + len(stem):])

    print("feature dictionary size: " + str(len(SUFFIX)))
    TAGS, SUFFIX = list(TAGS), list(SUFFIX)
    SUFFIX_SIZE = len(SUFFIX)
    WORD_FEATURES_SIZE = 1
    FEATURES_SIZE = SUFFIX_SIZE + WORD_FEATURES_SIZE
    X_train, Y_train = [], []
    for sentence in training_data:
        prev = [0] * FEATURES_SIZE
        for word, tag in sentence:
            wordFeatures = [int(word.islower())]
            stem = stemmer.stem(word)
            index = word.find(stem)
            curr = [0] * SUFFIX_SIZE  # suffix features
            if index != -1:
                st = word[index + len(stem):]
                curr[SUFFIX.index(st)] = 1
            X_train.append(np.array(prev + curr + wordFeatures))
            Y_train.append(TAGS.index(tag))
            prev = curr + wordFeatures

    print("training data size: " + str(len(X_train)) + " " + str(len(Y_train)))

    X_test = []
    Y_test = []
    for sentence in testing_data:
        prev, curr = [0] * FEATURES_SIZE, None
        for word, tag in sentence[1:]:
            wordFeatures = [int(word.islower())]
            curr = [0] * SUFFIX_SIZE
            stem = stemmer.stem(word)
            index = word.find(stem)
            if index != -1:
                st = word[index + len(stem):]
                curr[SUFFIX.index(st)] = 1
            X_test.append(np.array(prev + curr + wordFeatures))
            Y_test.append(TAGS.index(tag))
            prev = curr + wordFeatures

    print("testing data size: " + str(len(X_test)) + " " + str(len(Y_test)))
    return (np.array(X_train), np.array(Y_train)), \
           (np.array(X_test), np.array(Y_test))


if __name__ == '__main__':
    nltk.download('brown')  # Brown corpus
    nltk.download('universal_tagset')  # tag set

    corpusData = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'), dtype=object)

    random.shuffle(corpusData)
    length = len(corpusData)
    corpusData = corpusData[:length // 5]
    cv = KFold(n_splits=5)
    accuracies = list()
    for train_index, test_index in cv.split(corpusData):
        train_data, test_data = createDataSet(corpusData[train_index], corpusData[test_index])
        clf = svm.LinearSVC()
        clf.fit(train_data[0], train_data[1])
        print("Training done")
        X_test, Y_test = test_data
        accuracy = clf.score(X_test, Y_test)
        print("accuracy: " + str(accuracy))
        accuracies.append(accuracy)

    print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)
