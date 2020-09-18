import numpy as np
import nltk
import random
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from statistics import mean
from sklearn import svm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd


class SVM():
    def __init__(self, corpusData):
        self.WORDS = None
        self.TAGS = None
        self.WORDS_SIZE = None
        self.corpusData = corpusData
        self.X_TRAIN = None
        self.Y_TRAIN = None
        self.SUF = None
        self.PRE = None
        self.suf_len = None
        self.pre_len = None

    def encode_tag(self, tag):
        l = [0] * (len(self.TAGS) + 1)
        l[self.TAGS.index(tag) + 1] = 1
        return l

    def encode_word(self, word):
        l = [0] * (self.WORDS_SIZE + 2)
        try:
            l[self.WORDS.index(word)] = 1
        except ValueError:
            pass
        l[-2] = int(word.islower())
        l[-1] = int(word.isalpha())
        return l

    def suf_pre(self, word):
        suf = [0] * (self.suf_len)
        pre = [0] * self.pre_len
        if word not in self.WORDS:
            for i in range(1, min(len(word), 4)):
                s = word[-i:]
                p = word[i:]
                try:
                    suf[self.SUF.index(s)] = 1
                    pre[self.PRE.index(p)] = 1
                except Exception:
                    pass
        return suf + pre

    def createDataSet(self, training_data):
        self.WORDS = Counter()
        self.TAGS = set()
        self.SUF = Counter()
        self.PRE = Counter()
        for sentence in training_data:
            for word, tag in sentence:
                self.TAGS.add(tag)
                self.WORDS[word] += 1
                for i in range(1, min(len(word), 4)):
                    self.SUF[word[-i:]] += 1
                    self.PRE[word[:i]] += 1
        THRESHOLD = 180
        self.SUF = [word for word, v in self.SUF.items() if v > 10 * THRESHOLD]
        self.PRE = [word for word, v in self.PRE.items() if v > 10 * THRESHOLD]
        self.suf_len = len(self.SUF)
        self.pre_len = len(self.PRE)
        print(self.suf_len, self.pre_len)

        self.TAGS = list(self.TAGS)
        self.WORDS = [word for word, v in self.WORDS.items() if v > THRESHOLD]
        self.WORDS_SIZE = len(self.WORDS)

        self.X_TRAIN = []
        self.Y_TRAIN = []

        for sentence in tqdm(training_data):
            curr = self.encode_word(sentence[0][0])
            prev_tag = [0] * (len(self.TAGS) + 1)
            for i in range(len(sentence) - 1):
                nex = self.encode_word(sentence[i + 1][0])
                self.X_TRAIN.append(curr + self.suf_pre(sentence[i][0]) + prev_tag)
                self.Y_TRAIN.append(self.TAGS.index(sentence[i][1]))
                prev_tag = self.encode_tag(sentence[i][1])
                curr = nex
            self.X_TRAIN.append(curr + self.suf_pre(sentence[-1][0]) + prev_tag)
            self.Y_TRAIN.append(self.TAGS.index(sentence[-1][1]))
        return

    def evaluate(self):
        random.shuffle(self.corpusData)
        corpusData = self.corpusData
        cv = KFold(n_splits=5)
        accuracies = list()
        for train_index, test_index in cv.split(corpusData):
            self.createDataSet(corpusData[train_index])
            print("Fitting")
            clf = svm.LinearSVC()
            clf.fit(self.X_TRAIN, self.Y_TRAIN)
            print("Training done")
            Y_test, Y_star = [], []
            for sentence in corpusData[test_index]:
                curr = self.encode_word(sentence[0][0])
                prev_tag = [0] * (len(self.TAGS) + 1)
                for i in range(len(sentence) - 1):
                    nex = self.encode_word(sentence[i + 1][0])
                    x = curr + self.suf_pre(sentence[i][0]) + prev_tag
                    prediction = clf.predict([x])
                    Y_test.append(self.TAGS.index(sentence[i][1]))
                    Y_star.append(prediction[0])
                    prev_tag = [0] * (len(self.TAGS) + 1)
                    prev_tag[prediction[0] + 1] = 1
                    curr = nex
                x = curr + self.suf_pre(sentence[-1][0]) + prev_tag
                prediction = clf.predict([x])

                Y_test.append(self.TAGS.index(sentence[-1][1]))
                Y_star.append(prediction[0])
            accuracy = sum([(a == b) for a, b in zip(Y_test, Y_star)]) / len(Y_test)
            accuracies.append(accuracy)
            # (un)comment lines till 136, not to display confusion matrix and per POS accuracy
            confusionMatrix = confusion_matrix(Y_test, Y_star, labels=range(12))
            for i in range(confusionMatrix.shape[0]):
                print(self.TAGS[i], " accuracy: ", (confusionMatrix[i][i] / sum(confusionMatrix[i])) * 100)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            df = pd.DataFrame(confusionMatrix)
            df.columns = self.TAGS
            df.index = self.TAGS
            df.style
            print(df)
            break
        print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)


if __name__ == '__main__':
    nltk.download('brown')  # Brown corpus
    nltk.download('universal_tagset')  # tag set

    corpusData = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'))

    svmObj = SVM(corpusData)
    svmObj.evaluate()