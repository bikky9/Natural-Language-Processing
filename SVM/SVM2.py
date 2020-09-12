import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import random
from collections import defaultdict , Counter
from sklearn.model_selection import KFold
from statistics import mean
from sklearn import svm

class SVM():
    def __init__(self,corpusData):
        self.WORDS = None
        self.TAGS = None
        self.WORDS_SIZE = None
        self.corpusData = corpusData
    
    def encode_tag(self,tag):
        l = [0]*(len(self.TAGS)+1)
        l[self.TAGS.index(tag)+1] = 1
        return l
    def encode_word(self,word):
        l = [0]*(self.WORDS_SIZE+2)
        try:
            l[self.WORDS.index(word)] = 1
        except ValueError:
            pass
        l[-2] = int(word.islower())
        l[-1] = int(word.isalpha())
        return l

    def createDataSet(self,training_data):
        stemmer = SnowballStemmer("english")
        self.WORDS = Counter()
        self.TAGS = set()
        
        for sentence in training_data:
            for word, tag in sentence:
                self.TAGS.add(tag)
                self.WORDS[word] += 1
        
        SETS = []
        THRESHOLD =10
        self.TAGS = list(self.TAGS)
        self.WORDS = [word for word,v in self.WORDS.items() if v > THRESHOLD]
        self.WORDS_SIZE = len(self.WORDS)
        
        X_TRAIN = [] 
        Y_TRAIN = []
        for sentence in training_data:
            prev = [0]*(self.WORDS_SIZE+2)
            curr = self.encode_word(sentence[0][0])
            nex = None
            prev_tag = [0]*(len(self.TAGS)+1)
            for i in range(len(sentence)-1):
                nex = self.encode_word(sentence[i+1][0])
                X_TRAIN.append(curr + prev + nex + prev_tag)
                Y_TRAIN.append(self.TAGS.index(sentence[i][1]))
                prev_tag = self.encode_tag(sentence[i][1])
                prev = curr
                curr = nex
            X_TRAIN.append(curr + prev + [0]*(self.WORDS_SIZE+2)+prev_tag)
            Y_TRAIN.append(self.TAGS.index(sentence[-1][1]))
        return (np.array(X_TRAIN), np.array(Y_TRAIN))
        
    def evaluate(self):
        random.shuffle(self.corpusData)
        length = len(self.corpusData)
        corpusData = self.corpusData[:length//15]
        cv = KFold(n_splits=5)
        accuracies = list()
        for train_index, test_index in cv.split(corpusData):
            train_data= self.createDataSet(corpusData[train_index])
            clf = svm.LinearSVC()
            clf.fit(train_data[0], train_data[1])
            print("Training done")
            Y_test , Y_star = [] , []
            for sentence in corpusData[test_index]:
                prev = [0]*(self.WORDS_SIZE+2)
                curr = self.encode_word(sentence[0][0])
                nex = None
                prev_tag = [0]*(len(self.TAGS)+1)
                for i in range(len(sentence)-1):
                    nex = self.encode_word(sentence[i+1][0])
                    x = curr + prev + nex + prev_tag
                    prediction = clf.predict([x])
                    Y_test.append(self.TAGS.index(sentence[i][1]))
                    Y_star.append(prediction[0])
                    
                    prev_tag = [0]*(len(self.TAGS)+1)
                    prev_tag[prediction[0]+1] = 1
                    prev = curr
                    curr = nex
            x = curr + prev + [0]*(self.WORDS_SIZE+2)+prev_tag
            prediction = clf.predict([x])
            
            Y_test.append(self.TAGS.index(sentence[-1][1]))
            Y_star.append(prediction[0])
            accuracy = sum([(a == b) for a,b in zip(Y_test,Y_star)])/len(Y_test)
            print("accuracy: " + str(accuracy))
            accuracies.append(accuracy)
        print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)


if __name__ == '__main__':
    nltk.download('brown')  # Brown corpus
    nltk.download('universal_tagset')  # tag set

    corpusData = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'))
    
    svmObj = SVM(corpusData)
    svmObj.evaluate()