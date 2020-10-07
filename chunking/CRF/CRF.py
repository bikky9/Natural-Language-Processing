import nltk
from tqdm import tqdm
from sklearn.metrics import confusion_matrix , classification_report
import pandas as pd
import pycrfsuite

TAGS = ['B','I','O']

# def features(**args):
#         return args

class ChunkTagger():
    def __init__(self,TRAIN_DATA):
        self.classifier = pycrfsuite.Trainer()
        for sent in TRAIN_DATA:
            X , Y = [] , []
            sentLen = len(sent)
            prevtag = None
            for i , (word,pos,tag) in enumerate(sent):
                features = [
                    'word.lower='+word.lower(),
                    'wordLast3='+word[-3:],
                    'wordLast2='+word[-2:],
                    'postag='+pos,
                ]
                if(i == 0): features.append('<START>')
                else:
                    prevword , prevpos = sent[i-1][0] , sent[i-1][1]
                    features.extend([
                        "prevwordlower="+prevword.lower(),
                        "prevpos="+prevpos,
                        "prevcurrpos="+f'{prevpos}+{pos}',
                    ])

                if(i == (sentLen-1)): features.append("<END>")
                else:
                    nextword , nextpos = sent[i+1][0] , sent[i+1][1]
                    features.extend([
                        "nextwordlower="+nextword.lower(),
                        "nextpos="+nextpos,
                        "currnextpos"+f'{pos}+{nextpos}'
                    ])
                
                X.append(features); Y.append(tag)
                prevtag = tag
            self.classifier.append(X,Y)
        print("Started  Training")
        self.classifier.set_params({
            'c1': 0.1,
            'c2': 0.01,  
            'max_iterations': 200,
            'feature.possible_transitions': True
        })
        self.classifier.train("csrf.model")
        print("Classifier Trained")

        self.tagger = pycrfsuite.Tagger()
        self.tagger.open('csrf.model')

    
    def tag(self,sent):
        X = [] 
        sentLen = len(sent)
        prevtag = None
        for i , (word,pos,tag) in enumerate(sent):
            features = [
                'word.lower='+word.lower(),
                'wordLast3='+word[-3:],
                'wordLast2='+word[-2:],
                'postag='+pos,
            ]
            if(i == 0): features.append('<START>')
            else:
                prevword , prevpos = sent[i-1][0] , sent[i-1][1]
                features.extend([
                    "prevwordlower="+prevword.lower(),
                    "prevpos="+prevpos,
                    "prevcurrpos="+f'{prevpos}+{pos}',
                ])

            if(i == (sentLen-1)): features.append("<END>")
            else:
                nextword , nextpos = sent[i+1][0] , sent[i+1][1]
                features.extend([
                    "nextwordlower="+nextword.lower(),
                    "nextpos="+nextpos,
                    "currnextpos"+f'{pos}+{nextpos}'
                ])
            
            X.append(features)
        Y = self.tagger.tag(X)
        return Y
        
    
    def evaluate(self,TEST_DATA):
        preds = [self.tag(sent) for sent in TEST_DATA]
        y_test , y_pred = [] , []
        corr , total = 0, 0
        for i,s in enumerate(TEST_DATA):
            total += len(s)
            for (w,pos,t1) , t2 in zip(s,preds[i]):
                corr += (t1 == t2)
                y_test.append(t1[0]); y_pred.append(t2)
        print(classification_report(y_test , y_pred , target_names=TAGS))
        confusionMatrix = confusion_matrix(y_test, y_pred, labels=TAGS)
        
        for i in range(confusionMatrix.shape[0]):
            print(TAGS[i], " accuracy: ", (confusionMatrix[i][i] / sum(confusionMatrix[i])) * 100)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        df = pd.DataFrame(confusionMatrix)
        df.columns = TAGS
        df.index = TAGS
        df.style
        print(df)
        print("Accuracy :" , 100 * corr / total )
    


if __name__ == "__main__":

    nltk.download("conll2000")
    corpus_train = nltk.corpus.conll2000.chunked_sents('train.txt')
    corpus_test = nltk.corpus.conll2000.chunked_sents("test.txt")

    def preprocess(sent):
        tgs = nltk.tree2conlltags(sent)
        tgs = [(w,pos,t[0]) for w,pos,t in tgs]
        return tgs
    TRAIN_DATA = [preprocess(sent) for sent in corpus_train]
    TEST_DATA = [preprocess(sent) for sent in corpus_test]

    cP = ChunkTagger(TRAIN_DATA)
    cP.evaluate(TEST_DATA)
    
    
    # TRAIN_DATA_FILE = open("../assignment2dataset/train.txt","r")
    # TEST_DATA_FILE = open("../assignment2dataset/test.txt" , 'r')

    # TRAIN_DATA , TEST_DATA = [] , []
    # curr = []
    # for line in TRAIN_DATA_FILE:
    #     line = line.strip()
    #     if not line:
    #         TRAIN_DATA.append(curr)
    #         curr = []
    #     else:
    #         word , tag , boi = line.split()
    #         if boi != "O":
    #             boi = boi[0]
    #         curr.append((word,tag,boi))

    # for line in TEST_DATA_FILE:
    #     line = line.strip()
    #     if not line:
    #         TEST_DATA.append(curr)
    #         curr = []
    #     else:
    #         word , tag , boi = line.split()
    #         if boi != "O":
    #             boi = boi[0]
    #         curr.append((word,tag,boi))
    # print(TRAIN_DATA[0])
    # print(len(TRAIN_DATA) , len(TEST_DATA))

