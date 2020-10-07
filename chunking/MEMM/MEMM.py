import nltk
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

TAGS = ['B','I','O']

def features(**args):
        return args

class ChunkTagger():
    def __init__(self,TRAIN_DATA):
        train_set = []
        for sent in tqdm(TRAIN_DATA):
            sentLen = len(sent)
            prevword , prevpos , prevtag = "<NONE>" , "<NONE>" , "<NONE>"
            for i , (word,pos,tag) in enumerate(sent):
                if(i == (sentLen-1)): nextword, nextpos = "<NULL>" , "<NULL>"
                else: nextword, nextpos = sent[i+1][0] , sent[i+1][1]
                feat = features(pos=pos,word=word,prevpos=prevpos,nextpos=nextpos,prevtag=prevtag,prevcurrpos=f"{prevpos}+{pos}",currnextpos=f"{pos}+{nextpos}")
                train_set.append((feat,tag))
                prevword , prevpos , prevtag = word , pos , tag
        print("Started MaxEnt Class Training")
        self.classifier = nltk.MaxentClassifier.train(train_set,trace=0,algorithm="megam")
        print("Classifier Trained")
    
    def tag(self,sent):
        tags = []
        sentLen = len(sent)
        prevword , prevpos , prevtag = "<NONE>" , "<NONE>" , "<NONE>"
        for i , (word,pos,tag) in enumerate(sent):
            if(i == (sentLen-1)): nextword, nextpos = "<NULL>" , "<NULL>"
            else: nextword, nextpos = sent[i+1][0] , sent[i+1][1]
            feat = features(pos=pos,word=word,prevpos=prevpos,nextpos=nextpos,prevtag=prevtag,prevcurrpos=f"{prevpos}+{pos}",currnextpos=f"{pos}+{nextpos}")
            tag  = self.classifier.classify(feat)
            tags.append(tag)
            prevword , prevpos , prevtag = word , pos , tag
        return list(zip(sent,tags))
    
    def evaluate(self,TEST_DATA):
        preds = [self.tag(sent) for sent in TEST_DATA]
        y_test , y_pred = [] , []
        corr , total = 0, 0
        for s in preds:
            total += len(s)
            for (w,pos,t1) , t2 in s:
                corr += (t1 == t2)
                y_test.append(t1[0]); y_pred.append(t2)
        confusionMatrix = confusion_matrix(y_test, y_pred, labels=TAGS)
        # print(confusion_matrix)
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
    nltk.config_megam('./megam-64.opt')
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

