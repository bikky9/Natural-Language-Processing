import nltk
from tqdm import tqdm

def features(**args):
        return args

class ChunkTagger():
    def __init__(self,TRAIN_DATA):
        train_set = []
        for sent in tqdm(TRAIN_DATA):
            sentLen = len(sent)
            prevword , prevpos = "<NONE>" , "<NONE>"
            for i , (word,pos,tag) in enumerate(sent):
                if(i == (sentLen-1)): nextword, nextpos = "<NULL>" , "<NULL>"
                else: nextword, nextpos = sent[i+1][0] , sent[i+1][1]
                feat = features(pos=pos,word=word,prevpos=prevpos,nextpos=nextpos)
                train_set.append((feat,tag))
        print("Started MaxEnt Class Training")
        self.classifier = nltk.MaxentClassifier.train(train_set,trace=4)
        print("Classifier Trained")
    
    def tag(self,sent):
        tags = []
        sentLen = len(sent)
        prevword , prevpos = "<NONE>" , "<NONE>"
        for i , (word,pos,tag) in enumerate(sent):
            if(i == (sentLen-1)): nextword, nextpos = "<NULL>" , "<NULL>"
            else: nextword, nextpos = sent[i+1][0] , sent[i+1][1]
            feat = features(pos=pos,word=word,prevpos=prevpos,nextpos=nextpos)
            tag  = self.classifier.classify(feat)
            tags.append(tag)
        return list(zip(sent,tags))
    

class ChunkParser():
    def __init__(self,TRAIN_DATA):
        self.tagger = ChunkTagger(TRAIN_DATA)

    def evaluate(self,TEST_DATA):
        preds = [self.tagger.tag(sent) for sent in TEST_DATA]
        correct , total = 0 , 0
        for s in preds:
            total += len(s)
            for (w,pos,t1) , t2 in s:
                correct += (t1 == t2)
        print("Accuracy :" , correct/total)


if __name__ == "__main__":

    corpus_train = nltk.corpus.conll2000.chunked_sents('train.txt')
    corpus_test = nltk.corpus.conll2000.chunked_sents("test.txt")

    TRAIN_DATA = [nltk.tree2conlltags(sent) for sent in corpus_train]
    TEST_DATA = [nltk.tree2conlltags(sent) for sent in corpus_test]

    cP = ChunkParser(TRAIN_DATA)
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

