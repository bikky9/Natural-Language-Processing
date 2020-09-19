import numpy as np
import os
import nltk

from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding , Input , LSTM , Bidirectional , Dense , TimeDistributed
from keras import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pandas as pd


SEQ_LENGTH = 200
EMBED_DIMENSIONS = 100

nltk.download('brown')  # Brown corpus
nltk.download('universal_tagset')  # tag set
corpusData = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'),dtype=object)

WORDS = set()
TAGS = set()

for sentence in corpusData:
  for word,tag in sentence:
    WORDS.add(word)
    TAGS.add(tag)

WORDS = list(WORDS)
TAGS = list(TAGS)

wordembed = {}
int2word = {}

for i,word in enumerate(WORDS):
  wordembed[word] = i+1
  int2word[i+1] = word

X , Y = [] , []
for sentence in corpusData:
  p, q = [],[]
  for word,tag in sentence:
    p.append(wordembed[word])
    q.append(TAGS.index(tag)+1)
  X.append(p)
  Y.append(q)

X = np.asarray(X)
Y = np.asarray(Y)


embeddings = {}
with open('./glove/glove.6B.100d.txt', encoding="utf8") as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs


X = pad_sequences(X,maxlen=SEQ_LENGTH,padding='post')
Y = pad_sequences(Y,maxlen=SEQ_LENGTH,padding='post')

X,Y = shuffle(X,Y)
Y = to_categorical(Y , num_classes=len(TAGS)+1)

weightsMatrix = np.random.random((len(WORDS)+1,EMBED_DIMENSIONS))

for word , index in int2word.items():
  if word in embeddings:
    weightsMatrix[index] = embeddings[word]


cv = KFold(n_splits=5)
accuracies = []
model = None
from statistics import mean
for train_index, test_index in cv.split(X):
  model = Sequential([
                      Input(shape=(SEQ_LENGTH,)),
                      Embedding(len(WORDS)+1 , EMBED_DIMENSIONS , weights=[weightsMatrix] , input_length=SEQ_LENGTH , trainable=True),
                      Bidirectional(LSTM(64,return_sequences=True)),
                      TimeDistributed(Dense(len(TAGS)+1 , activation='softmax')),
                      ])
  model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
  model.summary()
  model.fit(X[train_index],Y[train_index])
  preds = model.predict(X[test_index], verbose=0)
  success = 0
  total = 0
  y = Y[test_index]
  Y_pred = []
  Y_test = []
  for i,sent in enumerate(y):
    start = sent[-1]
    for j,word in enumerate(sent):
      if np.array_equal(word,start):
        break
      Y_pred.append(np.argmax(preds[i][j]))
      Y_test.append(np.argmax(word))

  accuracy = sum([(a == b) for a, b in zip(Y_test, Y_pred)]) / len(Y_test)
  accuracies.append(accuracy)
  # (un)comment lines till 154, not to display confusion matrix and per POS accuracy
  confusionMatrix = confusion_matrix(Y_test, Y_pred, labels=range(1,13))
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
print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)