import nltk
import random

nltk.download('brown') #Brown cropus
nltk.download('universal_tagset')  #tag set

corpus_data = list(nltk.corpus.brown.tagged_sents(tagset='universal'))
random.shuffle(corpus_data)
total = len(corpus_data)

train_data , test_data = corpus_data[:int(4/5*total)] , corpus_data[int(4/5*total):]

tagged_train_words = []
tagged_test_words = []

for sentence in train_data:
    for wordtuple in sentence:
        tagged_train_words.append(wordtuple)

for sentence in test_data:
    for wordtuple in sentence:
        tagged_test_words.append(wordtuple)

alltags = set()
allwords = set()
for word,tag in tagged_train_words:
    allwords.add(word)
    alltags.add(tag)

TAGS_COUNT = len(alltags)
WORDS_COUNT = len(allwords)

print(TAGS_COUNT,WORDS_COUNT)