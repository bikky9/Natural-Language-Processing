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

def computeEmissionProb(word,tag):
    words_with_tag = [_word for _word,_tag in tagged_train_words if(_tag == tag)]
    tagOccurencesCount = len(words_with_tag)
    wordOccurancesCount = sum([(word == _word) for _word in words_with_tag])

    return (wordOccurancesCount/tagOccurencesCount)

def computeTransitionProb(tag2,tag1):
    tags = [_tag for _word,_tag in tagged_train_words]
    tag1Ocuurences = sum([(_tag == tag1) for _tag in tags])
    transitions = 0
    for i in range(len(tags)-1):
        if tags[i] == tag1 and tags[i+1] == tag2:
            transitions += 1
    return transitions/tag1Ocuurences

print(TAGS_COUNT,WORDS_COUNT)