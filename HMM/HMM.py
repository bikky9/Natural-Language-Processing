import nltk
import random
import numpy as np
from collections import Counter

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

TAGS = list(alltags)
WORDS = list(allwords)
TAGS_COUNT = len(TAGS)
WORDS_COUNT = len(WORDS)

print(TAGS_COUNT,WORDS_COUNT)

TAG_OCCURENCES = Counter()
for word,tag in tagged_train_words:
    TAG_OCCURENCES[tag] += 1

def computeEmissionProb(word,tag):
    tagOccurencesCount = TAG_OCCURENCES[tag]
    wordOccurancesCount = sum([(word == _word and tag == _tag) for _word,_tag in tagged_train_words])

    return (wordOccurancesCount/tagOccurencesCount)

def computeTransitionProb(tag2,tag1):
    tags = [_tag for _word,_tag in tagged_train_words]
    tag1Ocuurences = TAG_OCCURENCES[tag1]
    transitions = 0
    for i in range(len(tags)-1):
        if tags[i] == tag1 and tags[i+1] == tag2:
            transitions += 1
    return transitions/tag1Ocuurences

transition_matrix = np.zeros((TAGS_COUNT,TAGS_COUNT) , dtype= 'float32')
for i in range(TAGS_COUNT):
    for j in range(TAGS_COUNT):
        transition_matrix[i,j] = computeTransitionProb(TAGS[i],TAGS[j])


def viterbi(words):
    tags_observed = []
    for word in words:
        probabilites = []
        prev_tag_index = TAGS.index(tags_observed[-1]) if tags_observed else TAGS.index('.')
        
        for i in range(TAGS_COUNT):
            transitionProb = transition_matrix[i,prev_tag_index]
            emissionProb = computeEmissionProb(word,TAGS[i])

            probabilites.append(emissionProb*transitionProb)
        
        newtag_index = probabilites.index(max(probabilites))
        tags_observed.append(TAGS[newtag_index])
    return list(zip(words,tags_observed))
