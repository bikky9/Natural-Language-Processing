import nltk
import tqdm
import time
import random
import numpy as np
import pandas as pd
from statistics import mean
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def Viterbi(train_data, test_data):
    tagged_train_words = []
    tagged_test_words = []

    for sentence in train_data:
        tagged_train_words.append(('^', '^'))
        for wordtuple in sentence:
            tagged_train_words.append(wordtuple)

    for sentence in test_data:
        tagged_test_words.append(('^', '^'))
        for wordtuple in sentence:
            tagged_test_words.append(wordtuple)

    alltags = set()
    allwords = set()
    for word, tag in tagged_train_words:
        allwords.add(word)
        alltags.add(tag)

    TAGS = list(alltags)
    WORDS = list(allwords)
    TAGS_COUNT = len(TAGS)

    TAG_OCCURENCES = Counter()
    for word, tag in tagged_train_words:
        TAG_OCCURENCES[tag] += 1

    wordEmissionProb = defaultdict(lambda: [0 for i in range(TAGS_COUNT)])

    def computeAllEmissionProb():
        for _word, _tag in tagged_train_words:
            wordEmissionProb[_word][TAGS.index(_tag)] += 1

        for word, v in wordEmissionProb.items():
            l = []
            for i in range(TAGS_COUNT):
                l.append(v[i] / TAG_OCCURENCES[TAGS[i]])
            wordEmissionProb[word] = l
        print("Computing Emissions Done")

    def computeTransitionProb(tag2, tag1):
        tags = [_tag for _word, _tag in tagged_train_words]
        tag1Ocuurences = TAG_OCCURENCES[tag1]
        transitions = 0
        for i in range(len(tags) - 1):
            if tags[i] == tag1 and tags[i + 1] == tag2:
                transitions += 1
        return transitions / tag1Ocuurences

    transition_matrix = np.zeros((TAGS_COUNT, TAGS_COUNT), dtype='float32')
    for i in range(TAGS_COUNT):
        for j in range(TAGS_COUNT):
            transition_matrix[i, j] = computeTransitionProb(TAGS[i], TAGS[j])

    def unknown(word):
        if word.endswith('ing') or word.endswith('ed'):
            return "VERB"
        if word.endswith('s') or not word.islower():
            return "NOUN"
        else:
            return "X"

    def viterbi(words):
        tags_observed = []
        computeAllEmissionProb()
        time.sleep(1)
        for word in tqdm.tqdm(words):
            probabilites = []
            prev_tag_index = TAGS.index(tags_observed[-1]) if tags_observed else TAGS.index('.')
            emissionProbs = wordEmissionProb[word]

            for i in range(TAGS_COUNT):
                transitionProb = transition_matrix[i, prev_tag_index]
                emissionProb = emissionProbs[i]
                probabilites.append(emissionProb * transitionProb)

            newtag_index = probabilites.index(max(probabilites))

            tags_observed.append(TAGS[newtag_index] if probabilites[newtag_index] else unknown(word))
        return list(zip(words, tags_observed))

    test_words = [word for word, tag in tagged_test_words]
    tagged_seq = viterbi(test_words)

    check = sum([(i == j) for i, j in zip(tagged_seq, tagged_test_words)])
    accuracy = check / len(tagged_seq)
    Y_pred = [tag for word, tag in tagged_seq]
    Y_test = [tag for word, tag in tagged_test_words]

    # (un)comment lines till line 116, not to display confusion matrix and per POS accuracy
    # confusionMatrix = confusion_matrix(Y_test, Y_pred, labels=TAGS)
    #
    # for i in range(confusionMatrix.shape[0]):
    #     print(TAGS[i], " accuracy: ", (confusionMatrix[i][i] / sum(confusionMatrix[i])) * 100)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    # df = pd.DataFrame(confusionMatrix)
    # df.columns = TAGS
    # df.index = TAGS
    # df.style
    # print(df)

    print('Viterbi Algorithm Accuracy: ', accuracy * 100)
    return accuracy


if __name__ == '__main__':
    nltk.download('brown')  # Brown corpus
    nltk.download('universal_tagset')  # tag set

    corpus_data = np.array(nltk.corpus.brown.tagged_sents(tagset='universal'), dtype=object)
    random.shuffle(corpus_data)

    cv = KFold(n_splits=5)
    accuracies = list()
    for train_index, test_index in cv.split(corpus_data):
        train_data, test_data = corpus_data[train_index], corpus_data[test_index]
        accuracy = Viterbi(train_data, test_data)
        accuracies.append(accuracy)
        # (un)comment this for 5-fold cross validation
        # break
    print("Mean Accuracy after 5-fold cross Validation: ", mean(accuracies) * 100)
