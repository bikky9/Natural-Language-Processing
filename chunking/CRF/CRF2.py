import nltk


def featureFunction(sent, i):
    sentLen = len(sent)
    (word, pos) = sent[i]
    features = [
        'word.lower=' + word.lower(),
        'wordLast3=' + word[-3:],
        'wordLast2=' + word[-2:],
        'postag=' + pos,
    ]
    if i == 0:
        features.append('<START>')
    else:
        prevword, prevpos = sent[i - 1][0], sent[i - 1][1]
        features.extend([
            "prevwordlower=" + prevword.lower(),
            "prevpos=" + prevpos,
            "prevcurrpos=" + f'{prevpos}+{pos}',
        ])

    if i == (sentLen - 1):
        features.append("<END>")
    else:
        nextword, nextpos = sent[i + 1][0], sent[i + 1][1]
        features.extend([
            "nextwordlower=" + nextword.lower(),
            "nextpos=" + nextpos,
            "currnextpos" + f'{pos}+{nextpos}'
        ])

    return features


if __name__ == "__main__":
    nltk.download("conll2000")
    corpus_train = nltk.corpus.conll2000.chunked_sents('train.txt')
    corpus_test = nltk.corpus.conll2000.chunked_sents("test.txt")


    def preprocess(sent):
        tgs = nltk.tree2conlltags(sent)
        tgs = [((w, pos), t[0]) for w, pos, t in tgs]
        return tgs


    def preprocesstest(sent):
        tgs = nltk.tree2conlltags(sent)
        tgs = [(w, pos) for w, pos, t in tgs]
        return tgs


    TRAIN_DATA = [preprocess(sent) for sent in corpus_train]
    # TEST_DATA = [preprocess(sent) for sent in corpus_test]

    ct = nltk.tag.CRFTagger(feature_func=featureFunction)
    ct.train(TRAIN_DATA, 'model.crf.tagger')

    TEST_DATA = [preprocesstest(sent) for sent in corpus_test]
    TEST_DATA_tagged = [preprocess(sent) for sent in corpus_test]
    print(ct.evaluate(TEST_DATA_tagged))
