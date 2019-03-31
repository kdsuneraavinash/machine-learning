import collections
import nltk
import numpy
import random
import pickle

lemmatizer = nltk.WordNetLemmatizer()


def _format_string(string):
    tokenized = nltk.tokenize.word_tokenize(string.lower())
    lemmatized = (lemmatizer.lemmatize(word) for word in tokenized)
    return lemmatized


def create_lexicon(pos, neg, high_thresh=1500, low_thresh=20):
    """
    Creates a featureset of format
        [w1, w2, w3, w4, ..., wn]
    where each word appears between 20 to 1000 occurences.
    """
    counter = collections.Counter()

    for file_name in (pos, neg):
        with open(file_name, 'r') as fr:
            for line in fr:
                formatted = _format_string(line)
                sent_counter = collections.Counter(formatted)
                counter.update(sent_counter)

    for key in list(counter.keys()):
        if not (high_thresh > counter[key] > low_thresh):
            del counter[key]

    return list(counter.keys())


def create_featureset(sample_file, lexicon, classification):
    """
    Creates a featureset of the format
    [
        [[0 1 2 1 1 0 0 0], [0 1]],
        [[0 1 0 1 1 0 1 0], [1 0]],
        ...
    ]
    where classification is either [0 1] or [1 0]
    """
    lexicon_size = len(lexicon)
    indexed_lexicon = dict(zip(lexicon, range(lexicon_size)))

    feature_set = []

    with open(sample_file, 'r') as fr:
        for line in fr:
            formatted = _format_string(line)
            arr = numpy.zeros(lexicon_size)
            for word in formatted:
                if word in indexed_lexicon:
                    arr[indexed_lexicon[word]] += 1

            feature_set.append([arr, classification])

    return feature_set


def create_train_test_sets(pos, neg, test_percentage=0.1):
    lexicon = create_lexicon(pos, neg)
    feature_set = create_featureset(pos, lexicon, [1, 0])
    feature_set += create_featureset(neg, lexicon, [0, 1])
    random.shuffle(feature_set)

    total_size = len(feature_set)
    test_size = int(test_percentage*total_size)

    test_xy = feature_set[:test_size]
    train_xy = feature_set[test_size:]

    test_x = [d[0] for d in test_xy]
    test_y = [d[1] for d in test_xy]

    train_x = [d[0] for d in train_xy]
    train_y = [d[1] for d in train_xy]

    return train_x, train_y, test_x, test_y


def next_batch(inputs, features, batch_size):
    pass
    start = 0
    length = len(inputs[0])
    start = 0
    while start<length:
        batch_inputs = inputs[start:start + batch_size]
        batch_features = features[start:start + batch_size]
        start += batch_size
        yield batch_inputs, batch_features


if __name__ == "__main__":
    train_test_data_set = create_train_test_sets('data/pos.txt', 'data/neg.txt')
    with open('save.pickle', 'wb') as fb:
        pickle.dump(train_test_data_set, fb)
