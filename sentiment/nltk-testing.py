import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

hm_lines = 10000000

def create_lex(pos,neg):
    lex = []
    final_lex = []

    for filename in [pos,neg]:
        with open(filename,'r') as f:
            data = f.readlines()
            for line in data[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lex += list(all_words)

    lex = [lemmatizer.lemmatize(i) for i in lex]

    word_counts = Counter(lex)


    for word in word_counts:
        if 1000 > word_counts[word] > 10:
            final_lex.append(word)

    return final_lex

def sample_handling(sample, lex, classification):
    feature_set = []

    with open(sample,'r') as f:
        data = f.readlines()
        for line in data[:hm_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lex))

            for word in current_words:
                if word.lower() in lex:
                    word_index = lex.index(word.lower())
                    features[word_index] += 1

            features = list(features)
            feature_set.append([features,classification])

    return feature_set

def create_feature_sets_and_labels(pos,neg,test_size=.1):
    lex = create_lex(pos,neg)
    features = []
    features += sample_handling(pos,lex,[1,0])
    features += sample_handling(neg,lex,[0,1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = test_size * len(features)

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    print("initializing")
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels("positive_data.txt","negitive_data.txt")
    with open("senti.pickle","wb") as f:
        pickle.dump([train_x, train_y, test_x, test_y],f)
