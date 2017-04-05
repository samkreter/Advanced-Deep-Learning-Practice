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


    for word in word_counts
        if 1000 > word_counts[word] > 10:
            final_lex.append(word)

    return final_lex





