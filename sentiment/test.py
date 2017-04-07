import pickle

with open("senti.pickle","rb") as fi:
    train_x, train_y, test_x, test_y = pickle.load(fi)