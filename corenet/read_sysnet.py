import pickle

with open('corenet/datacomp_1_2B_vocab.pkl', 'rb') as file:
    data = pickle.load(file)


print(data)