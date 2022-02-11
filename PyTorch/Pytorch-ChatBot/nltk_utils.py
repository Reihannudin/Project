# import library
import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# membuat fungsi tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# membuat fungsi stemmed
def stem(word):
    return stemmer.stem(word.lower())

# membuat fungsi bag of word
def bag_of_words(tokenized_sentece, all_words):
    '''
    sentence = ["hello","how","are","you"]
    words = ["hi","hello","i","you","bye","thank","you","cool"]
    bag = [ 1 ,1 ,0 , 1 ,0 ,0 ,1 ,0]
    '''
    tokenized_sentece = [stem(w) for w in  tokenized_sentece]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentece:
            bag[idx] = 1.0
            
    return bag
    

# test

# tokenize
# kalimat asli
# a = "How long does shipping take?"
# print(a)
# kalimat setelah di tokenize
# a = tokenize(a)
# print(a)

# stemmed
# words = ["University", "universal", "universe"]
# stemmed_word = [stem(w) for w in words]
# print(stemmed_word)
# 

# bag of words
# sentence = ["hello","how","are","you"]
# words = ["hi","hello","i","you","bye","thank","you","cool"]
# bag = bag_of_words(sentence, words)
# print(bag)