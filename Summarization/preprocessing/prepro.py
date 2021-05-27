from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from html.parser import HTMLParser
import re
import nltk
nltk.download('stopwords')

import numpy as np
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

class CleanData:
  def __init__(self):
    self.stopword = set(stopwords.words('english'))
    self.contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is","I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would","i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have","mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have","she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is","should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as","this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would","there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have","they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have","wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are","what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is","where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have","you're": "you are", "you've": "you have"}

  def Lower(self, text):
    return text.lower()

  def ContractionMapping(self, text):
    return ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

  def RemovePuctuation(self, text):
    newStr = re.sub(r'\([^)]*\)', '', text)
    newStr = re.sub('"','', newStr)
    newStr = re.sub(r"'s\b", "", newStr)
    newStr = re.sub('[^a-zA-Z]',' ', newStr)
    return newStr
  
  def RemoveStopword(self, text):
    tokens = [w for w in text.split() if not w in self.stopword]
    return ' '.join(tokens)

  def RemoveHTML(self, text):
    newStr = BeautifulSoup(text, 'lxml').text
    newStr = BeautifulSoup(newStr, 'html.parser').text
    return newStr

class Encoding:
  def __init__(self, vocab_size):
    self.token = Tokenizer(num_words = vocab_size)
    self.word_index = self.token.word_index
    self.index_word = self.token.index_word

  def FitOnText(self, text):
    self.token.fit_on_texts(text)

  def TextToSeq(self, text):
    seq = self.token.texts_to_sequences(text)
    self.word_index = self.token.word_index
    return seq
  
  @staticmethod
  def Padding(seq, maxlen, padding='post', truncating='post'):
    pad = pad_sequences(seq, maxlen=maxlen, padding=padding, truncating=truncating)
    return pad

class Vocab:
  def __init__(self, vocab_size):
    self.vocab = {}
    self.length = 0
    self.VOCAB_SIZE = vocab_size
    
  def BuildVocab(self, word_index):
    for word in word_index.keys():
      if word_index[word] == 0:
        breaks
      if word_index[word] > self.VOCAB_SIZE:
        continue
      else:
        self.vocab[word] = word_index[word]
        self.length += 1
    return self.vocab

class Embedding:
  def __init__(self):
    self.embeddings_index = {}
    self.embedding_matrix = None

  def LoadEmbeddingModel(self, path_emb):
    with open(path_emb, encoding='utf-8') as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        self.embeddings_index[word] = coefs
      f.close()
    print("Found %s word vector." % len(self.embeddings_index))
    return self.embeddings_index

  def EmbeddingMatrixCreater(self, word_index, emb_dim):
    self.embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    for word, i in word_index.items():
      embedding_vector = self.embeddings_index.get(word)
      if embedding_vector is not None:
        self.embedding_matrix[i] = embedding_vector
    return self.embedding_matrix


