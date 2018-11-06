import numpy as np
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer

# this is where we'll tokenize our data
class DataProcessor:
  @staticmethod
  def get_ngram_text(seqs, n=8):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs])

  @staticmethod
  def tokenize(n_gram_text):
    # returns both the number of words, the tokenized input data, and the tokenizer
    encoder = Tokenizer()
    encoder.fit_on_texts(n_gram_text)
    input_data = encoder.texts_to_sequences(n_gram_text)
    return len(encoder.word_index) + 1, sequence.pad_sequences(input_data, padding='post'), encoder
