import numpy as np 
import pandas as pd 
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Input, Model
from keras.layers import Dense, Embedding, LSTM, Activation

def get_input(filename):
  data = pd.read_csv(filename)
  return data['sequence'].values + data['q8'].values

def get_output(filename):
  output = []
  file = np.load(filename)
  for key in file:
    output.append(file[key])
  return output

def get_ngram_text(seqs, n=8):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs])

if __name__ == '__main__':

  # Load data
  train_input = get_input('train_input.csv')
  train_output = get_output('train_output.npz')
  test_input = get_input('test_input.csv')

  # Define a maximum length (we can probably catch the structure with 128 length sequences)
  maxlen_seq = 128

  # Create 8-gram structures for amino acids
  input_grams = get_ngram_text(train_input)

  # Tokenize our input and pad accordingly
  tokenizer_encoder = Tokenizer()
  tokenizer_encoder.fit_on_texts(input_grams)
  input_data = tokenizer_encoder.texts_to_sequences(input_grams)
  input_data = sequence.pad_sequences(input_data, maxlen=maxlen_seq, padding='post')

  # Fetch matrices and squash them to average 
  output_data = []
  for output in train_output:
      output_data.append(np.average(output))

  output_data = np.array(output_data)

  # Get the vocab size for the Embedding layer
  n_words = len(tokenizer_encoder.word_index) + 1

  # Functional Keras Model (to allow for multiple input later)
  sequence_text = Input(shape=(None, ))
  embedding_layer = Embedding(input_dim=n_words, output_dim=128)(sequence_text)
  x = LSTM(128)(embedding_layer)
  x = Activation('relu')(x)
  y = Dense(1, activation='relu')(x)

  model = Model(sequence_text, y)

  # Use Adam optimizer and MSE
  model.compile(optimizer='adam', loss='mean_squared_error')

  model.fit(input_data, output_data,
          epochs=20, validation_split=0.2, verbose=2)

  # Tokenize Test data
  test_input_grams = get_ngram_text(test_input)
  output_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
  output_data = sequence.pad_sequences(output_data, maxlen=maxlen_seq, padding='post')

  # Get predictions
  predictions = model.predict(output_data)

  # Create matrices
  final_matrix = []
  test_input = pd.read_csv('test_input.csv')

  for index,row in test_input.iterrows():
      # For each test example, create a temp_matrix having all values same as predicted value
      temp_matrix = np.full((row['length'],row['length']),predictions[index][0])
      
      # Set the diagonal values to 0
      np.fill_diagonal(temp_matrix, 0)
      
      # Append to final_matrix
      final_matrix.append(temp_matrix)

  # Save matrices
  np.savez('test_%d.npz'%5,*final_matrix)
