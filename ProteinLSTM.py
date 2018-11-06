from keras.models import Model, Input
from keras.layers import Dense, TimeDistributed, Embedding, LSTM, Bidirectional

class ProteinLSTM:
  ## Need to figure out how to work this model properly
  @staticmethod
  def get_model(n_words, n_tags, output_dim, input_length):
    input = Input(shape=(output_dim,))
    x = Embedding(input_dim=n_words, 
                  output_dim=output_dim,
                  input_length=input_length)(input)
    x = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           recurrent_dropout=0.1))(x)
    y = TimeDistributed(Dense(n_tags, activation='softmax'))(x)
    return Model(input, y)
  
  @staticmethod
  def compile(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
  
  @staticmethod
  def fit(model, train_input, train_output, validation_train, validation_test, batch_size=50, epochs=5):
    model.fit(train_input, 
              train_output, 
              batch_size=50, 
              epochs=5,
              validation_data=(validation_train, validation_test),
              verbose=1)
