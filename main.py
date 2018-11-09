from IO import ProteinIO as io
from DataProcessor import DataProcessor as dp 
import ProteinLSTM as lstm
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

train_input = io.get_input('train_input.csv')
train_output = io.get_output('train_output.npz')

test_input = io.get_input('test_input.csv')
test_output = io.get_output('test_5.npz')

X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, 
                                                    test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_words, X_train['sequence'], encoder = dp.tokenize(dp.get_ngram_text(X_train['sequence'].values))
X_test['sequence'] = sequence.pad_sequences(encoder.texts_to_sequences(dp.get_ngram_text(X_test['sequence'])), padding='post')

print(X_train['sequence'].values)