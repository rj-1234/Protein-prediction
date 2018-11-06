from IO import ProteinIO as io
import DataProcessor as dp 
import ProteinLSTM as lstm
from sklearn.model_selection import train_test_split

train_input = io.get_input('train_input.csv')
train_output = io.get_output('train_output.npz')

test_input = io.get_input('test_input.csv')
test_output = io.get_output('test_5.npz')

# Here we want to loop through the input
# and tokenize each one

X_train, x_test, y_train, y_test = train_test_split(train_input, train_output, 
                                                    test_size=0.2, random_state=0)

print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)