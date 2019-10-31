import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import load_model
import json
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#from keras.initializers import glorot_uniform
import os
import re

Y_LABELS = ['PARAMETER', 'NETLC', 'INVALID'] 

def TrainingModel():
    
    paths = {"para":"/Signals/Sw_Parameters/", "netlc":"/Signals/LC_NetSignals/", "invalid":"/Signals/Invalid_Words/"}    
    y_values = {"para":0, "netlc":1, "invalid":2}

    X_train, X_test, Y_train, Y_test, vob_size, maxLen =  build_training_data(paths, y_values, info = False)
    
    model = Characters_LSTM((maxLen, vob_size),d_output = len(y_values))
    
    model.summary()
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    model.fit(X_train, Y_train, epochs = 150, batch_size = 32, shuffle=True)
    
    
    __, acc = model.evaluate(X_test, Y_test)
    
    print("Test accuracy = ", acc)

    model.save("./Models/SignalModel.h5")
    
    return X_test, Y_test
    

def SignalsClassifier(Signals):
    
    Y_PREDICT_LABELS = []

    with open('./Models/CharToIx.json', 'r') as fp:
        CharToIx = json.load(fp)
    
    X_indices = words_to_indices(Signals, CharToIx, CharToIx['MaxLen'])
    
    InputX = convert_to_matrix(X_indices, CharToIx['VocSize'])
    
    modell = load_model("./Models/SignalModel.h5")
    if len(Signals):
        predictions = modell.predict(InputX)
        y_predict = np.argmax(predictions, axis = 1).tolist()
        
        for idx in y_predict:
            Y_PREDICT_LABELS.append(Y_LABELS[idx])    
    
    return Y_PREDICT_LABELS

def Characters_LSTM(input_shape, d_output):
    """
    Function creating the characters model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    
    Returns:
    model -- a model instance in Keras
    """
    
    inputs = Input(shape = input_shape, dtype = 'float32')
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences = True)(inputs)
    
    X = Dropout(0.4)(X)
    
    X = LSTM(128, return_sequences = True)(X)
    
    X = Dropout(0.4)(X)

    X = LSTM(64, return_sequences = True)(X)
    
    X = Dropout(0.4)(X)
    
    X = LSTM(64, return_sequences = False)(X)
    
    X = Dropout(0.4)(X)

    '''
    X = LSTM(64, return_sequences = False)(X)
    X = Dropout(0.5)(X)
    

    X = Dense(64, activation = "relu")(X) #Just try it
 
    X = Dropout(0.5)(X) #Just try it
    
    '''    
    X = Dense(d_output, activation = "softmax")(X)
    
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = inputs, outputs = X)
    
    return model   

def load_data(relative_path):
    
    data = ""
    filenames = os.listdir(os.getcwd() + relative_path)
    for filename in filenames:
        if filename in ('.DS_Store'):
            pass
        else:
            print("Handling file {}".format(filename))
            f = open(os.getcwd() + relative_path + filename, 'r')
            data += f.read()
            f.close()
     
    data+= "" #Keep one space for padding
    data = data.lower()
    chars = list(set(data))
    
    #data = re.split('\n', data)
    data = data.splitlines()
    
    data = [x.strip() for x in data]
    
    data = list(set(data)) # Remove duplicates
    
    data = list(filter(None, data)) # Remove empty elements

    return data, chars

def read_dataset_from(paths):
    
    # Paths of signals: Dictionary
    
    signals = {}
    chars = {}
    sizes = {}
    all_chars = []
    all_signals = []
    
    for label in paths: 
     print('Loading data from %s............' %(label))
     signals[label], chars[label] = load_data(paths[label])
     all_signals += signals[label]
     all_chars+= chars[label]
     sizes[label] = len(signals[label])
     print("Leng of {} signals are {}".format(label,sizes[label]))
     
    return all_signals, all_chars, sizes

def build_training_data(paths, y_values, info):
    
    all_signals, all_chars, sizes = read_dataset_from(paths)

    maxLen = len(list(max(all_signals, key = len)))
    
    all_signals = np.array(all_signals) # Convert list to numpy array
    
    all_chars = list((set(all_chars)))
    all_chars.append(" ")
    
    m = len(all_signals) # Training examples
    print("Training examples is {}".format(m))
    
    Y_signals = np.zeros(m, dtype = 'int32')
    
    start_index = 0
    end_index = 0
    for label in y_values:
      end_index = start_index + sizes[label] 
      #print("start index: end_index {} {}".format(start_index, end_index))
      Y_signals[start_index:end_index] = y_values[label]
      start_index = end_index
  
    char_to_ix = { ch:i for i,ch in enumerate(sorted(all_chars)) }
    #ix_to_char = { i:ch for i,ch in enumerate(sorted(all_chars)) }
    
    vob_size = len(char_to_ix)
    
    char_to_ix['MaxLen'] = maxLen #Save max leng and char to idx
    char_to_ix['VocSize'] = vob_size
    
    with open('./Models/CharToIx.json', 'w') as fp:
        json.dump(char_to_ix, fp)
        
    X_indices = words_to_indices(all_signals, char_to_ix, maxLen)
     
    if info: 
     print("vob length is {}".format(vob_size)) 
     print(char_to_ix)
     print("Max length is {}".format(maxLen))
    
    X_onehot, Y_onehot = convert_to_one_hot(X_indices, Y_signals, len(all_chars), len(y_values))  
    
    X_train, X_test, Y_train, Y_test =  split_training_data(X_onehot, Y_onehot)
    
    return X_train, X_test, Y_train, Y_test, vob_size, maxLen


def print_training_data(X, Y):
    for i in range(len(X)):
        print(X[i], Y[i])

def words_to_indices(Signals, char_to_ix, max_len):
    
    m = len(Signals)                                  # number of training examples
    
    X_indices = np.zeros((m, max_len), dtype = 'int32')
    
    X_indices[:,:] = char_to_ix[" "]
    
    for i in range(m):                               # loop over training examples
        
        characters = list(Signals[i].lower())
        
        j = 0
        
        for c in characters:
            try:
                X_indices[i, j] = int(char_to_ix[c])
                j = j + 1               
            except KeyError:
                print("Character {} is not available in dictionary".format(c))
            except IndexError:
                print("Index {} is out of bound".format(j))
                break
    return X_indices

def convert_to_one_hot(X, Y, Cx, Cy):
    
    m = X.shape[0]
    Tx = X.shape[1]
    
    Y_matrix = np.eye(Cy)[Y.reshape(-1)]
    X_matrix = np.zeros((m, Tx, Cx))
    
    onehot = np.eye(Cx)
    
    for i in range(m):
        for j in range(Tx):
            X_matrix[i,j] = onehot[X[i,j]]
        
    return X_matrix, Y_matrix 

def convert_to_matrix(X, C):
    
    m = X.shape[0]
    Tx = X.shape[1]
    
    X_matrix = np.zeros((m, Tx, C))
    
    onehot = np.eye(C)
    
    for i in range(m):
        for j in range(Tx):
            X_matrix[i,j] = onehot[X[i,j]]
        
    return X_matrix

def split_training_data(X,Y):
    
    indices = np.random.permutation(X.shape[0])
    
    training_idx, test_idx = indices[:X.shape[0] - 100], indices[-100:]
    
    X_train, X_test, Y_train, Y_test = X[training_idx], X[test_idx], Y[training_idx], Y[test_idx]
    return X_train, X_test, Y_train, Y_test 
    
def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
