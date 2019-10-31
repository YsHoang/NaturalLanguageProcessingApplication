import numpy as np
from utils.utils import convert_to_one_hot, read_glove_vecs, read_csv
from utils.assistant import sentence_level_model, sentences_to_indices
from utils.func2Index import label2Index
import matplotlib.pyplot as plt
from keras.models import load_model
import os

N_test = 200
N_classes = 71
np.random.seed(22)
__, index2label = label2Index()
X, Y = read_csv('./zQC_Spec/z_SwT_Data/TrainingData.csv')
m = X.shape[0]
indices = np.random.choice(m, m)

X_train, Y_train    = X[indices[:m-N_test]], Y[indices[:m-N_test]]
X_test, Y_test = X[indices[-N_test:]], Y[indices[-N_test:]]

maxLen = max(len(max(X_train, key=len).split()), 30)
Y_oh_train = convert_to_one_hot(Y_train, C = N_classes)
Y_oh_test = convert_to_one_hot(Y_test, C = N_classes)

cwd = os.getcwd()  # Get the current working directory (cwd)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(cwd + '/data/glove_6B_50d.txt', "./Data/NewWords.txt")

model = sentence_level_model((maxLen,), word_to_vec_map, word_to_index, C = N_classes)

#Model descriptions
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

Y_train_oh = convert_to_one_hot(Y_train, C = N_classes)

model.fit(X_train_indices, Y_train_oh, epochs = 125, batch_size = 32, shuffle=True)
model.save("./Models/FunctionModel.h5")

model = load_model("./Models/FunctionModel.h5")
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = N_classes)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("Test accuracy = ", acc)

# This code allows you to see the mislabelled examples
y_test_oh = np.eye(N_classes)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)

for i in range(len(X_test)):
    num = np.argmax(pred[i])
    if (num != Y_test[i]):
        print('Expected function:'+ index2label[Y_test[i]] + ":" + X_test[i] + ": " + index2label[num].strip())
