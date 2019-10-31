import numpy as np
from utils.utils import convert_to_one_hot, read_glove_vecs, read_csv
from utils.assistant import sentence_level_model, sentences_to_indices
from utils.func2Index import label2Index, idex_and_label
from CollectingData import get_spec_list
from keras.models import load_model
import os
#git push -f origin <branch>
# Inputs are excel files.

maxLen = 36  # Should be load from model.
preds = []
test_number = 1
__, index2label = label2Index()
myspecs = get_spec_list('/zQC_Spec/DTC.xlsx')

cwd = os.getcwd()  # Get the current working directory (cwd)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(cwd + '/data/glove_6B_50d.txt', "./Data/NewWords.txt")

model = load_model("./Models/FunctionModel.h5")

for spec in myspecs:
    X = np.array(spec.preTextList + spec.stiTestList)
    X_indices = sentences_to_indices(X, word_to_index, maxLen)
    pred = model.predict(X_indices)
    preds = np.argmax(pred, axis=1).tolist()
    print("This is for test case ", test_number)
    print("------------------------------------")
    print("Descriptions:")
    print(spec.preTrainingData)
    print(spec.stiTrainingData)
    for pred in preds:
        print(index2label[pred])
    print("------------------------------------")

    test_number+=1



