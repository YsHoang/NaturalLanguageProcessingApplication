import csv
import pandas as pd
import os
import re
MappingFile = "./zQC_Spec/z_SwT_Data/LabelMapping.csv"
fromFile = "./zQC_Spec/z_SwT_Data/TrainingData.csv"
DataFile    = "./zQC_Spec/z_SwT_Data/TrainingData1_2.csv"
#TestFile    = "./zQC_Spec/z_SwT_Data/test.csv"
def label2Index(mapping_file = MappingFile):
    with open(mapping_file) as csvMappingFile:
        csvReader = csv.reader(csvMappingFile)
        label2index = {row[0]: row[1] for row in csvReader}
        index2label = {int(index):label for label, index in label2index.items()}
    return label2index, index2label

def index_data(fromfile):
    label_index, __ = label2Index(MappingFile)

    df = pd.read_csv(fromfile)

    for name, index in label_index.items():
        df.loc[df['Function Name'] == name, 'Value'] = index
    df.to_csv(DataFile, index = False)

def idex_and_label(fromFile, toFile, preds):
    __, index2label = label2Index(MappingFile)
    labels = []
    for pred in preds:
        try:
            labels.append(index2label[pred])   
        except KeyError:
            print("Which create errors")
            print(pred)
        #labels.append(index2label[pred])

    df = pd.read_csv(fromFile)
    df['Function Name'] = labels
    df['Value'] = preds
    df.to_csv(toFile, index=False)
    
if __name__ == "__main__":
    index_data(fromFile)
    '''
    labels, indexs = label2Index(MappingFile)
    for index, label in indexs.items():
        print("{} : {}".format(index, label))
    '''