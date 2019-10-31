import csv
import os
import time
from SignalModel import TrainingModel
from SignalModel import SignalsClassifier
import re
import pandas as pd
import SymSpell
from utils.utils import read_glove_vecs
from Spec import Specification

#0019_DTC, 0027_NET, 0054_HBA, 0057_HRB, 0058_HFC, 0061_HHC, 0062_AVH, 0068_AEB, 0069_CDP
#0106_PT, 0110_SAS, 0188_YRS, 0518_HMI,0614_ABS,  0618_SCM, 0865_BSM, 1048_EPM, 1138_LeanTCS
RELATIVE_SW_SPEC_PATH   = "/zQC_Spec/SwT/"
RELATIVE_SYT_SPEC_PATH  = "/zQC_Spec/SysT/"
SW_DataSPEC_PATH        = "./zQC_Spec/z_SwT_Data/test.csv" #SwT_Data
SyT_DataSPEC_PATH       = "./zQC_Spec/z_SysT_Data/test.csv"
EMBEDDED_PATH           = "./Data/glove_6B_50d.txt" 
NEWWORDS_PATH           = "./Data/NewWords.txt"
FUNCTION_PATH           = "./Signals/FunctionNames.txt"
ENUM_PATH               = "./Signals/ENUMs.txt" 
SHEET_NAME = "Test: Description"
cwd = os.getcwd()  # Get the current working directory (cwd)
specs = []

def getExceptedWords(text, embeddedPath = EMBEDDED_PATH, newWordsPath = NEWWORDS_PATH):
    
     NotInDictWords = []
     setOfWords = set(text.split())
     word_to_index, __, __ = read_glove_vecs(embeddedPath, newWordsPath)
     all_words = word_to_index.keys()
     NotInDictWords = [word for word in setOfWords if word not in all_words]
     return NotInDictWords, all_words
 
def getSheetSpec(specPath, display = False):
    
    df = pd.read_excel(specPath, sheet_name = 'Sheet1', encoding = "utf-8")
    descriptions = df[SHEET_NAME];     sheet = ""
    number_of_tests = 0
    for des in descriptions:
        spec = Specification(des)
        specs.append(spec)
        sheet = os.linesep.join([sheet, spec.preTrainingData + spec.stiTrainingData])
        number_of_tests += 1
        if display:
            print("STT is {}".format(number_of_tests))
            print(spec.preTrainingData + spec.stiTrainingData)
    return sheet

def enum_function(labels):
    label_dict = {}
    remain_labels = labels.copy()
    with open(FUNCTION_PATH, 'r') as file:
        FUNCTION_TEXT = file.read().lower()
        
    with open(ENUM_PATH, 'r') as file:
        ENUM_TEXT = file.read().lower()

    FUNCTION_TEXTs = FUNCTION_TEXT.split()
    ENUM_TEXTs = ENUM_TEXT.split()

    for label in labels:
        if label.lower() in FUNCTION_TEXTs:
            label_dict[label] = "function"
            remain_labels.remove(label)
    for label in labels:
        if label.lower() in ENUM_TEXTs:
            label_dict[label] = "value"
            remain_labels.remove(label)
    return label_dict, remain_labels

def allPassFilter(sheet):
    signals = []
    unknowWords, all_words = getExceptedWords(sheet.lower(), EMBEDDED_PATH, NEWWORDS_PATH)
    accepted_signals_dict, resignals = enum_function(unknowWords)
    spellchecked_dict = SymSpell.checker(resignals)

    for key in spellchecked_dict:
        spellchecked_dict[key] = None if spellchecked_dict[key] not in all_words else spellchecked_dict[key]

    for key in spellchecked_dict:
        if not spellchecked_dict[key]:
            signals.append(key)
        else:
            accepted_signals_dict[key] = spellchecked_dict[key]
    
    spellchecked_dict.clear()
    proper_signals = SignalsClassifier(signals)
    signals_dict = dict(zip(signals, proper_signals))
    signals_dict.update(accepted_signals_dict)
    return signals_dict

def CollectData():
    sheet = None
    csvFile = open(SyT_DataSPEC_PATH, 'a')
    for filename in os.listdir(os.getcwd() + RELATIVE_SYT_SPEC_PATH):
        if filename in ('.DS_Store'):
            pass
        else:
            print("Handling file {}".format(filename))
           
            sheet = getSheetSpec(os.getcwd() + RELATIVE_SYT_SPEC_PATH + filename, True)
        if not sheet:
            continue
        signals_dict = allPassFilter(sheet)
        print(signals_dict)
        
        for spec in specs:
            for label, new_label in signals_dict.items():
                spec.preTrainingData = re.sub(r'\b' + label + r'\b', new_label, spec.preTrainingData, flags = re.IGNORECASE)
                spec.stiTrainingData = re.sub(r'\b' + label + r'\b', new_label, spec.stiTrainingData, flags=re.IGNORECASE)
                
            preTextList = spec.preTrainingData.splitlines()
            stiTextList = spec.stiTrainingData.splitlines()
            for preText in preTextList:
                csvFile.write("\"" + preText + "\"\n")
            for preText in stiTextList:
                csvFile.write("\"" + preText + "\"\n")       
        specs.clear()
    csvFile.close()

def get_spec_list(relative_path, spec_display = False):
    sheet = None
    specs.clear()

    sheet = getSheetSpec(os.getcwd() + relative_path, spec_display)
    signals_dict = allPassFilter(sheet)
    for spec in specs:
        for label, new_label in signals_dict.items():
            spec.preTrainingData = re.sub(r'\b' + label + r'\b', new_label, spec.preTrainingData, flags = re.IGNORECASE)
            spec.stiTrainingData = re.sub(r'\b' + label + r'\b', new_label, spec.stiTrainingData, flags=re.IGNORECASE)
        spec.preTextList = spec.preTrainingData.splitlines()
        spec.stiTestList = spec.stiTrainingData.splitlines()
        spec.preTextList = list(filter(None, spec.preTextList))
        spec.stiTestList = list(filter(None, spec.stiTestList))
        spec.content = None  #don't need anymore.
    return specs

if __name__ == "__main__":
    #CollectData()
    myspecs = get_spec_list('/zQC_Spec/DTC.xlsx')
