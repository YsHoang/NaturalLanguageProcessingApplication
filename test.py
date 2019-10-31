import csv
import os
import re

# Some Notes:
# - Inside a character range, \b represents the backspace character
ENUM_MAPPING_PATH   = "./Mappings/EnumECUValuesMapping.csv"
COMMON_VERBS_PATH   = "./Data/CommonVerbs.txt"
THRESHOLD_MERGE_LINES = 0.6
 
class Pattern:
    
       VALUE                   = r'\s([+-]?\b\d+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)' 
       HEX_VALUES              = [r'\s(0x[0-9A-Fa-f]+)', r'(\s+[0-9A-Fa-f][0-9A-Fa-f]){2,32}\b']
       DEVIDE_VALUES           = [r'(/\d+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)']  #MAYBE r should be removed
       INVALID_WORDS           = [r'Refer +RB_STIMULATION', r'\.\.', r'\’|\'|"', r'[\[\]\(\)]', r'•']  #almost last step     
       ONE2MULLINES            = [ r'\s&\s', r'\s&&\s']
       SPLIT_SENTENCE          = r"(and|,)(( +[a-zA-Z-+*/\^]+){6,})"
       EXPRESSIONS             = [r'\b(time)(\+)']
       END_OF_LINE             = r'\s*[:;.\n\r\n]' 
       END_OF_TITLE            = r'(\s*[:;.\n\r\n])'
       DATA_HOLDER             = r'\s*[:;.\n\r\n]\s*(.+)|$' 
       START_OF_LINES          = [r'Step\s*[\d\w-]+[\.]*[\d]*\s*[.:\n\r\n\s]', r' *\d+\.\d+\s*[.:) ]', r' *\d+\s*[.:)]', r' *\w?\.*\d*\s*[.:)]']#spaces
       SPECIAL_CHARACTERS      = [r'([]()[{}])', r'(>|<|=|\+|\*)', r'(,|:)'] # Without space at the end of patterns
       SPECIAL_CHARACTERS_2    = [r'([.?:;]\s)', r'(%\s)', r'(\.)$']
       SI_UNITS                = [r'\s*(V|m/s\^2|m/s|m/s2|km/h|km/hr|kmph|kph|kmh|%|s|msec|min|mm/s|mm|mbar|millibar|ms|millisec|sec|nm|n|bar|bars|deg/s|deg|m/s²|°|ohm|kbit|mbit|w)\b', r'\s*(%|°)']
       NOTE_KEYWORDS            = ['Note', 'Hint']
       PUNCTUATION             = r'[:;.]'
       EXAMPLES_INFO           = [r'[.,;:]?\s*((Eg|ex|example)\s*[.:].+)', r'(i[.]e\s+.+)', r'(e.g\s[\w ]+)'] #Maybe further analysis
       PURPOSE_TITLES          = ["Purpose"]
       PRECONDITION_TITLES     = ['Precondition', 'Preconditions', "Pre-Condition"]
       STIMULATION_TITLES      = ["Test Procedure", "Test Steps", "Stimulation", "STIM_ACTIVE"]
       SEPERATE_SECTION        = [r'-{2,}', r'\.{2,}', r'\*{2,}']
       
       PURPOSE_TITLE           = 'PURPOSE'
       PRECONDITION_TITLE      = 'PRECONDITION'
       STIMULATION_TITLE       = 'STIMULATION' 
       STANDARD_NOTE           = 'NOTE'
       STANDARD_TITLES         = ['PURPOSE', 'PRECONDITION', 'STIMULATION' ]
       
       
class Specification:
    
    def __init__(self, content):
        
        self.content = content
        self.isSolPre, self.isSolSti = False, False
        #self.preInfo, stiInfo = [], []
        self.purpose = None
        self.precondition = None
        self.stimulation = None
        self.preTrainingData = None
        self.stiTrainingData = None
        self.preTextList = None
        self.stiTestList = None
        self.__cleanup__()
        self.__dataProcessing__()
        self.__trainingDataProcessing__()
    
    def removeAttachInfo(self):
        for sep_section in Pattern.SEPERATE_SECTION:
            self.content = re.sub(sep_section, "", self.content)

        self.content = self.precondition = re.split(r"^.*attachment.*info", self.content, flags = re.IGNORECASE|re.MULTILINE)[0]
        self.content = re.sub(r"[\'\"]","", self.content)
        self.content = re.sub(r".*http.*|.*bosch.com.*|.*\.url.*","", self.content, flags = re.IGNORECASE) 
        self.content = re.sub(r".*attachment.*info.*|.*walkt.+rough.*protocol.*","", self.content, flags = re.IGNORECASE)

    @staticmethod
    def __isTitlesNotIn__(text):
        states = [re.search(title + Pattern.END_OF_LINE, text, flags=re.I) for title in Pattern.STANDARD_TITLES]
        return (states.count(None) == len(states))

    def __removeNoteInfo__(self, keywords=Pattern.NOTE_KEYWORDS):  # get first part of the content
        self.content = re.sub(r"\s*" + Pattern.STANDARD_NOTE + Pattern.END_OF_LINE + r".{4,}", "", self.content, flags=re.I) #at least 4 characters after note.
        if Specification.__isTitlesNotIn__(self.content):
            return
        if Pattern.STANDARD_NOTE in self.content:
            #segments = re.split(Pattern.STANDARD_NOTE + Pattern.END_OF_LINE, self.content, flags = re.IGNORECASE)
            segments = re.split(Pattern.STANDARD_NOTE + r"\s*[:;., ]+" + os.linesep, self.content, flags = re.I|re.M)
        else:
            return
        new_segment = []
        for segment in segments:
            segment = "" if Specification.__isTitlesNotIn__(segment) else re.sub("^.+" + os.linesep, "", segment, flags = re.I)#remove first line after split
            new_segment.append(segment)
        self.content = "".join(new_segment)
        
    def removeEmptyLines(self):
        self.content = os.linesep.join([line for line in self.content.splitlines() if line.strip()])

    def __toStandard__(self):
        for title in Pattern.PURPOSE_TITLES:
            self.content = re.sub(title + Pattern.END_OF_TITLE, Pattern.PURPOSE_TITLE + r'\1', self.content, flags = re.IGNORECASE) 
        for title in Pattern.PRECONDITION_TITLES:
            self.content = re.sub(title + Pattern.END_OF_TITLE, Pattern.PRECONDITION_TITLE + r'\1', self.content, flags = re.IGNORECASE) 
        for title in Pattern.STIMULATION_TITLES: 
            self.content = re.sub(title + Pattern.END_OF_TITLE, Pattern.STIMULATION_TITLE + r'\1', self.content, flags = re.IGNORECASE)  
        for keyword in Pattern.NOTE_KEYWORDS: 
            self.content = re.sub(keyword + Pattern.END_OF_TITLE, Pattern.STANDARD_NOTE + r'\1', self.content, flags=re.IGNORECASE)
                
    def __replaceEnum2Value__(self, enumMappingPath = ENUM_MAPPING_PATH):
        with open(enumMappingPath) as csvMappingFile:
            csvReader = csv.reader(csvMappingFile)
            next(csvReader)
            enumMappings = {row[0]:row[1]+row[2] for row in csvReader}  
        for enum, value in enumMappings.items():
            self.content = re.sub(r'\b' + enum + r'\b', value, self.content, flags = re.M)

    def __timePlus__(self):
        for pattern in Pattern.EXPRESSIONS:
            self.content = re.sub(pattern, r'\1 \2', self.content, flags = re.I)
            
    def __cleanup__(self):
        self.removeAttachInfo()
        self.removeEmptyLines()
        self.__toStandard__()
        self.__removeNoteInfo__()
        self.__replaceEnum2Value__()
        self.__timePlus__()

    def __splitFields__(self):
        self.purpose = re.search(Pattern.PURPOSE_TITLE + Pattern.DATA_HOLDER, self.content, re.IGNORECASE).group(1)
        self.stimulation = re.search(Pattern.STIMULATION_TITLE + Pattern.DATA_HOLDER, self.content, flags = re.IGNORECASE|re.DOTALL).group(1)

        self.precondition = re.split(Pattern.STIMULATION_TITLE + Pattern.END_OF_LINE, self.content, flags = re.IGNORECASE|re.MULTILINE)[0]
        self.precondition = re.search(Pattern.PRECONDITION_TITLE + Pattern.DATA_HOLDER, self.precondition, flags = re.IGNORECASE|re.DOTALL).group(1)
        
        self.precondition = os.linesep.join([line for line in self.precondition.splitlines() if line.strip()]) if self.precondition else ""
         
        self.stimulation = self.content if not self.stimulation else self.stimulation

    def __setPrefixs__(self):
     
        prefixs = [len(re.findall(r"^ *"+ sol, self.stimulation, flags = re.IGNORECASE|re.MULTILINE)) for sol in Pattern.START_OF_LINES]
        self.isSolSti = True if sum(prefixs)/len(self.stimulation.splitlines()) >= THRESHOLD_MERGE_LINES else False 
    def __semiMergeLines__(self):
        
        solPatterns = r"\n(?!(" + "|".join(Pattern.START_OF_LINES) + r"))"
        self.stimulation = re.sub(solPatterns, " ", self.stimulation, flags = re.IGNORECASE) if self.isSolSti else self.stimulation

    def __deletePrefixs__(self):
        for sol in Pattern.START_OF_LINES:
            self.stimulation = re.sub( r"^" + sol + r"\s*", r"", self.stimulation, flags = re.I|re.M) 
            self.precondition = re.sub( r"^" + sol + r"\s*", r"", self.precondition, flags = re.I|re.M)
            
    def __putExampleInParentheses__(self):
        for pattern in Pattern.EXAMPLES_INFO:
            self.precondition = re.sub(pattern, r'(\1)', self.precondition, flags = re.I) 
            self.stimulation = re.sub(pattern, r'(\1)', self.stimulation, flags = re.I) 
            
    def __sepSpecialChar__(self):
        for pattern in Pattern.SPECIAL_CHARACTERS:
            self.stimulation = re.sub(pattern, r' \1 ', self.stimulation, flags = re.I)
            self.precondition = re.sub(pattern, r' \1 ', self.precondition, flags = re.I) 
            
        for pattern in Pattern.SPECIAL_CHARACTERS_2:
            self.stimulation = re.sub(pattern, r' \1', self.stimulation, flags = re.I)
            self.precondition = re.sub(pattern, r' \1', self.precondition, flags = re.I) 

        for pattern in Pattern.SI_UNITS:
            self.stimulation = re.sub(Pattern.VALUE + pattern, r' \1 \2 ', self.stimulation, flags = re.I)
            self.precondition = re.sub(Pattern.VALUE + pattern, r' \1 \2 ', self.precondition, flags = re.I) 
            
            self.stimulation = re.sub(r'm/s\^2', r'm/s2', self.stimulation, flags = re.I)
            self.precondition = re.sub(r'm/s\^2', r'm/s2', self.precondition, flags = re.I) 
            
 
    def __deleteANDinParentheses__(self):
        for pattern in Pattern.ONE2MULLINES:  
            self.precondition = re.sub(r"\(.+" + pattern + r".+\)", "", self.precondition, flags = re.I) 
            self.stimulation = re.sub(r"\(.+" + pattern + r".+\)", "", self.stimulation, flags = re.I)
        self.precondition = re.sub(r"[[(].+?(,|and).+?[\])]", "", self.precondition, flags = re.I) 
        self.stimulation = re.sub(r"[[(].+?(,|and).+?[\])]", "", self.stimulation, flags = re.I)

    #@staticmethod
    def __strimLeadingTrailingSpace__(self):
        self.precondition = os.linesep.join([line.strip() for line in self.precondition.splitlines() if line.strip()])
        self.stimulation = os.linesep.join([line.strip() for line in self.stimulation.splitlines() if line.strip()])
       
    def __splitSentence__(self):
        self.precondition = re.sub(Pattern.SPLIT_SENTENCE, os.linesep + r'\2', self.precondition, flags = re.I)
        self.stimulation = re.sub(Pattern.SPLIT_SENTENCE, os.linesep + r'\2', self.stimulation, flags = re.I)
              
    def __one2multiLines__(self):
             
        with open(COMMON_VERBS_PATH, 'r') as file:
            TEXT = file.read().lower()
            TEXT = re.sub(",", "", TEXT)  
            
        COMMON_VERBS = TEXT.split()
        for verb in COMMON_VERBS:
            self.precondition = re.sub(r'\sAnd\s+(' + verb + r"\s)", os.linesep + r'\1', self.precondition, flags = re.I) 
            self.stimulation = re.sub(r'\sAnd\s+(' + verb + r"\s)", os.linesep + r'\1', self.stimulation, flags = re.I)
                    
            self.precondition = re.sub(r'\s,\s+(' + verb + r"\s)", os.linesep + r'\1', self.precondition, flags = re.I) 
            self.stimulation = re.sub(r'\s,\s+(' + verb + r"\s)", os.linesep + r'\1', self.stimulation, flags=re.I)

            self.precondition = re.sub(r'\s[&]+\s+(' + verb + r"\s)", os.linesep + r'\1', self.precondition, flags = re.I) 
            self.stimulation = re.sub(r'\s[&]+\s+(' + verb + r"\s)", os.linesep + r'\1', self.stimulation, flags=re.I)
        '''        
        for pattern in Pattern.ONE2MULLINES:
            self.precondition = re.sub(pattern, os.linesep, self.precondition, flags = re.I) 
            self.stimulation = re.sub(pattern, os.linesep, self.stimulation, flags = re.I)
        '''           
    def __removeDuplicatedSpaces__(self):
        self.precondition = re.sub(" +", " ", self.precondition, flags = re.M) 
        self.stimulation = re.sub(" +", " ", self.stimulation, flags=re.M)

    def __removeDuplicatedSpaces2__(self):
        self.preTrainingData = re.sub(" +", " ", self.preTrainingData, flags = re.M) 
        self.stiTrainingData = re.sub(" +", " ", self.stiTrainingData, flags=re.M)

    def __removeEndOfSentence__(self):
        self.precondition = re.sub(r"[.,:;](\n|\r\n|$)" , os.linesep, self.precondition) 
        self.stimulation = re.sub(r"[.,:;](\n|\r\n|$)",os.linesep, self.stimulation) 
        self.preTrainingData = re.sub(r"[.,:;](\n|\r\n|$)" , os.linesep, self.preTrainingData) if self.preTrainingData else self.preTrainingData
        self.stiTrainingData = re.sub(r"[.,:;](\n|\r\n|$)" , os.linesep, self.stiTrainingData) if self.stiTrainingData else self.stiTrainingData
            
    def __rebuildFields__(self):
        # Removing spaces at the end of lines
        self.stimulation = os.linesep.join([line.strip() for line in self.stimulation.splitlines() if line.strip()])
        self.precondition = os.linesep.join([line.strip() for line in self.precondition.splitlines() if line.strip()])
        self.stimulation = self.stimulation + os.linesep #For at the end of fields when concatenate fields
        self.precondition = self.precondition + os.linesep
                  
    def __dataProcessing__(self):
         self.__splitFields__()
         self.__setPrefixs__()
         self.__semiMergeLines__()
         self.__deletePrefixs__()
         self.__putExampleInParentheses__()
         self.__sepSpecialChar__()
         self.__deleteANDinParentheses__()
         self.__one2multiLines__()
         self.__splitSentence__()
         self.__removeDuplicatedSpaces__()
         self.__removeEndOfSentence__()
         self.__rebuildFields__()

         
    def __removeInsideParentheses__(self):  #Info inside [] or ()
        self.preTrainingData = re.sub(r"[(][^)]*?[(].+?[)][^(]*?[)]","", self.precondition) # Nested ()
        self.stiTrainingData = re.sub(r"[(][^)]*?[(].+?[)][^(]*?[)]","", self.stimulation)  # Nested ()
        self.preTrainingData = re.sub(r"[[(].+?[\])]","", self.preTrainingData) 
        self.stiTrainingData = re.sub(r"[[(].+?[\])]","", self.stiTrainingData)  

    def __value2String__(self):

        for pattern in Pattern.HEX_VALUES:
            self.preTrainingData = re.sub(pattern, ' HEX_VALUE ', self.preTrainingData, flags = re.I)
            self.stiTrainingData = re.sub(pattern, ' HEX_VALUE ', self.stiTrainingData, flags = re.I) 
        
        for pattern in Pattern.DEVIDE_VALUES:
            self.preTrainingData = re.sub(pattern, '', self.preTrainingData, flags = re.I)
            self.stiTrainingData = re.sub(pattern, '', self.stiTrainingData, flags=re.I)
            
        self.preTrainingData = re.sub(Pattern.VALUE + "( |\n|\r\n)", r' VALUE \2', self.preTrainingData, flags = re.I)
        self.stiTrainingData = re.sub(Pattern.VALUE + "( |\n|\r\n)", r' VALUE \2', self.stiTrainingData, flags = re.I)        


    def __removeInvalidWords__(self):
        for pattern in Pattern.INVALID_WORDS:
            self.preTrainingData = re.sub(pattern, "", self.preTrainingData, flags = re.I)
            self.stiTrainingData = re.sub(pattern, "", self.stiTrainingData, flags = re.I)
            
    def __rebuildTrainingData__(self):
        # Removing spaces at the end of lines
        self.stiTrainingData = os.linesep.join([line.strip() for line in self.stiTrainingData.splitlines() if line.strip()])
        self.preTrainingData = os.linesep.join([line.strip() for line in self.preTrainingData.splitlines() if line.strip()])
        
        self.__removeEndOfSentence__()
        self.__removeDuplicatedSpaces2__()
        self.stiTrainingData = self.stiTrainingData + os.linesep #For at the end of fields when concatenate fields
        self.preTrainingData = self.preTrainingData + os.linesep

        
    def __trainingDataProcessing__(self):
        self.__removeInsideParentheses__()
        self.__value2String__()
        self.__removeInvalidWords__()
        self.__rebuildTrainingData__()
        # Folowing notes need to take into account:  Refer RB_Stimulation

if __name__ == "__main__":

    with open("example.txt", 'r', encoding = "utf-8") as in_file:
        mystring = in_file.read()
    
    spec = Specification(mystring)
    print(spec.preTrainingData)
    print(spec.stiTrainingData)

