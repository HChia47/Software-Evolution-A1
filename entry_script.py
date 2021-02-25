import csv
import sys
import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *


def write_output_file():
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('/output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        writer.writerow(["UC1", "L1, L34, L5"]) 
        writer.writerow(["UC2", "L5, L4"]) 


if __name__ == "__main__":
    '''
    Entry point for the script
    '''
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    match_type = 0

    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)    

    print(f"Hello world, running with matchtype {match_type}!")

    # Read input low-level requirements and count them (ignore header line).
    with open("/input/low.csv", "r") as inputfile:
        print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")


    '''
    This is where you should implement the trace level logic as discussed in the 
    assignment on Canvas. Please ensure that you take care to deliver clean,
    modular, and well-commented code.
    '''
    print('The nltk version is {}.'.format(nltk.__version__))
    
    def getInputLowRequirements():
        df = pd.read_csv("/input/low.csv")
        dfnp = df.to_numpy()
        return dfnp

    def getInputHighRequirements():
        df = pd.read_csv("/input/high.csv")
        dfnp = df.to_numpy()
        return dfnp
    
    # Tokenize a sentence and remove commas and dots
    # def tokenizeSentence(sentence):
    #     tokens = word_tokenize(sentence)
    #     i = 0
    #     while i < len(tokens):
    #         if tokens[i] == '.':
    #             tokens.remove('.')
    #         elif tokens[i] ==  ',':
    #             tokens.remove(',')
    #         elif tokens[i] == '\'':
    #             tokens.remove('\'')
    #         elif tokens[i] == 's':
    #             tokens.remove('s')  
    #         else:
    #             stop_words = set(stopwords.words('english'))
    #             if tokens[i] in stop_words:
    #                 tokens.remove(tokens[i])
    #             else:
    #                 i+=1   
    #     return tokens

    # Tokenize a sentence and remove commas and dots
    def tokenizeSentenceRegex(sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        i = 0
        while i < len(tokens):
            tokens[i] = tokens[i].lower()
            stop_words = set(stopwords.words('english'))
            if tokens[i] in stop_words:
                tokens.remove(tokens[i])
            else:
                i+=1   
        return tokens

    def stemmedSentences(list):
        stemmedlist = [stemmer.stem(token) for token in list ]
        return stemmedlist
        


    def tokenizeAllRequirements(data):
        listReqAndTokens = []
        allReqIDs = data[:, 0]
        allReqSentences = data[:, 1]
        allReqTokenizedSentences = []
        for x in allReqSentences:
            allReqTokenizedSentences.append(tokenizeSentenceRegex(x))
        
        print("allReqTokenizedSentences")
        print(allReqTokenizedSentences)

        allReqTokenizedStemmedSentences = []
        for x in allReqTokenizedSentences:
            allReqTokenizedStemmedSentences.append(stemmedSentences(x))

        for ID, Sentence in zip(allReqIDs, allReqTokenizedStemmedSentences):
            tempList = [ID, Sentence]
            listReqAndTokens.append(tempList)

        print("listReqAndTokens")    
        print(listReqAndTokens)
        print(type(listReqAndTokens))
        return listReqAndTokens

    def extendsAllTokenizedLists(tokenizedData):
        extTokList = []
        for x in tokenizedData:
            extTokList.extend(x[1])
        return extTokList

    # make dictionary

    # frequecy of word direct
    def masterDictionaryCountMethod(listWords):
        dictCount = dict((x,listWords.count(x)) for x in set(listWords))
        print(dictCount)
        return dictCount

    # n total number of requirement
    def countAllRequirement(low,high):
        totalAmountRequirements =len(low)+len(high)
        print(totalAmountRequirements)
        return totalAmountRequirements

    # in how many requirements is the word 

    #list no duplicates
    def noDubListMethode(list):
        newlist = []
        for i in list:
            if i not in newlist:
                newlist.append(i)
        print("noDubList")
        print(newlist)
        return newlist

    dataLow = getInputLowRequirements()
    dataHigh = getInputHighRequirements()

    stemmer = PorterStemmer()
    tokenizeDataLow = tokenizeAllRequirements(dataLow)
    tokenizeDataHigh = tokenizeAllRequirements(dataHigh)



    extTokListLow = extendsAllTokenizedLists(tokenizeDataLow)
    print("extTokListLow")
    print(extTokListLow)
    extTokListHigh = extendsAllTokenizedLists(tokenizeDataHigh)
    print(len(extTokListHigh))
    print(extTokListHigh)
    extTokListHigh.extend(extTokListLow)
    extTokListOriginal = extTokListHigh
    extTokListCopy = extTokListOriginal
    extTokListCopy1 = extTokListOriginal

    noDublist = noDubListMethode(extTokListCopy1)

    masterDictionaryEmpty = { word : 0 for word in extTokListHigh}
    print(masterDictionaryEmpty)
    masterDictionaryCount = masterDictionaryCountMethod(extTokListCopy)
    totalAmountRequirements = countAllRequirement(tokenizeDataLow,tokenizeDataHigh)


    write_output_file()