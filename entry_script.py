import csv
import sys
import nltk
import pandas as pd
import numpy as np
import math
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from collections import Counter
from scipy import spatial


def write_output_file(traceLinks):
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('/output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        for x in traceLinks:
            tempX = x
            strY = ""  
            for y in range(len(tempX[1])):
                if(y != (len(tempX[1]) - 1)):
                    strY += x[1][y]
                    strY += ","
                else:
                    strY += x[1][y]
            writer.writerow([x[0], strY])

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

# Complete master dictionary
def masterDictionaryFinished(dictionary,dictionary2):
    for k in dictionary.keys():
        dictionary[k] = dictionary[k] * dictionary2[k]
        print(dictionary[k])
    # print(result)
    return dictionary



# frequecy of word dictionary
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
def checkWordInNumberOfRequirements(noDubList,emptyMasterDict,allRequirement):
    for word in noDubList:
        i = 0
        while i < len(allRequirement):
            if word in allRequirement[i][1]:
                emptyMasterDict[word] += 1
            i+=1
    return emptyMasterDict

# idf value for every key in master dictionary
def idfMasterDictionary(dictionary, totalRequirements):
    for key in dictionary:
        dictionary[key] = math.log2(totalRequirements/dictionary.get(key))
    print(dictionary)
    return dictionary            
            
#list no duplicates
def noDubListMethode(list):
    newlist = []
    for i in list:
        if i not in newlist:
            newlist.append(i)
    print("noDubList")
    print(newlist)
    sortedlist = sorted(newlist)
    print("sortedlist")
    print(sortedlist)
    return sortedlist

def createVectorRepresentation(sortedNoDubTokenList, singleReqTokenList, masterDict):
    vectorRep = []
    for x in sortedNoDubTokenList:
        if(x in singleReqTokenList):
            vectorRep.append(masterDict[x])
        else:
            vectorRep.append(0)
    #print("singleReqTokenList")
    #print(singleReqTokenList)
    #print("vectorRep")
    #print(vectorRep)
    return vectorRep

def createAllVectorRepresentations(sortedNoDubTokenList, tokenData, masterDict):
    allVectorRep = []
    for x in tokenData:
        vectorRep = createVectorRepresentation(sortedNoDubTokenList, x[1], masterDict)
        allVectorRep.append(vectorRep)
    return allVectorRep

def calcCosineSimilarity(vector1, vector2):
    result = 1 - spatial.distance.cosine(vector1, vector2)
    return result

def calcSimilarityMatrix(highReqVectors, lowReqVectors):
    simMatrix = np.zeros((len(highReqVectors), len(lowReqVectors)))
    for i in range(len(highReqVectors)):
        for j in range(len(lowReqVectors)):
            simMatrix[i,j] = calcCosineSimilarity(highReqVectors[i], lowReqVectors[j])
    return simMatrix

def createTracelinks(simMatrix, tokenDataLow, tokenDataHigh, matchtype):
    traceLinks = []
    if(matchtype == 1):
        traceLinks = createTraceLinksOne(simMatrix, tokenDataLow, tokenDataHigh)
    elif(matchtype == 2):
        traceLinks = createTraceLinksTwo(simMatrix, tokenDataLow, tokenDataHigh)
    elif(matchtype == 3):
        traceLinks = createTraceLinksThree(simMatrix, tokenDataLow, tokenDataHigh)
    else:
        traceLinks = createTraceLinksFour(simMatrix, tokenDataLow, tokenDataHigh)
    return traceLinks

def createTraceLinksOne(simMatrix, tokenDataLow, tokenDataHigh):
    matrixShape = simMatrix.shape
    rowLen = matrixShape[0]
    columnLen = matrixShape[1]
    traceLinks = []
    for x in range(rowLen):
        currentHighReqLink = []
        currentReqLinks = []
        for y in range(columnLen):
            if(simMatrix[x,y] > 0):
                currentReqLinks.append(tokenDataLow[y][0])
        currentHighReqLink.append(tokenDataHigh[x][0])
        currentHighReqLink.append(currentReqLinks)
        traceLinks.append(currentHighReqLink)
    return traceLinks

def createTraceLinksTwo(simMatrix, tokenDataLow, tokenDataHigh):
    matrixShape = simMatrix.shape
    rowLen = matrixShape[0]
    columnLen = matrixShape[1]
    traceLinks = []
    for x in range(rowLen):
        currentHighReqLink = []
        currentReqLinks = []
        for y in range(columnLen):
            if(simMatrix[x,y] >= 0.25):
                currentReqLinks.append(tokenDataLow[y][0])
        currentHighReqLink.append(tokenDataHigh[x][0])
        currentHighReqLink.append(currentReqLinks)
        traceLinks.append(currentHighReqLink)
    return traceLinks

def createHighestSimList(simMatrix):
    highSim = np.amax(simMatrix, axis=1)
    return highSim

def createTraceLinksThree(simMatrix, tokenDataLow, tokenDataHigh):
    matrixShape = simMatrix.shape
    rowLen = matrixShape[0]
    columnLen = matrixShape[1]
    highestSimPerRow = createHighestSimList(simMatrix)
    traceLinks = []
    for x in range(rowLen):
        currentHighReqLink = []
        currentReqLinks = []
        for y in range(columnLen):
            if(simMatrix[x,y] >= (0.67 * highestSimPerRow[x])):
                currentReqLinks.append(tokenDataLow[y][0])
        currentHighReqLink.append(tokenDataHigh[x][0])
        currentHighReqLink.append(currentReqLinks)
        traceLinks.append(currentHighReqLink)
    return traceLinks

def createTraceLinksFour(simMatrix, tokenDataLow, tokenDataHigh):
    matrixShape = simMatrix.shape
    rowLen = matrixShape[0]
    columnLen = matrixShape[1]
    highestSimPerRow = createHighestSimList(simMatrix)
    traceLinks = []
    for x in range(rowLen):
        currentHighReqLink = []
        currentReqLinks = []
        for y in range(columnLen):
            if(simMatrix[x,y] >= (0.67 * highestSimPerRow[x])):
                currentReqLinks.append(tokenDataLow[y][0])
        currentHighReqLink.append(tokenDataHigh[x][0])
        currentHighReqLink.append(currentReqLinks)
        traceLinks.append(currentHighReqLink)
    return traceLinks

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

    noDubList = noDubListMethode(extTokListCopy1)

    masterDictionaryEmpty = { word : 0 for word in extTokListHigh}
    masterDictionaryCount = masterDictionaryCountMethod(extTokListCopy)
    totalAmountRequirements = countAllRequirement(tokenizeDataLow,tokenizeDataHigh)
    masterDictionaryEmpty = checkWordInNumberOfRequirements(noDubList,masterDictionaryEmpty,tokenizeDataLow)
    masterDictionaryEmpty = checkWordInNumberOfRequirements(noDubList,masterDictionaryEmpty,tokenizeDataHigh)
    masterDictionaryFull = masterDictionaryEmpty
    masterDictionaryIDF = idfMasterDictionary(masterDictionaryFull,totalAmountRequirements)
    masterDictionaryComplete = masterDictionaryFinished(masterDictionaryIDF, masterDictionaryCount)
    print(masterDictionaryComplete)

    print(noDubList)
    vectorRepReqLow = createAllVectorRepresentations(noDubList, tokenizeDataLow, masterDictionaryComplete)
    vectorRepReqHigh = createAllVectorRepresentations(noDubList, tokenizeDataHigh, masterDictionaryComplete)
    print("vectorRepReqLow")
    print(vectorRepReqLow)
    print(vectorRepReqLow[0])
    print(len(vectorRepReqLow[0]))
    print(vectorRepReqHigh[0])
    print(len(vectorRepReqHigh[0]))
    cosinesim = calcCosineSimilarity(vectorRepReqHigh[0], vectorRepReqLow[0])
    print(cosinesim)
    simMatrix = calcSimilarityMatrix(vectorRepReqHigh, vectorRepReqLow)
    print(simMatrix)
    traceLinksM1 = createTraceLinksThree(simMatrix, tokenizeDataLow, tokenizeDataHigh)
    #print("traceLinksM1")
    #print(traceLinksM1)
    simL = createHighestSimList(simMatrix)
    print("simL")
    print(simL)
    print("traceLinksM1")
    print(traceLinksM1)
    links = createTracelinks(simMatrix, tokenizeDataLow, tokenizeDataHigh, match_type)
    print("links")
    print(links)
    write_output_file(links)