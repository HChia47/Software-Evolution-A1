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
from sklearn.metrics import confusion_matrix


def write_output_file(traceLinks):
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('/output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)
        #loop through the links and get the IDs to add to the writer in the correct format
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

#read the requirements from the low.csv file and create a numpy array for the data in it
def getInputLowRequirements():
        df = pd.read_csv("/input/low.csv")
        dfnp = df.to_numpy()
        return dfnp

#read the requirements from the high.csv file and create a numpy array for the data in it
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

# Tokenize a sentence and remove commas and dots and remove stopwords
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
    
#tokenizes all requirements and return it in a list with elements of the format [ID, TokenizedRequirement]
def tokenizeAllRequirements(data):
    listReqAndTokens = []
    allReqIDs = data[:, 0]
    allReqSentences = data[:, 1]
    allReqTokenizedSentences = []
    for x in allReqSentences:
        allReqTokenizedSentences.append(tokenizeSentenceRegex(x))
    
    #print("allReqTokenizedSentences")
    #print(allReqTokenizedSentences)

    allReqTokenizedStemmedSentences = []
    for x in allReqTokenizedSentences:
        allReqTokenizedStemmedSentences.append(stemmedSentences(x))

    for ID, Sentence in zip(allReqIDs, allReqTokenizedStemmedSentences):
        tempList = [ID, Sentence]
        listReqAndTokens.append(tempList)

    #print("listReqAndTokens")    
    #print(listReqAndTokens)
    #print(type(listReqAndTokens))
    return listReqAndTokens

#adds all tokens of all requirements together (needed to create master vocabulary)
def extendsAllTokenizedLists(tokenizedData):
    extTokList = []
    for x in tokenizedData:
        extTokList.extend(x[1])
    return extTokList

# Complete master dictionary
def masterDictionaryFinished(dictionary,dictionary2):
    for k in dictionary.keys():
        dictionary[k] = dictionary[k] * dictionary2[k]
    # print(result)
    return dictionary



# frequecy of word dictionary
def masterDictionaryCountMethod(listWords):
    dictCount = dict((x,listWords.count(x)) for x in set(listWords))
    return dictCount

# n total number of requirement
def countAllRequirement(low,high):
    totalAmountRequirements =len(low)+len(high)
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
    sortedlist = sorted(newlist)
    return sortedlist

#create a vector representation for a requirement
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

#create a vector representation for all requirements that are in the parameter tokenData
def createAllVectorRepresentations(sortedNoDubTokenList, tokenData, masterDict):
    allVectorRep = []
    for x in tokenData:
        vectorRep = createVectorRepresentation(sortedNoDubTokenList, x[1], masterDict)
        allVectorRep.append(vectorRep)
    return allVectorRep

#Calculates the cosine similarity between a high and low level requirement vector representation
def calcCosineSimilarity(vector1, vector2):
    result = 1 - spatial.distance.cosine(vector1, vector2)
    return result

#Calculates the similarity matrix with all combination of high and low level requirement vector representations
def calcSimilarityMatrix(highReqVectors, lowReqVectors):
    simMatrix = np.zeros((len(highReqVectors), len(lowReqVectors)))
    for i in range(len(highReqVectors)):
        for j in range(len(lowReqVectors)):
            simMatrix[i,j] = calcCosineSimilarity(highReqVectors[i], lowReqVectors[j])
    return simMatrix

#creates the tracelinks and the method of doing so according to the matchtype
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

#creates tracelinks based on match_type 1, meaning a similarity score of at least >0
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

#creates tracelinks based on match_type 2, meaning a similarity score of at least 0.25
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

#creates tracelinks based on match_type 3, meaning that for every high level requirement 
#the highest similarity between the between the high and low requirement is found and used
#to determine that it has to have at least a similarity score of sim(h, l') >= 0.67 sim(h,l)
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
            if(simMatrix[x,y] >= (0.9 * highestSimPerRow[x])):
                currentReqLinks.append(tokenDataLow[y][0])
        currentHighReqLink.append(tokenDataHigh[x][0])
        currentHighReqLink.append(currentReqLinks)
        traceLinks.append(currentHighReqLink)
    return traceLinks

def getList(data):
    listRequirements = data[:, 0]
    listRequirements = listRequirements.tolist()
    return listRequirements

def createFunctionalRequirementUseCaseList(frList,useCaseList):

    frUseCaseList= []
    i = 0
    while i < len(frList):
        useCaseDict = {}
        keys = useCaseList
        for key in keys:
            useCaseDict[key] =  0
        frUseCaseList.append(useCaseDict)
        i+=1
    return frUseCaseList    

def getManualLink():
    with open('/input/links.csv', newline='') as f:
        reader = csv.reader(f)
        data1 = list(reader)
    return data1

def getToolLink():
    with open('/output/links.csv', newline='') as f:
        reader = csv.reader(f)
        data1 = list(reader)
    return data1

def removeWhiteSpace(list):
    returnlist = []
    for usecase in list:
        string = usecase.strip()
        returnlist.append(string)
    return returnlist

def transformData(data):
    datalistsplit = []
    for line in data:
        newline = []
        newline = line[1].split(",")
        newline = removeWhiteSpace(newline)
        datalistsplit.append(newline)
    datalistsplit.pop(0)
    return datalistsplit


def createBinaryList(binaryTraceLinkList, traceLink):
    i=0
    while i < len(binaryTraceLinkList):
        j=0
        while j < len(traceLink[i]):
            if traceLink[i][j] in binaryTraceLinkList[i]:
                binaryTraceLinkList[i][traceLink[i][j]] = 1
            j+=1
        i+=1
    return binaryTraceLinkList

def onlyBinaryValues(binaryList):
    completeBinaryList = []
    for dicts in binaryList:
        binaryValues = dicts.values()
        completeBinaryList.extend(binaryValues)
    return completeBinaryList

def binaryXORmethod(binarylistManual,binaryListTool):
    bitsXOR = np.bitwise_xor(binarylistManual,binaryListTool)
    return(bitsXOR.tolist())
    
def transformBinaryIntoOutput(functionalRequirementList,useCaseList,binaryMethod):
    i=0
    outputList = []
    binarycounter = 0
    while i < len(functionalRequirementList):
        functionlist = []
        insideUseCaseList = []
        j = 0
        while j < len(useCaseList):
            if j == 0:
                functionlist.append(functionalRequirementList[i])
            elif binaryMethod[binarycounter] == 1:
                insideUseCaseList.append(useCaseList[j])
                binarycounter+=1
            else:
                binarycounter+=1
            j+=1
        functionlist.append(insideUseCaseList)
        outputList.append(functionlist)
        i+=1
    return(outputList)

def write_output_fileMisHap(traceLinks):
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('/output/misclassifications.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)
        #loop through the links and get the IDs to add to the writer in the correct format
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
    #print("extTokListLow")
    #print(extTokListLow)
    extTokListHigh = extendsAllTokenizedLists(tokenizeDataHigh)
    #print(len(extTokListHigh))
    #print(extTokListHigh)
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
    #print(masterDictionaryComplete)

    #print(noDubList)
    vectorRepReqLow = createAllVectorRepresentations(noDubList, tokenizeDataLow, masterDictionaryComplete)
    vectorRepReqHigh = createAllVectorRepresentations(noDubList, tokenizeDataHigh, masterDictionaryComplete)
    #print("vectorRepReqLow")
    #print(vectorRepReqLow)
    #print(vectorRepReqLow[0])
    #print(len(vectorRepReqLow[0]))
    #print(vectorRepReqHigh[0])
    #print(len(vectorRepReqHigh[0]))
    cosinesim = calcCosineSimilarity(vectorRepReqHigh[0], vectorRepReqLow[0])
    #print(cosinesim)
    simMatrix = calcSimilarityMatrix(vectorRepReqHigh, vectorRepReqLow)
    #print(simMatrix)
    traceLinksM1 = createTraceLinksThree(simMatrix, tokenizeDataLow, tokenizeDataHigh)
    #print("traceLinksM1")
    #print(traceLinksM1)
    simL = createHighestSimList(simMatrix)
    #print("simL")
    #print(simL)
    #print("traceLinksM1")
    #print(traceLinksM1)
    links = createTracelinks(simMatrix, tokenizeDataLow, tokenizeDataHigh, match_type)
    #print("links")
    #print(links)
    write_output_file(links)

    # 
    #
    # Get confusion matrix out of results
    #
    #
    

    #get low and high requirements datalist
    dataLow = getInputLowRequirements()
    dataHigh = getInputHighRequirements()

    #get all usecases and functional requirements
    useCaseList = getList(dataLow)
    functionalRequirementList = getList(dataHigh)

    #create two binary empty confusion matrix set with manual value and tool value
    binaryTraceLinkManualList = createFunctionalRequirementUseCaseList(functionalRequirementList,useCaseList)
    binaryTraceLinkToolList = createFunctionalRequirementUseCaseList(functionalRequirementList,useCaseList)

    #get data of all trace-link files
    data1 = getManualLink()
    data2 = getToolLink()

    #get all trace links in cvs file in double list format
    manualTraceLinkUseCases = transformData(data1)
    toolTraceLinkUseCases = transformData(data2)

    #fill in empty confusion matrix set values
    binaryTraceLinkManualList = createBinaryList(binaryTraceLinkManualList , manualTraceLinkUseCases)
    binaryTraceLinkToolList = createBinaryList(binaryTraceLinkToolList , toolTraceLinkUseCases)

    #tranform it in a from such that it can be filled in function confusion matrix
    binaryTraceLinkManualList = onlyBinaryValues(binaryTraceLinkManualList)
    binaryTraceLinkToolList = onlyBinaryValues(binaryTraceLinkToolList)

    #create output csv file of all misclassifications
    outputlistMishap = transformBinaryIntoOutput(functionalRequirementList,useCaseList,binaryXORmethod(binaryTraceLinkManualList,binaryTraceLinkToolList))
    write_output_fileMisHap(outputlistMishap)

    #create the actual confusion matrix
    conf_mat = confusion_matrix(binaryTraceLinkManualList, binaryTraceLinkToolList)

    tn, fp, fn, tp = conf_mat.ravel()

    recall = tp/(fn+tp) 
    precision = tp/(fp+tp)
    fMeasure = (2*recall*precision)/(precision+recall)
    print(conf_mat)
    print("tn : " + str(tn))
    print("fp : " + str(fp))
    print("fn : " + str(fn))
    print("tp : " + str(tp))
    print("recall: "+ str(recall))
    print("precision: "+ str(precision))
    print("fMeasure: "+ str(fMeasure))

    
