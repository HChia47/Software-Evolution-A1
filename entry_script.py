import csv
import sys
import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')
from nltk.tokenize import word_tokenize


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
        return df

    def getInputHighRequirements():
        df = pd.read_csv("/input/high.csv")
        return df
    
    # Tokenize a sentence and remove commas and dots
    def tokenizeSentence(sentence):
        tokens = word_tokenize(sentence)
        i = 0
        while i < len(tokens):
            if tokens[i] == '.':
                tokens.remove('.')
            elif tokens[i] ==  ',':
                tokens.remove(',')   
            else:
                i+=1
        return tokens

    
    dfLow = getInputLowRequirements()
    npLow = dfLow.to_numpy()
    print(npLow)
    print(npLow[0,0])
    print(npLow[0, :])
    print(word_tokenize(npLow[0,1]))
    tokensLow1 = tokenizeSentence(npLow[0,1])
    print(tokensLow1)
    print(type(tokensLow1))
    write_output_file()