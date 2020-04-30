#!/usr/bin/env python3
import sys
import csv
import numpy as np
from random import randrange
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

import numpy as np
import string
import pickle
import os
csv.field_size_limit(sys.maxsize)

#
# Language Detection System for Text
# April 26th, 2020
#
# Corey Manders

#----------------- utility functions -------------------

# getWordFreq takes as input a single word as 
# a character string. The output is the frquency 
# of the word from the oneLang list of words for a 
# single language.
# 
# input: word (character string)
#        oneLang (dim 2 and 3 from the frequency model)
#
# output: frequency of word from the oneLang frequency table
#          if it is in the table, 0 if it is not found

def getWordFreq(word, oneLang):
    justWord = [row[0] for row in oneLang]

    if word in justWord:
        return float(oneLang[justWord.index(word)][1])
    else:
        return 0

# convertToFrqCounts returns a list of word frequencies 
# corresponding to a list of words given as input. 
# 
# input: wordList    - a list of words to find frequencies for
#        languageRef - the 3D language frequency model generated
#                      by buildLanguageCounts
#        languageIndex - the index to be used (first dim of languageRef)
#
# output: a list of frquencies corresponding to wordList

def convertToFreqCounts(wordList, languageRef, languageIndex):
    freqs=[]
    numOccurances = 0
   # print("->",wordList)
    for word in wordList:
       # print("checking!!",word)
        freq= getWordFreq(word,languageRef[languageIndex])
        if freq > 0:
            numOccurances+=1
        freqs.append(freq)
    return freqs, numOccurances

# seperateToWords takes a character string as input,
# and returns a list of words from the character string
#
# input: character string
# output: a list of strings, where every string is one word

def seperateToWords(input):
     # initialization of string to "" 
    new = "" 
    output = []
    
    # traverse in the string  
    for x in input: 
        if x.isspace():
            output.append(new)
            new=""
        else: 
            new += x
            
    return output, len(output)

def getNGramCounts(ngram_models):
    noOfNgms = []
    for j,lang in enumerate(languages):
            model = ngram_models[j]
            total = 0
            for key,v in model:
                total = total + v
            noOfNgms.append(total)
    return noOfNgms

def buildNGramModel(languages, langCounts, ngram ):
    nGramModels=[]
    for i,lang in enumerate(languages):
        justWord = [row[0] for row in langCounts[i]]
        langlist = ' '.join(justWord)

        #print(langlist[0:50])
        print("creating ngram model ",languages[i])
        # extracting the nGrams and sorting them according to their frequencies
        if ngram == 2:
            finder = BigramCollocationFinder.from_words(langlist)
        elif ngram == 3:
            finder = TrigramCollocationFinder.from_words(langlist)
        else:
            print("ngram type doesn't exist")
            exit(1)
        finder.apply_freq_filter(5)
        nGramModel = finder.ngram_fd.items()
        ngramModel = sorted(finder.ngram_fd.items(), key=lambda item: item[1],reverse=True)  
        nGramModels.append(nGramModel)


         
    return nGramModels



# buildLanguageCounts uses the openSubtitles frequency dataset
# to build ordered lists for the frquencies of the sizeOfDict
# most common words that at least minWordSize characters in length
#
# input: a list of languages to construct ordered frquency lists
#        for, the language should be in their two-letter abbreviations
#        for example, en for English, fr for French. 
#
# important to note, the function expects the text files of various 
# languages word frequencies to be in the relative directory
# 2018/(two-letter language abbreviation)/(two-letter language abbreviation)_50k.txt
#
# output: a three dimensional list
#         dim 1: the index of the language from the list passed in
#         dim 2: the sizeOfDict most frequent words which are minWordSize
#                or larger, in order of frquency
#         dim 3: the frequency of the word from dim2, unnormalized

def buildLanguageCounts(languages, dictSize, minWSize, filename):
    languageWordCounts=[[[0] * 2]*dictSize]*len(languages)
    
    currLang = 0
    for lang in languages:
        print("building frquency for language: ",lang)
        path = "2018/"+lang+"/"+lang+"_50k.txt"
        line_count = 0
        mostFreqCount = 0;

        with open(path) as lang_file:
            unsortedList = [[-1] * 2]*sizeOfDict
            csv_reader = csv.reader(lang_file, delimiter=' ')
            for row in csv_reader:
                if line_count < dictSize and len(row[0])>=minWSize:
                    unsortedList[line_count] = row
                    line_count+=1
                  

        languageWordCounts[currLang] =unsortedList
        currLang+=1
    
    return languageWordCounts

# grabRandomSentence will extract one sentence randomly from 
# a character string dataSet. The function assumes that sentences 
# end with "." and have spaces seperating words. 
#
# input: dataSet is a character string comprised of several sentences
# output: a list of words makimng one sentence (return value 1), and
#         the number of words in the sentence

def grabRandomSentence(dataSet):
    foundSentence = False

    while not foundSentence:
        foundStart = False
        start=randrange(len(dataSet))
        while start < len(dataSet) and not foundStart:
            if dataSet[start]=='.':
                start+=1
                while(start<len(dataSet) and dataSet[start].isspace() ):
                    start+=1
                    foundStart=True
            else:
                start+=1
        foundEnd = False
        end = start
        while (end< len(dataSet) and not foundEnd):
                if dataSet[end] == '.':
                    foundEnd = True
                else:
                    end+=1

        sentence, sentLen = seperateToWords(dataSet[start:end])
        if (sentLen> minTestSentenceLength):
            return sentence, sentLen
    #print(dataSet[start:end])
    return NoneType

#
# read data reads the data file (.csv) containing the two-letter
# language code as the first item in the csv field, the text 
# sample of the language in the second field, and the number of 
# words in the third field
#
# input: path to a csv file (for example formatted_data.csv)
# output: a 2d list, where the first dimension indexes the language sample
#         the second dimension is the index for (0), the two-letter 
#         language abbreviation, (1), the language sample, (2) the count
#         of words in the language sample
# 
def readData(inputPath):
    data=[]
    languages = []

    with open(inputPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                line_count += 1
                data.append(row)

    for line in data:
        languages.append(line[0])

    return data, languages

def writeToReportFileNGram(filehandle, trialRes, languages, numTrials):
    print("-->",trialRes)
    for k in range(len(trialRes)):
        filehandle.write(languages[k])
        filehandle.write(", ")
        filehandle.write(str(numTrials))
        filehandle.write(", ")
        count = 0;
        for t in range(numTrials):
            if trialResults[k][t]:
                count+=1;
        filehandle.write (str(count))
        filehandle.write('\n')

#-------------------------------------------
# testing code 
# -code to drive the testing of the detection stragetgy
#
# Detection strategy: 
# The language detection system leverages the data found in
# https://github.com/hermitdave/FrequencyWords 
# this repository uses the opensubtitle database, in which subtitles
# exist for many languages. For example, each language has a file
# with the top 50,000 words from the language, in decending order of frequency
# 
# The strategy evolved slightly during initial testing. The initial 
# thought was to first normalize the frquency counter from the FrequencyWords
# list. Then,for any list of words, get the normalized frequency for 
# each word, for each language being tested. Then add all of the frequencies,
# and use the highest summed value to predict the language. 
# However, though this did work oftern, there were cases where it would
# not. For example, if the list of words is short, and one of the words
# exists in two languages. If the frequency normalization resulted in 
# large number for one of the languages, even if it matched more words
# in a competing prediction, the language with the single frequency
# count may win the prediction, and result in a wrong prediction.
#
# Given this issue, a second method was selected. This was to take the 
# list of words, count the number of matches in the top N most frequent 
# words in the language, and then choose the language which had the 
# higest count. 
#
# This strategy overall works exceedingly well. There are likely ways
# to further improve the result, as well as making the prediction 
# more computationally light, however, this method seems to be 
# a good "base" starting point.
# 
# The code for testing this strategy/algorithm implements two methods
# of testing and generating accuracy values. Note that as the "training"
# data is the frequency lists for each language, and in this implementation
# none of the data from "formatted_data.csv" was actually used for 
# training, the complete content from this file is used for testing.
#
# The first method of testing is to pick M words from the testing set
# from one language. Then use the algortith to predict the language
# given all of the languages available in "formatted_data.csv" as
# potential candidates. The code below uses M={5,15,30}, however,
# this could be any number. The value 5 was chosen as a starting point,
# as certainly with the cross-over of identical words from different 
# languages, one sample may easily be too few to make a prediction. 
# Five seems to be a reasonable amount as a minimum number, however. 
# this could be reduced if required. 
# 
# There is a trade-off in the minimum size to of words used to 
# predict the language, and, the number of top N most frequent 
# words to use in the the frequency table. As the minimum size
# of words increases, the higher probability that words will be
# in the top frequencies, therefore improving the probability 
# of an accurate prediction. This would also allow reducing N, 
# thereby reducing the time needed to make predictions. 
#
# Each language was tested in this manner, and, the results are
# presented in the accompanying report. 
#
# The second method of testing was to sample random sentences 
# from "formatted_data.csv". Again, samples were taken from 
# each language sample, and predictions were made. Since the 
# number of words that make up a sentence in any language is
# highly variable, this is also considered. The minimum length 
# of sentence tested in this case was 5, and the output of the testing
# also included the length of sentence being tested. 
# As one would expect, errors generally only occur when the 
# length of the sentence is very short. 
#
# Both manners of testing are in the accompanying report. 

##  Experiments:
##  the following functions are are various ways of predicting the language
##
##  experiment 1: use word based frequency counting, predict by returning the
##                language that produces the most matches
def testRandomSequenceFreq(fileHandle, languages, numTrials, testLanguage, sequencesLens):
    prediction = -1
    randSeqResults=[]
    trialResultsSeq=[]

    if (numTrials > 0):
        fileHandle.write("testing :"+str(testLanguage)+
        " random sequence lengths "+str(sequencesLens)+"\n")
        print("random word sequences of lengths: ",sequencesLens)

    counts = np.zeros((len(languages),len(sequencesLens)))
    for trialNum in range(numTrials):
        print("running",languages[i],"trial:",trialNum+1,"of",numRandomSeqLengthTrial)
        for index,lang in enumerate(languages):
         
            randStart = randrange(numWordData)
            for seqLengthInd in range(len(sequencesLens)):
     
                currLength = int(sequencesLens[seqLengthInd])
                f, ret= convertToFreqCounts(test[randStart:randStart+currLength], langCounts,index)
                counts[index][seqLengthInd]=ret

        results = np.argmax(counts,axis = 0)
        resultsArgMax = np.equal(results, i)
        trialResultsSeq.append(resultsArgMax)
    if (numRandomSeqLengthTrial > 0):
        reportFile.write('\n'.join([','.join(['{:4}'.format(item) for item in row]) 
        for row in trialResultsSeq]))
        reportFile.write("\n")
    return trialResultsSeq

##  Experiment 2: Like experiment 1. But, instead of using sequential words of various
#                 length, use random sentences 

def testSentenceFreq(filehandle):

    if (numRandomSentenceTrial > 0):
        print("starting sentence testing")
        reportFile.write("testing: "+str(languages[i])+
        " random sentences from data ""\n")

    trialResultsSentence = []
    counts = np.zeros(len(languages))

    for trialNum in range (numRandomSentenceTrial):
        print("trialnum ",trialNum)
        sentence,sentLen = grabRandomSentence(data[i][1])
        for index,lang in enumerate(languages):
            
            freqs, counts[index] =convertToFreqCounts(sentence, langCounts,index)
        print(counts)
        results = np.argmax(counts)
        resultsArgMax = np.equal(results, i)
        trialResultsSentence.append([resultsArgMax, sentLen]) 
        #print(trialResultsSeq)

    if (numRandomSentenceTrial > 0):
        reportFile.write('\n'.join([','.join(['{:4}'.format(item) for item in row]) 
          for row in trialResultsSentence]))
        reportFile.write("\n")

## Experiment 3: rather than using word based frequency counting, use character n-grams
##               (for example bigrams or trigrams), and again return the language which
##               produces the highest number of matches. This test uses word sequences,
##               similar to experiment 1.

def testNGramsSeq(fileHandle, ngram_models, no_of_ngms, ngramSize,  trialRes):
        counts = np.zeros(len(languages))

        if ngramSize == 2:
            numTrials = numBigramSeqTrials
        elif ngramSize == 3:
            numTrials = numRandomTrigramTrials

        for trialNum in range (numTrials):
            print("----------------")
            print("trialnum ",trialNum)
            randStart = randrange(numWordData)
            ##########
            counts = np.zeros((len(languages),len(randWordSeqLength)))
          
            print("running"," ","trial:",trialNum+1,"of",numRandomSeqLengthTrial)
            for index,lang in enumerate(languages):
                    for seqLengthInd in range(len(randWordSeqLength)):
             
                        currLength = int(randWordSeqLength[seqLengthInd])

                        randString =test[randStart:randStart+currLength]

                        sentString  = ' '.join(randString)
                        if ngramSize == 2:
                            finder = BigramCollocationFinder.from_words(sentString)
                        elif ngramSize == 3:
                            finder = TrigramCollocationFinder.from_words(sentString)
                        
                        freq_sum = np.zeros(21) 
                        for k,v in finder.ngram_fd.items():
                            isthere = 0
                            for t,lang in enumerate(languages):
                                model=ngram_models[t]
                                for key,f in model:
                                    if k == key:
                                        #freq_sum[t] = freq_sum[t]+float(f)/float(no_of_bigms[t])
                                        freq_sum[t] +=1
                                        isthere = 1
                                        break
                   # if isthere == 0:
                   #     freq_sum[t] = freq_sum[t] + 

                        max_val = freq_sum.max()
                        index= freq_sum.argmax()
                        print("prediction:",index, "current Length = ",currLength)
        

##  Experiment 4: This experiment uses n-grams (bigrams and trigrams), but rather than using
##                word sequences, it uses random sentences, like experiment 2


def testNGrams(fileHandle, ngram_models, no_of_ngms, ngramSize,  trialRes):
        counts = np.zeros(len(languages))

        if ngramSize == 2:
            numTrials = numRandomBigramTrials
        elif ngramSize == 3:
            numTrials = numRandomTrigramTrials

        for trialNum in range (numTrials):
            print("----------------")
            print("trialnum ",trialNum)
            sentence2,sentLen = grabRandomSentence(data[i][1])
            sentString  = ' '.join(sentence2)

            #print(sentString)
            if ngramSize == 2:
                finder = BigramCollocationFinder.from_words(sentString)
            elif ngramSize == 3:
                finder = TrigramCollocationFinder.from_words(sentString)

            freq_sum = np.zeros(21) 

            for k,v in finder.ngram_fd.items():
                isthere = 0
                for t,lang in enumerate(languages):
                    model=ngram_models[t]
                    for key,f in model:
                        if k == key:
                            #freq_sum[t] = freq_sum[t]+float(f)/float(no_of_bigms[t])
                            freq_sum[t] +=1
                            isthere = 1
                            break
                   # if isthere == 0:
                   #     freq_sum[t] = freq_sum[t] + 

            max_val = freq_sum.max()
            index= freq_sum.argmax()
            trialRes.append(index)
            print("result =",max_val, index, languages[index])
            print("correct answer = ",languages[i])


inFilename = 'formatted_data.csv'
outFilename = 'report_testing.csv'
languageCountsPath = 'languageCounts.pcl'


#variables to be used for exploration and reporting
randWordSeqLength = (5, 15, 30)
numRandomSentenceTrial = 0
numRandomSeqLengthTrial = 0
numRandomBigramTrials = 0
numRandomTrigramTrials = 2
numBigramSeqTrials = 0

sizeOfDict = 50000
minWordSize = 4 # when building the word frequency tables, this is 
                # the minimum size word to consider
minTestSentenceLength = 5

data,languages = readData(inFilename)
reportFile = open(outFilename,"w+")

ngmModels=[]
noOfNgms=0


#create or load languageCount data
if os.path.isfile(languageCountsPath):
    langCounts = pickle.load( open( languageCountsPath, "rb" ))
else:
    langCounts = buildLanguageCounts(languages,sizeOfDict, minWordSize, languageCountsPath)
    pickle.dump(langCounts, open( languageCountsPath, "wb" ) )

if numRandomBigramTrials >0 or numBigramSeqTrials> 0:
    #create bigram model
    ngmModels = buildNGramModel(languages, langCounts, 2)
    noOfNgms = getNGramCounts(ngmModels)

if numRandomTrigramTrials > 0:
    ngmModels = buildNGramModel(languages, langCounts, 3)
    noOfNgms = getNGramCounts(ngmModels)

trialResults = []

for i in range(0,len(languages)):
#for i in range(0,2):

    print("---------------------")
    
    test,numWordData = seperateToWords(data[i][1])
    sentence,sentLen = grabRandomSentence(data[i][1])
    #sentenceInWords,numWords = seperateToWords(sentence)

    print("testing :",languages[i])

    ##### testing sequences of words from a random starting point
    testRandomSequenceFreq(reportFile,languages, numRandomSeqLengthTrial, languages[i],randWordSeqLength)
  

    ##### testing random sentences from the input files
    testSentenceFreq(reportFile)

    ##### testing random sentences from the input files using bigram 
    if numRandomBigramTrials> 0:
        results = []
        testNGrams(reportFile, ngmModels, noOfNgms, 2,results)
        trialResults.append(np.equal(results,i))
    ##### testing random sentences from the input files using bigram 
    if numRandomTrigramTrials > 0:
        results = []
        testNGrams(reportFile, ngmModels, noOfNgms, 3,results)
        trialResults.append(np.equal(results,i))

    #### test random sequences with bigram and trigram
    if numBigramSeqTrials >0:
        results = []
        testNGramsSeq(reportFile, ngmModels, noOfNgms, 2,results)


if numRandomBigramTrials >0:
    writeToReportFileNGram(reportFile, trialResults, languages, numRandomBigramTrials)

if numRandomTrigramTrials >0:
    writeToReportFileNGram(reportFile, trialResults, languages, numRandomTrigramTrials)
       
reportFile.close()

