"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

from typing import List

import numpy as np
from utils import utils
from utils.utils import Puzzle
import json
import math

import scipy.linalg

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20
K = 4


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """reduce_dimensions
    
    Takes the raw feature vectors and reduces them down to the required number of
    dimensions and uses the eigenvectors in the model, calculated in process_training_data, to
    reduce the features to desired size.
    
    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #finds deviation of data
    data_mean = np.mean(data, axis=0)
    data_c = data - data_mean

    #uses pca to reduce feature vectors with eigenvectors
    reduced_data = data_c @ np.array(model["vectors"])

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """process_training_data

    This processes the models training data by finding the eigenvectors of the training data
    selecting the top 20, storing them to the model and using these to cut down the features
    of the training data. Stores training labels in the model and calculate a convolution table
    going over the training data once features have been reduced to be used in best match 
    classification later on.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    #gets eigen vectors of covariance matrix of features
    featureVectors_mean = np.mean(fvectors_train, axis=0)
    featureVectors_c = fvectors_train - featureVectors_mean
    covx = np.cov(featureVectors_c, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, subset_by_index=(N - N_DIMENSIONS, N - 1))
    v = np.fliplr(v)

    #builds model
    model = {}
    print(v.shape)
    model["vectors"] = v.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    #fvectors_train_reduced ,labels_train = condensed_NN(fvectors_train_reduced,labels_train,K)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    model["labels_train"] = labels_train.tolist()

    #build confusions table for use in best match classification
    training_data_labels = KNN_batch(fvectors_train_reduced,labels_train,fvectors_train_reduced,K)
    confusions = np.zeros((26, 26))
    for i in range(len(training_data_labels)):
        confusions[ord(model["labels_train"][i])-65, ord(training_data_labels[i])-65] += 1

    model["confusions"] = confusions.tolist()
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """classify_squares

    In this function K nearest neighbor is used with dimension reduced 
    training data to assign labels to this test data. With these labels 
    then passed out in a list.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    #reduces features(used in early testing not necessary when using evaluate.py)
    #fvectors_test = reduce_dimensions(fvectors_test,model)

    #classifies features
    return KNN_batch(np.array(model["fvectors_train"]),np.array(model["labels_train"]),fvectors_test,K)


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """find_words

    This function takes a 2D array of partially correctly identified letters and a list
    of words for which to search the array for. This is achieved through searching the 
    array for the closest match. Closest match is defined as a word of the same length 
    that takes the least number of substituted letters to be identical, minus the fraction 
    of letters mistaken for the incorrect letter when classifying the correct letter in the 
    confusion table, for each incorrect letter.

    This way more common slipups by the classifier such as mistaking a m or an n are not 
    penalized as harshly.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    y, x = labels.shape

    #creates list for recording the position and the quality of the best match for each word
    results =  [(0,0,0,0) for i in range(len(words))]
    bestMatch = [0 for i in range(len(words))]

    for i in range(y * x):#iterates through 2D array
        ix = i // y
        iy = i % y
        for direct in range(9):#checks for words in all 8 directions from starting letter
            dx = (direct // 3)-1
            dy = (direct % 3)-1
            count = len(min(words, key=len))-1
            nwords = words
            if dx != 0 or dy != 0:
                #checks through all relevant lengths from start position for matches
                while len(max(words, key=len)) > (count) and inBounds(ix+count*dx,iy+count*dy,x,y):
                    nwords = list(filter(lambda word: len(word) == (count+1), words))
                    for word in nwords:#checks all words of same length for better match
                        ind = words.index(word)
                        #if better match reassigns position and value
                        xEnd = ix+count*dx
                        yEnd = iy+count*dy
                        similarity = similarity2(word.upper(), extractWord(ix,iy,xEnd,yEnd,labels),model)
                        if similarity > bestMatch[ind]:
                            bestMatch[ind] = similarity
                            results[ind] = (iy,ix,(yEnd),(xEnd))
                    count += 1

    return results

def inBounds(x,y,xBound,yBound):
    return x >= 0 and y >= 0 and x < xBound and y < yBound

#checks how similar two string are
def similarity(str1,str2):
    count = 0
    for i in range(len(str1)):#measures substitutions
        if str1[i] == str2[i]:
            count += 1

    return count / len(str1)

#checks how similar two string are
def similarity2(str1,str2,model):
    count = 0
    con = model["confusions"]
    #measures substitutions and likelihood that incorrect letters are matches
    for i in range(len(str1)):
        if str1[i] == str2[i]:
            count += 1
        else:
            count += con[ord(str1[i])-65][ord(str2[i])-65] / sum(con[ord(str1[i])-65])

    return count / len(str1)

#extracts a string from a 2D array of chars by its coordinates
def extractWord(x1,y1,x2,y2,grid):
    xs = list(range(x1,(x2 + int(math.copysign(1,x2-x1))),int(math.copysign(1,x2-x1))))
    ys = list(range(y1,(y2+int(math.copysign(1,y2-y1))),int(math.copysign(1,y2-y1))))
    string = ""
    if len(xs) == 1:
        for y in ys:
            string += grid[y,x1]
    elif len(ys) == 1:
        for x in xs:
            string += grid[y1,x]
    else:
        for i in range(len(xs)):
            string += grid[ys[i],xs[i]]

    return string
        


#get the distances between two (n)D points
def distances(X_train, test_point):
    distances = np.sqrt(np.sum(np.square((X_train - test_point)),axis=1))
    return distances

#uses the K nearest points in training data to classify a point
def KNN(X_train, y_train, test_point, K = 1):
    dist = distances(X_train, test_point)
    ind= np.argsort(dist)
    dist = dist[ind]
  
    y_train = y_train[ind]
    nearest_labels = y_train[0:K]

    values, counts = np.unique(nearest_labels, return_counts=True)
    label = values[np.argmax(counts)]
    
    return str(label)

def KNN_batch(trainingData,trainingLabels,test_points,K=1):#executes KNN over a batch of inputs
    labels = []
    for i in range(test_points.shape[0]):
        labels.append(KNN(trainingData, trainingLabels, test_points[i], K))
    return labels

def condensed_NN(X_train, y_train, K = 1):
    # Randomly permute our data indices to 
    # avoid any ordering problems
    inds = np.random.permutation(len(X_train))
    X_train = X_train[inds]
    y_train = y_train[inds]
    
    # Take the first K samples for the condensed store
    


    Sx = np.array(X_train[0:2])
    Sy = np.array(y_train[0:2])
    #print(Sx)
    #print(np.sum(Sx,axis=1))
    
    X_train = np.delete(X_train,0,axis= 0)
    y_train = np.delete(y_train,0,axis = 0)
    #print(y_train)

    # Repeat until no further changes
    changed = True
    while changed:

        # Keep track of any removed indices
        # We won't do this within the loop to avoid changing 
        # the list that we are working on.
        changed = False
        

        # TODO: Implement step 3 of the algorithm
        for x in range(len(X_train)):
            if KNN(Sx,Sy,X_train[x],K) != y_train[x]:
                Sx = np.append(Sx,[X_train[x]],axis=0)
                Sy = np.append(Sy,[y_train[x]],axis=0)

                X_train = np.delete(X_train,x,axis=0)
                y_train = np.delete(y_train,x,axis=0)
                changed = True
                
                break
            
        # Remove samples added to S from the remaining list
        #inds = np.delete(inds, removed)

    return Sx, Sy