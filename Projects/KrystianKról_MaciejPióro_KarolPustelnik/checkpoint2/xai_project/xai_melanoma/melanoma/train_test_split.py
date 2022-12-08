import pandas as pd
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
import scipy
from scipy import stats


melanoma = pd.read_csv('/home/kpusteln/melanoma/train_concat.csv')

def probability_mass(data):
    
    counts = Counter(data) # counting the classes
    total = sum(counts.values()) # total number of classes
    probability_mass = {k:v/total for k,v in counts.items()} # probability mass of the classes
    probability_mass = list(probability_mass.values()) # converting the dictionary to a list
    return probability_mass
    

def train_test_split(data, train_size = 0.8, precision = 0.005):
    """splitting data into train and test sets keeping the same distribution of classes using wasertein's method
    args: data - data frame containing the data
    train_size - size of the train set default
    precision - determines how close the train set size is to the train_size default 0.005 (the smaller the better, but it may take longer to generate sets)"""
    
    print('Splitting data into train and test sets...')
    
    data = pd.read_csv(data) # loading the data
    wass_dist = 1
    wass_dist2 = 1
    train_size = int(train_size * len(data)) # calculating the number of videos in the train set
    indexes = list(range(len(data))) # list of indexes of the data
    while (wass_dist > precision) and (wass_dist2 > precision): # while the wasserstein distance is greater than 0.005
        train = random.sample(indexes, train_size) # sampling the train set
        test_val = [x for x in indexes if x not in train] # sampling the test set
        test = random.sample(test_val, int(len(test_val)/2)) # sampling the validation set
        
        val = [x for x in test_val if x not in test] # sampling the validation set
        train_set = data.iloc[train] # creating the train set
        test_set = data.iloc[test] # creating the test set
        val_set = data.iloc[val] # creating the validation set
        probability_mass_train = probability_mass(train_set['target']) # calculating the probability mass of the train set
        probability_mass_test = probability_mass(test_set['target']) # calculating the probability mass of the test set
        probability_mass_val = probability_mass(val_set['target']) # calculating the probability mass of the validation set
        wass_dist = stats.wasserstein_distance(probability_mass_train, probability_mass_test) # wasserstein distance between distributions
        wass_dist2 = stats.wasserstein_distance(probability_mass_train, probability_mass_val) # wasserstein distance between distributions
    
    print('Done!')
    return train_set, val_set, test_set

train_set, val_set, test_set = train_test_split('/home/kpusteln/melanoma/train_concat.csv')

train_set.to_csv('/home/kpusteln/melanoma/train_set.csv', index = False)
val_set.to_csv('/home/kpusteln/melanoma/val_set.csv', index = False)
test_set.to_csv('/home/kpusteln/melanoma/test_set.csv', index = False)