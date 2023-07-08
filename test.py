import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
from tqdm import tqdm
import json
import pandas as pd
import pickle
import os
import h5py
import collections as cl
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import random
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from deem import deem


param_grid = {'LR_param': {'C':[10**k for k in range(-8, 4, 1)]}, 'scoring': 'roc_auc', 'cv': 3, 'random_state': 42}

with open("genre_map.json", "r") as f: # only consider 10 classes of Openmic dataset
    genre_map = json.load(f)
aligned_genre = list(genre_map)[:-1]

with open("instrument_map.json", "r") as f: # only consider 10 classes of Openmic dataset
    instrument_map = json.load(f)

embedding = 'vggish'
debias_method = '-mklda'

deb = deem(embedding = embedding, debias_method = debias_method, feature_dir='./embeddings.h5', 
           instrument_map=instrument_map, genre_map=genre_map, param_grid=param_grid)

meta_all = pd.read_csv("irmas_meta_all.csv")
A_feature = deb.load_feature(meta_all, 'A')
B_feature = deb.load_feature(meta_all, 'B')

deb.instrument_classfication(train_set='A', test_set='A', A_feature=A_feature, B_feature=B_feature)
deb.instrument_classfication(train_set='A', test_set='B', A_feature=A_feature, B_feature=B_feature)
deb.instrument_classfication(train_set='B', test_set='B', A_feature=A_feature, B_feature=B_feature)
deb.instrument_classfication(train_set='B', test_set='A', A_feature=A_feature, B_feature=B_feature)

deb.result_all.to_csv('results/results_' + embedding + debias_method + '.csv')
result_all = deb.result_all

# embedding = 'vggish'
# debias_method = '-lda'

# deb.debias_method = debias_method

# deb.instrument_classfication(train_set='A', test_set='A', A_feature=A_feature, B_feature=B_feature)
# deb.instrument_classfication(train_set='A', test_set='B', A_feature=A_feature, B_feature=B_feature)
# deb.instrument_classfication(train_set='B', test_set='B', A_feature=A_feature, B_feature=B_feature)
# deb.instrument_classfication(train_set='B', test_set='A', A_feature=A_feature, B_feature=B_feature)

# deb.result_all.to_csv('results/results_' + embedding + debias_method + '.csv')
# result_all = result_all.append(deb.result_all)