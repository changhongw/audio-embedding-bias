import sys 
import os
base_dir = "/home/changhongw/embedding-bias-correction"
# add parental directory to Python path
sys.path.insert(0, base_dir)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from deem import deem

param_grid = {'LR_param': {'C':[10**k for k in range(-8, 4, 1)]}, 'scoring': 'roc_auc', 'cv': 3, 'random_state': 42}

with open(os.path.join(base_dir, "data/openmic-2018/openmic_classmap_10.json"), "r") as f: # only consider 10 classes of Openmic dataset
    openmic_class_map = json.load(f)
    
# use a dict to align the classes between Openmic dataset (key) and Irmas dataset (val)
with open(os.path.join(base_dir, "data/class_align.json"), "r") as f: 
    class_align = json.load(f)

with open(os.path.join(base_dir, "data/genre_map.json"), "r") as f: 
    genre_map = json.load(f)

with open(os.path.join(base_dir, "data/instrument_map.json"), "r") as f: 
    instrument_map = json.load(f)

embedding = 'vggish'
debias_method = ''  # no debiasing, i.e. use the original embedding

deb = deem(embedding = embedding, debias_method = debias_method, feature_dir='embeddings.h5', 
           instrument_map=instrument_map, genre_map=genre_map, param_grid=param_grid, class_align=class_align)

irmas_feature = deb.load_irmas()
openmic_feature = deb.load_openmic()

# deb.instrument_classfication(train_set='irmas', test_set='irmas', irmas_feature=irmas_feature, openmic_feature=openmic_feature)
deb.instrument_classfication(train_set='irmas', test_set='openmic', irmas_feature=irmas_feature, openmic_feature=openmic_feature)
deb.instrument_classfication(train_set='openmic', test_set='openmic', irmas_feature=irmas_feature, openmic_feature=openmic_feature)
deb.instrument_classfication(train_set='openmic', test_set='irmas', irmas_feature=irmas_feature, openmic_feature=openmic_feature)