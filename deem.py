

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

random.seed(42)

class deem():
    
    def __init__(self, embedding, feature_dir, instrument_map, genre_map, param_grid, debias_method):
        self.instrument_map = instrument_map 
        self.genre_map = genre_map
        self.param_grid = param_grid
        self.embedding = embedding
        self.debias_method = debias_method
        self.feature_dir = feature_dir
        # use a Pandas DataFrame to record all results and save into a csv file later
        self.result_all =pd.DataFrame({'instrument': [],
                          'train_set': [],
                          'test_set': [],
                          'precision': [],
                          'recall': [],
                          'f1-score': [],
                          'support': [],
                          'accuracy': [],
                          'roc_auc': [],
                          'ap': [],
                          'embedding': [],
                         }) 

    def load_feature(self, meta):

        embeddings = h5py.File(self.feature_dir, "r")

        ###### IRMAS data ######
        feature = np.array(embeddings["irmas"][self.embedding]["features"])
        keys_ori = np.array(embeddings["irmas"][self.embedding]["keys"])
        key_clip = np.unique(keys_ori)

        feature_clip = []

        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys_ori[:]==key,:],axis=0))
            
        feature_clip = np.array(feature_clip)
        print(feature_clip.shape, key_clip.shape)

        key_train = list(meta.loc[(meta['split'] == 'train')]['file_name'])
        key_test = list(meta.loc[(meta['split'] == 'test')]['file_name'])

        idx_train = [list(key_clip).index(item) for item in key_train]
        idx_test = [list(key_clip).index(item) for item in key_test]

        # cast the idx_* arrays to numpy structures
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        # use the split indices to partition the features, labels, and masks
        X_train = feature_clip[idx_train,:]
        X_test = feature_clip[idx_test]

        Y_train = meta.loc[(meta['split'] == 'train')]['instrument'].values
        Y_test = meta.loc[(meta['split'] == 'test')]['instrument'].values

        genre_train = meta.loc[(meta['split'] == 'train')]['genre'].values
        genre_test = meta.loc[(meta['split'] == 'test')]['genre'].values

        return (X_train, Y_train), (X_test, Y_test), (genre_train, genre_test)


    def instrument_classfication(self, train_set, test_set, A_feature, B_feature):
        if train_set == test_set and self.debias_method == '':
            globals()['models_' + train_set] = dict()
        elif train_set == test_set and self.debias_method == '-lda':
            globals()['LDAcoef_' + train_set] = dict()

        (X_train_A, Y_train_A), (X_test_A, Y_test_A), (genre_train_A, genre_test_A) = A_feature
        (X_train_B, Y_train_B), (X_test_B, Y_test_B), (genre_train_B, genre_test_B) = B_feature
        
        print('Train on {}, test on {}'.format(train_set, test_set))
        # We'll iterate over all istrument classes, and fit a model for each one
        # After training, we'll print a classification report for each instrument
        for instrument in tqdm(self.instrument_map):
            
            Y_train_A_inst = Y_train_A==instrument
            Y_train_B_inst = Y_train_B==instrument

            Y_test_A_inst = Y_test_A==instrument
            Y_test_B_inst = Y_test_B==instrument

            X_train_A_inst = X_train_A[Y_train_A_inst]
            X_train_B_inst = X_train_B[Y_train_B_inst]

            Y_train_A_noninst = Y_train_A!=instrument
            Y_train_B_noninst = Y_train_B!=instrument

            X_train_A_noninst = X_train_A[Y_train_A_noninst]
            X_train_B_noninst = X_train_B[Y_train_B_noninst]

            genre_train_A_inst = genre_train_A[Y_train_A_inst]
            genre_train_B_inst = genre_train_B[Y_train_B_inst]

            genre_train_A_noninst = genre_train_A[Y_train_A_noninst]
            genre_train_B_noninst = genre_train_B[Y_train_B_noninst]

            dim_inst = min(X_train_A_inst.shape[0], X_train_B_inst.shape[0])
            dim_noninst = min(X_train_A_noninst.shape[0], X_train_B_noninst.shape[0])

            X_train_A_inst, genre_train_A_inst = self.resample_data(X_train_A_inst, genre_train_A_inst, dim_inst)
            X_train_B_inst, genre_train_B_inst = self.resample_data(X_train_B_inst, genre_train_B_inst, dim_inst)

            X_train_A_noninst, genre_train_A_noninst = self.resample_data(X_train_A_noninst, genre_train_A_noninst, dim_noninst)
            X_train_B_noninst, genre_train_B_noninst = self.resample_data(X_train_B_noninst, genre_train_B_noninst, dim_noninst)

            X_train_A_clf = np.vstack((X_train_A_inst, X_train_A_noninst))
            X_train_B_clf = np.vstack((X_train_B_inst, X_train_B_noninst))

            Y_train_A_clf = np.hstack(([True] * len(X_train_A_inst), [False] * len(X_train_A_noninst))).reshape(-1,)
            Y_train_B_clf = np.hstack(([True] * len(X_train_B_inst), [False] * len(X_train_B_noninst))).reshape(-1,)

            if train_set == 'A' and test_set == 'A':
                X_train_clf, Y_train_clf = X_train_A_clf, Y_train_A_clf
                X_test_clf, Y_test_clf = X_test_A, Y_test_A_inst
            elif train_set == 'A' and test_set == 'B':
                X_train_clf, Y_train_clf = X_train_A_clf, Y_train_A_clf
                X_test_clf, Y_test_clf = X_test_B, Y_test_B_inst
            elif train_set == 'B' and test_set == 'B':
                X_train_clf, Y_train_clf = X_train_B_clf, Y_train_B_clf
                X_test_clf, Y_test_clf = X_test_B, Y_test_B_inst
            else:
                X_train_clf, Y_train_clf = X_train_B_clf, Y_train_B_clf
                X_test_clf, Y_test_clf = X_test_A, Y_test_A_inst

            if self.debias_method == '-lda':
                ############### LDA ###############
                X_train_conca = np.vstack((X_train_A_inst, X_train_B_inst))
                Y_A = np.zeros(len(X_train_A_inst))
                Y_B = np.ones(len(X_train_B_inst))
                Y_conca = np.hstack((Y_A, Y_B))

                LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                LDA.fit(X_train_conca, Y_conca)

                v = LDA.coef_.copy()
                v /= np.sqrt(np.sum(v**2))
                A = np.outer(v, v)

                X_train_clf = X_train_clf.copy().dot(np.eye(len(A)) - A)
                X_test_clf = X_test_clf.copy().dot(np.eye(len(A)) - A)

            # initialize and a logistic regression model
            LRmodel = LogisticRegression(random_state=self.param_grid['random_state'], penalty='l2', 
                                         solver='liblinear', class_weight='balanced')
            
            # hyperparameter tunning for logistic regression model
            clf =  GridSearchCV(LRmodel, param_grid=self.param_grid['LR_param'], cv=self.param_grid['cv'],
                                 scoring=self.param_grid['scoring'])    
            
            # fit the model
            clf.fit(X_train_clf, Y_train_clf)

            # predict
            Y_pred_clf = clf.predict(X_test_clf)
            # Get prediction scores for the positive class
            Y_pred_scores = clf.predict_proba(X_test_clf)[:, 1]
            
            if train_set == test_set and self.debias_method == '':
                globals()['models_' + train_set][instrument] = clf
            elif train_set == test_set and self.debias_method == '-lda':
                globals()['LDAcoef_' + train_set][instrument] = LDA.coef_.copy()           

            # print result for each instrument
            # print('-' * 52); print(instrument); print('\tTEST')
            # print(classification_report(Y_test_clf, Y_pred_clf))
            
            model_auc = roc_auc_score(Y_test_clf, Y_pred_scores)
            model_ap = average_precision_score(Y_test_clf, Y_pred_scores)
            # print(f'ROC-AUC = {model_auc:.3f}\t\tAP = {model_ap:.3f}')
            
            # record the result for each instrument
            report = pd.DataFrame(classification_report(Y_test_clf, Y_pred_clf, output_dict=True))['True']
            report['roc_auc'] = model_auc
            report['ap'] = model_ap
            report_accuracy = pd.DataFrame(classification_report(Y_test_clf, Y_pred_clf, output_dict=True))['accuracy'][-2]
            result_inst = [instrument, train_set, test_set, report['precision'], report['recall'],
                        report['f1-score'], report['support'], report_accuracy, model_auc, model_ap, self.embedding + self.debias_method]   
            self.result_all = self.result_all.append(pd.DataFrame(np.expand_dims(np.array(result_inst), axis=0), 
                                                        columns=self.result_all.columns), ignore_index=True)

        if train_set == test_set and self.debias_method == '':
            with open('models/models_' + train_set + '_' + self.embedding + self.debias_method + '.pickle', 'wb') as fdesc:
                pickle.dump(globals()['models_'+train_set], fdesc)
        elif train_set == test_set and self.debias_method == '-lda':
            with open('models/LDAcoef_' + train_set + '_' + self.embedding + self.debias_method + '.pickle', 'wb') as fdesc:
                pickle.dump(globals()['LDAcoef_'+train_set], fdesc) 



    def resample_data(self, feature, genre, num):
        """
        select "num" number of samples from original feature with the same genre distribution
        ------
        feature: original feature
        genre: original genre
        num: target number of samples
        """
        
        feature_all = feature[0,:]
        genre_all = genre[0]
        ratio =  num / len(feature)  # ratio between target and original => using this ratio to get sample from each genre
        
        for genre_item in self.genre_map:
            genre_target = genre == genre_item  

            len_target = genre_target.sum()  # number of samples of this specific genre
            idx_shuffle = random.sample(range(len_target), len_target)  # shuffle the idx to select randomly  
            idx_keep = idx_shuffle[:int(len_target * ratio)]  # select "total * ratio" samples from this genre

            feature_new = feature[genre_target][idx_keep]
            genre_new = genre[genre_target][idx_keep]

            feature_all = np.vstack((feature_all, feature_new))
            genre_all = np.hstack((genre_all, genre_new))

        return feature_all[1:,], genre_all[1:]