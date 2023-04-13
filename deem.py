import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import h5py
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ast import literal_eval
import re
import json
import numpy.linalg as la

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.metrics import pairwise_distances

import warnings
warnings.filterwarnings('ignore')

class deem():
    
    def __init__(self, embedding, debias_method, feature_dir, instrument_map, genre_map, param_grid):
        self.embedding = embedding
        self.debias_method = debias_method
        self.feature_dir = feature_dir
        self.instrument_map = instrument_map 
        self.genre_map = genre_map
        self.param_grid = param_grid
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

    def load_feature(self, meta, part):

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

        key_train = list(meta.loc[(meta['subset'] == 'train') & (meta['part'] == part)]['file_name'])
        key_test = list(meta.loc[(meta['subset'] == 'test') & (meta['part'] == part)]['file_name'])

        idx_train = [list(key_clip).index(item) for item in key_train]
        idx_test = [list(key_clip).index(item) for item in key_test]

        # cast the idx_* arrays to numpy structures
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        # use the split indices to partition the features, labels, and masks
        X_train = feature_clip[idx_train,:]
        X_test = feature_clip[idx_test]

        Y_train = meta.loc[(meta['subset'] == 'train') & (meta['part'] == part)]['instrument'].values
        Y_test = meta.loc[(meta['subset'] == 'test') & (meta['part'] == part)]['instrument'].values

        genre_train = meta.loc[(meta['subset'] == 'train') & (meta['part'] == part)]['genre'].values
        genre_test = meta.loc[(meta['subset'] == 'test') & (meta['part'] == part)]['genre'].values

        return (X_train, Y_train), (X_test, Y_test), (genre_train, genre_test)


    def instrument_classfication(self, train_set, test_set, A_feature, B_feature):
        if train_set == test_set and (self.debias_method == '' or self.debias_method == '-k'):
            globals()['models_' + train_set] = dict()
        elif 'lda' in self.debias_method:
            globals()['LDAcoef_' + train_set] = dict()

        (X_train_A, Y_train_A), (X_test_A, Y_test_A), (genre_train_A, genre_test_A) = A_feature
        (X_train_B, Y_train_B), (X_test_B, Y_test_B), (genre_train_B, genre_test_B) = B_feature
        
        print('Train on {}, test on {}'.format(train_set, test_set))

        for instrument in tqdm(self.instrument_map):
            
            Y_train_A_inst = Y_train_A==instrument
            Y_train_B_inst = Y_train_B==instrument

            Y_test_A_inst = Y_test_A==instrument
            Y_test_B_inst = Y_test_B==instrument

            X_train_A_inst = X_train_A[Y_train_A_inst]
            X_train_B_inst = X_train_B[Y_train_B_inst]

            X_test_A_inst = X_test_A
            X_test_B_inst = X_test_B

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

            genre_train_A_clf = np.hstack((genre_train_A_inst, genre_train_A_noninst))
            genre_train_B_clf = np.hstack((genre_train_B_inst, genre_train_B_noninst))

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

            if 'k' in self.debias_method:
            
                scaler = StandardScaler()
                X_train_A_clf = scaler.fit_transform(X_train_A_clf)
                X_train_B_clf = scaler.fit_transform(X_train_B_clf)
                X_train_all = np.vstack((X_train_A_clf, X_train_B_clf))

                # kernelize embedding with fastfood
                Sampler = Fastfood(n_components=4*X_train_all.shape[1], random_state=self.param_grid['random_state'],
                                                sigma=np.median(pairwise_distances(X_train_all, metric='l2')))
                X_train_all = Sampler.fit_transform(X_train_all)  
                X_train_A_clf = X_train_all[:len(X_train_A_clf)]
                X_train_B_clf  = X_train_all[len(X_train_A_clf):]

                scaler = StandardScaler()
                X_train_clf = scaler.fit_transform(X_train_clf)  
                X_test_clf = scaler.transform(X_test_clf)       
                X_test_clf = np.nan_to_num(X_test_clf, np.nan)  

                X_train_clf = Sampler.transform(X_train_clf)
                X_test_clf  = Sampler.transform(X_test_clf)

            ############### LDA ###############
            # project the separation direction of the instrument class
            X_train_conca = np.vstack((X_train_A_clf, X_train_B_clf))  
            genre_train_conca = np.hstack((genre_train_A_clf, genre_train_B_clf))
            Y_A = np.zeros(len(X_train_A_clf))
            Y_B = np.ones(len(X_train_B_clf))
            Y_conca = np.hstack((Y_A, Y_B))

            if 'lda' in self.debias_method and '-m' in self.debias_method:
                genre_LDAcoef = []
                for genre in self.genre_map:
                    
                    X_train_conca_sub = X_train_conca[genre_train_conca == genre] 
                    Y_conca_sub = Y_conca[genre_train_conca == genre]  

                    LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                    LDA.fit(X_train_conca_sub, Y_conca_sub)

                    genre_LDAcoef.append(LDA.coef_.copy())

                genre_LDAcoef = np.squeeze(np.array(genre_LDAcoef))

                W = genre_LDAcoef.copy()
                U, s, V = la.svd(W, full_matrices=False)
                A = np.dot(V.T, V)

                X_train_clf = X_train_clf.copy().dot(np.eye(len(A)) - A)
                X_test_clf = X_test_clf.copy().dot(np.eye(len(A)) - A)

                globals()['LDAcoef_' + train_set][instrument] = genre_LDAcoef.copy() 

            elif 'lda' in self.debias_method:
                LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                LDA.fit(X_train_conca, Y_conca)

                v = LDA.coef_.copy()
                v /= np.sqrt(np.sum(v**2))
                A = np.outer(v, v)

                X_train_clf = X_train_clf.copy().dot(np.eye(len(A)) - A)
                X_test_clf = X_test_clf.copy().dot(np.eye(len(A)) - A)

                globals()['LDAcoef_' + train_set][instrument] = LDA.coef_.copy()  


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
            
            if train_set == test_set and (self.debias_method == '' or self.debias_method == '-k'):
                globals()['models_' + train_set][instrument] = clf         

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

        if train_set == test_set and (self.debias_method == '' or self.debias_method == '-k'):
            with open('models/models_' + train_set + '_' + self.embedding + self.debias_method + '.pickle', 'wb') as fdesc:
                pickle.dump(globals()['models_'+train_set], fdesc)
        elif train_set == test_set and 'lda' in self.debias_method:
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
        random.seed(self.param_grid['random_state'])
        
        ratio =  num / len(feature)  # ratio between target and original => using this ratio to get sample from each genre
        
        idx_shuffle = random.sample(range(feature.shape[0]), feature.shape[0])  # shuffle the idx to select randomly  
        idx_keep = idx_shuffle[:int(feature.shape[0] * ratio)]  # select "total * ratio" samples from this genre

        feature = feature[idx_keep,:]
        genre = genre[idx_keep]

        return feature, genre