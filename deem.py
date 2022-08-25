import numpy as np
from tqdm import tqdm_notebook as tqdm
import json
import pandas as pd
import os
import h5py
from ast import literal_eval
import re
import pickle
import numpy.linalg as la
import collections
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn_extra.kernel_approximation import Fastfood
   
class deem():
    
    def __init__(self, embedding=None, classf_param=None, debias_method=''):
        self.embedding = embedding
        self.debias_method = debias_method
        self.classf_param = classf_param
        self.class_align = {'cello': 'cel',
           'clarinet': 'cla',
           'flute': 'flu',
           'guitar': ['gac', 'gel'],
           'organ': 'org',
           'piano': 'pia',
           'saxophone': 'sax',
           'trumpet': 'tru',
           'violin': 'vio',
           'voice': 'voi'}

        with open('class-map-10.json', 'r') as f: # only consider 10 classes of Openmic dataset
            self.class_map = json.load(f)
            
        self.result_all = pd.DataFrame({'instrument': [],
              'train_set': [],
              'test_set': [],
              'precision': [],
              'recall': [],
              'f1-score': [],
              'support': [],
              'accuracy': [],
              'roc_auc': [],
              'ap': []
             })
        
        
    def data_loader(self, dataset, data_root):

        embeddings = h5py.File('embeddings.h5', 'r')

        feature = np.array(embeddings[dataset][self.embedding]['features'])
        keys_ori = np.array(embeddings[dataset][self.embedding]['keys'])
        key_clip = np.unique(keys_ori)

        feature_clip = []
        print('Loading ' + dataset + ' data:')
        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys_ori[:]==key,:],axis=0))
        feature_clip = np.array(feature_clip)
        key_clip = np.array([str(k, 'utf-8') for k in key_clip])

        if dataset == 'irmas':
            key_train = set(pd.read_csv('irmas_train.csv', header=None, squeeze=True))
            key_test = set(pd.read_csv('irmas_test.csv', header=None, squeeze=True))

            keys = [key[key.index('[')+1:key.index(']')] for key in key_clip]
            for key in self.class_align:
                keys = [key if x in self.class_align[key] else x for x in keys]
            Y_true = np.array(keys)

        elif dataset == 'openmic':
            key_train = set(pd.read_csv(os.path.join(data_root, 'openmic2018_train.csv'), header=None, squeeze=True))
            key_test = set(pd.read_csv(os.path.join(data_root, 'openmic2018_test.csv'), header=None, squeeze=True))

            np_load_old = np.load   # save np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)   # modify the default parameters of np.load
            Ytrue = np.load(os.path.join(data_root, 'openmic-2018.npz'))['Y_true']
            Ymask = np.load(os.path.join(data_root, 'openmic-2018.npz'))['Y_mask']
            sample_key = np.load(os.path.join(data_root, 'openmic-2018.npz'))['sample_key']
            np.load = np_load_old   # restore np.load for future normal usage
            del(np_load_old)

            Y_true = []
            Y_mask = []

            for key in key_clip:
                Y_true.append(Ytrue[sample_key==key])
                Y_mask.append(Ymask[sample_key==key])

            Y_true = np.squeeze(np.array(Y_true))
            Y_mask = np.squeeze(np.array(Y_mask))

        else:
            raise RuntimeError('Unknown dataset: {}! Abort!'.format(dataset))


        # These loops go through all sample keys, and save their row numbers to either idx_train or idx_test
        idx_train, idx_test = [], []

        for idx, key in enumerate(key_clip):
            if key in key_train:
                idx_train.append(idx)
            elif key in key_test:
                idx_test.append(idx)
            else:
                raise RuntimeError('Unknown sample key={}! Abort!'.format(key))

        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        X_train = feature_clip[idx_train]
        X_test = feature_clip[idx_test]
        Y_true_train = Y_true[idx_train]
        Y_true_test = Y_true[idx_test]
            
        if dataset == 'irmas':
            return (X_train, Y_true_train), (X_test, Y_true_test)

        else: # dataset == 'openmic'
            Y_mask_train = Y_mask[idx_train]
            Y_mask_test = Y_mask[idx_test]

            return (X_train, Y_true_train), (X_test, Y_true_test), (Y_mask_train, Y_mask_test)
    
    
    def irmas_irmas(self, train_data, test_data):

        X_train, Y_true_train = train_data[0], train_data[1]
        X_test, Y_true_test = test_data[0], test_data[1]
        train_set_name, test_set_name = 'irmas', 'irmas'
        result_all = self.result_all
        
        # use a dictionary to include the classifier for each instrument trained on the dataset based on the embedding
        globals()['models_'+train_set_name] = dict()   

        # iterate over all istrument classes, and fit a model for each one
        for instrument in self.class_align:

            # get the training and testing labels for each instrument
            Y_true_train_inst = Y_true_train==instrument
            Y_true_test_inst = Y_true_test==instrument

            # initialize and a logistic regression model
            LRmodel = LogisticRegression(random_state=0, penalty='l2', solver='liblinear', class_weight='balanced')

            # hyperparameter tunning for logistic regression model
            param_grid = {'C': self.classf_param['C']}
            scoring = self.classf_param['metric']; 
            cv = self.classf_param['cv']
            clf =  GridSearchCV(LRmodel, param_grid=param_grid, cv=cv, scoring=scoring)    

            # fit the model
            clf.fit(X_train, Y_true_train_inst)

            # predict
            Y_pred_test_inst = clf.predict(X_test)

            # Get prediction scores for the positive class
            Y_pred_test_scores = clf.predict_proba(X_test)[:, 1]

            # print result for each instrument
#             print('-' * 52); print(instrument); print('\tTEST')
#             print(classification_report(Y_true_test_inst, Y_pred_test_inst))

            model_auc = roc_auc_score(Y_true_test_inst, Y_pred_test_scores)
            model_ap = average_precision_score(Y_true_test_inst, Y_pred_test_scores)
#             print(f'ROC-AUC = {model_auc:.3f}\t\tAP = {model_ap:.3f}')

            # store the classifier in the model dictionary
            globals()['models_'+train_set_name][instrument] = clf

            # record the result for each instrument
            report = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['True']
            report['roc_auc'] = model_auc
            report['ap'] = model_ap

            report_accuracy = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['accuracy'][-2]
            result_inst = [instrument, train_set_name, test_set_name, report['precision'], report['recall'],
                           report['f1-score'], report['support'], report_accuracy, model_auc, model_ap]
            result_all = result_all.append(pd.DataFrame(np.expand_dims(np.array(result_inst), axis=0), 
                                                        columns=result_all.columns), ignore_index=True)

        with open('models/models_' + train_set_name + '_' + self.embedding + self.debias_method + '.pickle', 'wb') as fdesc:
            pickle.dump(globals()['models_'+train_set_name], fdesc)

        return result_all


    def openmic_openmic(self, train_data, test_data, mask):

        X_train, Y_true_train = train_data[0], train_data[1]
        X_test, Y_true_test = test_data[0], test_data[1]
        Y_mask_train, Y_mask_test = mask[0], mask[1]
        train_set_name, test_set_name = 'openmic', 'openmic'
        result_all = self.result_all

        # use a dictionary to include the classifier for each instrument trained on the dataset based on the embedding
        globals()['models_'+train_set_name] = dict()   

        # This part of the code follows the baseline model for instrument recognition on the openmic dataset:
        # https://github.com/cosmir/openmic-2018/blob/master/examples/modeling-baseline.ipynb

        # We'll iterate over all istrument classes, and fit a model for each one
        # After training, we'll print a classification report for each instrument
        for instrument in self.class_align:

            # Map the instrument name to its column number
            inst_num = self.class_map[instrument]

            # First, sub-sample the data: we need to select down to the data for which we have annotations
            # This is what the mask arrays are for
            train_inst = Y_mask_train[:, inst_num]
            test_inst = Y_mask_test[:, inst_num]

            # Here, we're using the Y_mask_train array to slice out only the training examples
            # for which we have annotations for the given class
            X_train_inst = X_train[train_inst]

            # Again, we slice the labels to the annotated examples
            # We thresold the label likelihoods at 0.5 to get binary labels
            Y_true_train_inst = Y_true_train[train_inst, inst_num] >= 0.5

            # Repeat the above slicing and dicing but for the test set
            X_test_inst = X_test[test_inst]
            Y_true_test_inst = Y_true_test[test_inst, inst_num] >= 0.5

            # initialize and a logistic regression model
            LRmodel = LogisticRegression(random_state=0, penalty='l2', solver='liblinear', class_weight='balanced')

            # hyperparameter tunning for logistic regression model
            param_grid = {'C': self.classf_param['C']}
            scoring = self.classf_param['metric']; 
            cv = self.classf_param['cv']
            clf =  GridSearchCV(LRmodel, param_grid=param_grid, cv=cv, scoring=scoring)    

            # fit the model
            clf.fit(X_train_inst, Y_true_train_inst)

            # predict
            Y_pred_test_inst = clf.predict(X_test_inst)
            # Get prediction scores for the positive class
            Y_pred_test_scores = clf.predict_proba(X_test_inst)[:, 1]

            # print result for each instrument
#             print('-' * 52); print(instrument); print('\tTEST')
#             print(classification_report(Y_true_test_inst, Y_pred_test_inst))

            model_auc = roc_auc_score(Y_true_test_inst, Y_pred_test_scores)
            model_ap = average_precision_score(Y_true_test_inst, Y_pred_test_scores)
#             print(f'ROC-AUC = {model_auc:.3f}\t\tAP = {model_ap:.3f}')

            # store the classifier in the model dictionary
            globals()['models_'+train_set_name][instrument] = clf

            # record the result for each instrument
            report = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['True']
            report['roc_auc'] = model_auc
            report['ap'] = model_ap
            report_accuracy = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['accuracy'][-2]
            result_inst = [instrument, train_set_name, test_set_name, report['precision'], report['recall'],
                           report['f1-score'], report['support'], report_accuracy, model_auc, model_ap]   
            result_all = result_all.append(pd.DataFrame(np.expand_dims(np.array(result_inst), axis=0), 
                                                        columns=result_all.columns), ignore_index=True)

        with open('models/models_' + train_set_name + '_' + self.embedding + self.debias_method + '.pickle', 'wb') as fdesc:
            pickle.dump(globals()['models_'+train_set_name], fdesc)

        return result_all


    def irmas_openmic(self, train_data, test_data, mask):

        X_train, Y_true_train = train_data[0], train_data[1]
        X_test, Y_true_test = test_data[0], test_data[1]
        Y_mask_train, Y_mask_test = mask[0], mask[1]
        train_set_name, test_set_name = 'irmas', 'openmic' 
        result_all = self.result_all
        
        globals()['models_'+train_set_name] = pickle.load(open('models/models_' + train_set_name + 
                                                               '_' + self.embedding + self.debias_method + '.pickle', 'rb'))

        # iterate over all istrument classes, and fit a model for each one
        for instrument in self.class_align:

            # Map the instrument name to its column number
            inst_num = self.class_map[instrument]

            # First, sub-sample the data: we need to select down to the data for which we have annotations 
            # This is what the mask arrays are for
            test_inst = Y_mask_test[:, inst_num]

            # Repeat the above slicing and dicing but for the test set
            X_test_inst = X_test[test_inst]
            Y_true_test_inst = Y_true_test[test_inst, inst_num] >= 0.5

            # evaluate the classifier 
            Y_pred_test_inst =  globals()['models_'+train_set_name][instrument].predict(X_test_inst)
            Y_pred_test_scores =  globals()['models_'+train_set_name][instrument].predict_proba(X_test_inst)[:, 1]

            # print result for each instrument
#             print('-' * 52); print(instrument); print('\tTEST')
#             print(classification_report(Y_true_test_inst, Y_pred_test_inst))

            model_auc = roc_auc_score(Y_true_test_inst, Y_pred_test_scores)
            model_ap = average_precision_score(Y_true_test_inst, Y_pred_test_scores)
#             print(f'ROC-AUC = {model_auc:.3f}\t\tAP = {model_ap:.3f}')

            # record the result for each instrument
            report = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['True']
            report['roc_auc'] = model_auc
            report['ap'] = model_ap
            report_accuracy = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['accuracy'][-2]
            result_inst = [instrument, train_set_name, test_set_name, report['precision'], report['recall'],
                           report['f1-score'], report['support'], report_accuracy, model_auc, model_ap]   
            result_all = result_all.append(pd.DataFrame(np.expand_dims(np.array(result_inst), axis=0), 
                                                        columns=result_all.columns), ignore_index=True)

        return result_all  


    def openmic_irmas(self, train_data, test_data):

        X_train, Y_true_train = train_data[0], train_data[1]
        X_test, Y_true_test = test_data[0], test_data[1]
        train_set_name, test_set_name = 'openmic', 'irmas' 
        result_all = self.result_all
        
        globals()['models_'+train_set_name] = pickle.load(open('models/models_' + train_set_name + 
                                                               '_' + self.embedding + self.debias_method + '.pickle', 'rb'))

        # iterate over all istrument classes, and fit a model for each one
        for instrument in self.class_align:

            # get the training and testing labels for each instrument
            Y_true_test_inst = Y_true_test==instrument

            # evaluate the classifier
            Y_pred_test_inst =  globals()['models_'+train_set_name][instrument].predict(X_test)
            Y_pred_test_scores =  globals()['models_'+train_set_name][instrument].predict_proba(X_test)[:, 1]

            # print result for each instrument
#             print('-' * 52); print(instrument); print('\tTEST')
#             print(classification_report(Y_true_test_inst, Y_pred_test_inst))

            model_auc = roc_auc_score(Y_true_test_inst, Y_pred_test_scores)
            model_ap = average_precision_score(Y_true_test_inst, Y_pred_test_scores)
#             print(f'ROC-AUC = {model_auc:.3f}\t\tAP = {model_ap:.3f}')

            # record the result for each instrument
            report = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['True']
            report['roc_auc'] = model_auc
            report['ap'] = model_ap
            report_accuracy = pd.DataFrame(classification_report(Y_true_test_inst, Y_pred_test_inst, output_dict=True))['accuracy'][-2]
            result_inst = [instrument, train_set_name, test_set_name, report['precision'], report['recall'],
                           report['f1-score'], report['support'], report_accuracy, model_auc, model_ap]   
            result_all = result_all.append(pd.DataFrame(np.expand_dims(np.array(result_inst), axis=0), 
                                                        columns=result_all.columns), ignore_index=True)

        return result_all  
    
    
    def projection(self, train_data, test_data):
        "project out dataset separation direction (already extracted)"
        
        X_train, X_test = train_data[0], test_data[0]
        embedding = self.embedding
        
        if self.debias_method != '':

            # load separation direction
            if '-k' in self.debias_method:
                
                # standardize embedding
                file = open('models/standScaler_' + embedding + '.pickle', 'rb')
                Scaler = pickle.load(file)
                file.close()
                X_train = Scaler.transform(X_train)
                X_test = Scaler.transform(X_test)

                # kernelize embedding with fastfood
                file = open('models/kernelizer_' + embedding + '.pickle', 'rb')
                Sampler = pickle.load(file)
                file.close()
                X_train = Sampler.transform(X_train)
                X_test = Sampler.transform(X_test)

                if 'lda' in self.debias_method and 'genre' in self.debias_method: # -klda-genre
                    # dataset separation LDA coefficients
                    file = open('models/LDAcoef_' + embedding + '-klda-genre.pickle', 'rb')
                    globals()['LDAcoef_' + embedding] = pickle.load(file)
                    file.close()
                elif 'lda' in self.debias_method: # -klda
                    # dataset separation LDA coefficients
                    file = open('models/LDAcoef_' + embedding + '-klda.pickle', 'rb')
                    globals()['LDAcoef_' + embedding] = pickle.load(file)
                    file.close()

            elif 'genre' in self.debias_method:  # -lda-genre
                file = open('models/LDAcoef_' + embedding + '-lda-genre.pickle', 'rb')
                globals()['LDAcoef_' + embedding] = pickle.load(file)
                file.close()
            else:  # -lda
                file = open('models/LDAcoef_' + embedding + '-lda.pickle', 'rb')
                globals()['LDAcoef_' + embedding] = pickle.load(file)
                file.close()

        # project out separation direction
        if 'lda' in self.debias_method and 'genre' in self.debias_method:
            W = globals()['LDAcoef_' + embedding]
            U, s, V = la.svd(W, full_matrices=False)
            A = np.dot(V.T, V)
            X_train = X_train.dot(np.eye(len(A)) - A)
            X_test = X_test.dot(np.eye(len(A)) - A)

        elif 'lda' in self.debias_method and 'genre' not in self.debias_method:
            v = globals()['LDAcoef_' + embedding]
            v /= np.sqrt(np.sum(v**2))
            A = np.outer(v, v)
            X_train = X_train.dot(np.eye(len(A)) - A)
            X_test = X_test.dot(np.eye(len(A)) - A)
        
        return X_train, X_test
           