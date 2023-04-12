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

random.seed(42)

class deem():
    
    def __init__(self, embedding, feature_dir, instrument_map, genre_map, param_grid, debias_method, class_align=None):
        self.instrument_map = instrument_map 
        self.genre_map = genre_map
        self.param_grid = param_grid
        self.embedding = embedding
        self.debias_method = debias_method
        self.feature_dir = feature_dir
        self.class_align = class_align
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

    def list_match(self, A, B):
        ele_A = set(map(str.lower, A))
        ele_B = set(map(str.lower, B))
        return len(ele_A.intersection(ele_B)) > 0

    def load_irmas(self):

        embeddings = h5py.File(self.feature_dir, "r")

        ###### IRMAS data ######
        feature = np.array(embeddings["irmas"][self.embedding]["features"])
        keys_ori = np.array(embeddings["irmas"][self.embedding]["keys"])
        print(feature.shape, keys_ori.shape)

        key_clip = np.unique(keys_ori)
        print(key_clip.shape)

        feature_clip = []

        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys_ori[:]==key,:],axis=0))
            
        feature_clip = np.array(feature_clip)
        print(feature_clip.shape, key_clip.shape)

        genre_clip = [item[item.rindex("[")+1:item.rindex("]")] for item in key_clip]
        genre_clip = np.array(genre_clip)

        key_train = set(pd.read_csv("irmas_train.csv", header=None, squeeze=True))
        key_test = set(pd.read_csv("irmas_test.csv", header=None, squeeze=True))

        # these loops go through all sample keys, and save their row numbers to either idx_train or idx_test
        idx_train, idx_test = [], []

        for k in range(len(key_clip)):
            if str(key_clip[k]) in key_train:
                idx_train.append(k)
            elif str(key_clip[k]) in key_test:
                idx_test.append(k)
            else:
                # This should never happen, but better safe than sorry.
                raise RuntimeError("Unknown sample key={}! Abort!".format(str(key_clip[k])))
                
        # cast the idx_* arrays to numpy structures
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        keys = np.array(key_clip)
        keys = [key[key.index("[")+1:key.index("]")] for key in keys]

        for key in self.class_align:
            keys = [key if x in self.class_align[key] else x for x in keys]
            
        keys = np.array(keys)
        np.unique(keys)

        # use the split indices to partition the features, labels, and masks
        X_train_irmas = feature_clip[idx_train,:]
        X_test_irmas = feature_clip[idx_test]

        Y_train_irmas = keys[idx_train]
        Y_test_irmas = keys[idx_test]

        genre_train_irmas = genre_clip[idx_train]
        genre_test_irmas = genre_clip[idx_test]

        genre_train_irmas = ["pop_rock" if item =="pop_roc" else item for item in genre_train_irmas]
        genre_train_irmas = ["jazz_blue" if item =="jaz_blu" else item for item in genre_train_irmas]
        genre_train_irmas = ["classical" if item =="cla" else item for item in genre_train_irmas]
        genre_train_irmas = ["country_folk" if item =="cou_fol" else item for item in genre_train_irmas]

        genre_test_irmas = ["pop_rock" if item =="pop_roc" else item for item in genre_test_irmas]
        genre_test_irmas = ["jazz_blue" if item =="jaz_blu" else item for item in genre_test_irmas]
        genre_test_irmas = ["classical" if item =="cla" else item for item in genre_test_irmas]
        genre_test_irmas = ["country_folk" if item =="cou_fol" else item for item in genre_test_irmas]

        genre_train_idx = np.array(genre_train_irmas) != "lat_sou"
        genre_test_idx = np.array(genre_test_irmas) != "lat_sou"

        genre_train_irmas = np.array(genre_train_irmas)[genre_train_idx]
        genre_test_irmas = np.array(genre_test_irmas)[genre_test_idx]

        return (X_train_irmas[genre_train_idx], Y_train_irmas[genre_train_idx]), \
            (X_test_irmas[genre_test_idx], Y_test_irmas[genre_test_idx]), (genre_train_irmas, genre_test_irmas)
    

    def load_openmic(self):

        with open("openmic_classmap_10.json", "r") as f: # only consider 10 classes of Openmic dataset
            self.openmic_map = json.load(f)

        embeddings = h5py.File(self.feature_dir, "r")

        ###### OpenMIC datat ######
        feature = np.array(embeddings["openmic"][self.embedding]["features"])
        keys = np.array(embeddings["openmic"][self.embedding]["keys"])
        print(feature.shape, keys.shape)

        key_clip = np.unique(keys)

        feature_clip = []

        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys[:]==key,:],axis=0))
            
        feature_clip = np.array(feature_clip)
        print(feature_clip.shape, key_clip.shape)

        # key-label map using the information from the dataset source
        data_root = "openmic-2018/"

        np_load_old = np.load   # save np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)   # modify the default parameters of np.load

        Ytrue = np.load(os.path.join(data_root, "openmic-2018.npz"))["Y_true"]
        Ymask = np.load(os.path.join(data_root, "openmic-2018.npz"))["Y_mask"]
        sample_key = np.load(os.path.join(data_root, "openmic-2018.npz"))["sample_key"]

        np.load = np_load_old   # restore np.load for future normal usage
        del(np_load_old)

        print(Ytrue.shape, Ymask.shape, sample_key.shape)

        Y_true = []
        Y_mask = []

        for key in tqdm(key_clip):
            Y_true.append(Ytrue[sample_key==key])
            Y_mask.append(Ymask[sample_key==key])
            
        Y_true = np.squeeze(np.array(Y_true))
        Y_mask = np.squeeze(np.array(Y_mask))

        print(feature_clip.shape, Y_true.shape, Y_mask.shape)

        # train-test split
        train_set = set(pd.read_csv(data_root + "openmic2018_train.csv", header=None, squeeze=True))
        test_set = set(pd.read_csv(data_root + "openmic2018_test.csv", header=None, squeeze=True))
        print("# Train: {},  # Test: {}".format(len(train_set), len(test_set)))

        idx_train, idx_test = [], []

        for idx, n in enumerate(key_clip):
            if n in train_set:
                idx_train.append(idx)
            elif n in test_set:
                idx_test.append(idx)
            else:
                raise RuntimeError("Unknown sample key={}! Abort!".format(key_clip[n]))
                
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        meta = pd.read_csv(data_root + "openmic-2018-metadata.csv")
        train_genre_meta = list(meta["track_genres"][idx_train])
        test_genre_meta = list(meta["track_genres"][idx_test])
        len(train_genre_meta), len(test_genre_meta)

        idx_genre_train = []
        genre_train_openmic = []

        for k in tqdm(range(len(train_genre_meta))):
            if isinstance(train_genre_meta[k], str):
                idx_genre_train.append(k)
                genre_excerpt = literal_eval(train_genre_meta[k])
                genre_train_openmic.append([item['genre_title']for item in genre_excerpt])

        idx_genre_train = np.array(idx_genre_train)

        idx_genre_test = []
        genre_test_openmic = []

        for k in tqdm(range(len(test_genre_meta))):
            if isinstance(test_genre_meta[k], str):
                idx_genre_test.append(k)
                genre_excerpt = literal_eval(test_genre_meta[k])
                genre_test_openmic.append([item['genre_title']for item in genre_excerpt])

        idx_genre_test = np.array(idx_genre_test)

        genre_train_final = []
        for genre_excerpt in genre_train_openmic:
            item_genre = []
            for item in genre_excerpt:
                for genre in self.genre_map:
                    if self.list_match(re.split("[^a-zA-Z]", item), re.split("[^a-zA-Z]", genre)):
                        item_genre.append(genre)
            if len(item_genre) > 0:
                genre_train_final.append(item_genre[0])
            else:
                genre_train_final.append(genre_excerpt[0])
                
        genre_test_final = []
        for genre_excerpt in genre_test_openmic:
            item_genre = []
            for item in genre_excerpt:
                for genre in self.genre_map:
                    if self.list_match(re.split("[^a-zA-Z]", item), re.split("[^a-zA-Z]", genre)):
                        item_genre.append(genre)
            if len(item_genre) > 0:
                genre_test_final.append(item_genre[0])
            else:
                genre_test_final.append(genre_excerpt[0])

        genre_train_openmic, genre_test_openmic = genre_train_final, genre_test_final

        genre_train_openmic = np.array([item if item in self.genre_map else "other" for item in genre_train_openmic])
        genre_test_openmic = np.array([item if item in self.genre_map else "other" for item in genre_test_openmic])

        X_train_openmic = feature_clip[idx_train,:][idx_genre_train,:]
        X_test_openmic = feature_clip[idx_test,:][idx_genre_test,:]

        Y_train_openmic = Y_true[idx_train,:][idx_genre_train,:]
        Y_test_openmic = Y_true[idx_test,:][idx_genre_test,:]

        Y_mask_train = Y_mask[idx_train,:][idx_genre_train,:]
        Y_mask_test = Y_mask[idx_test,:][idx_genre_test,:]

        X_train_openmic = X_train_openmic[genre_train_openmic!="other"]
        Y_train_openmic = Y_train_openmic[genre_train_openmic!="other"]
        Y_mask_train = Y_mask_train[genre_train_openmic!="other"]

        X_test_openmic = X_test_openmic[genre_test_openmic!="other"]
        Y_test_openmic = Y_test_openmic[genre_test_openmic!="other"]
        Y_mask_test = Y_mask_test[genre_test_openmic!="other"]

        genre_train_openmic = genre_train_openmic[genre_train_openmic!="other"]
        genre_test_openmic = genre_test_openmic[genre_test_openmic!="other"]

        return (X_train_openmic, Y_train_openmic), (X_test_openmic, Y_test_openmic), (Y_mask_train, Y_mask_test), \
                (genre_train_openmic, genre_test_openmic)
    

    def instrument_classfication(self, train_set, test_set, irmas_feature, openmic_feature):
        if train_set == test_set and (self.debias_method == '' or self.debias_method == '-k'):
            globals()['models_' + train_set] = dict()
        elif train_set == test_set and 'lda' in self.debias_method:
            globals()['LDAcoef_' + train_set] = dict()

        (X_train_irmas, Y_train_irmas), (X_test_irmas, Y_test_irmas), (genre_train_irmas, genre_test_irmas) = irmas_feature
        (X_train_openmic, Y_train_openmic), (X_test_openmic, Y_test_openmic), (Y_mask_train, Y_mask_test), \
                (genre_train_openmic, genre_test_openmic) = openmic_feature
        
        print('Train on {}, test on {}'.format(train_set, test_set))

        for instrument in tqdm(self.instrument_map):
            
            ###### OpenMIC
            # Map the instrument name to its column number
            inst_num = self.openmic_map[instrument]
            
            # First, sub-sample the data: we need to select down to the data for which we have annotations
            # This is what the mask arrays are for
            train_inst = Y_mask_train[:, inst_num]
            test_inst = Y_mask_test[:, inst_num]
            
            # Here, we're using the Y_mask_train array to slice out only the training examples
            # for which we have annotations for the given class
            X_train_inst_openmic = X_train_openmic[train_inst]
            genre_train_inst_openmic = genre_train_openmic[train_inst]
            
            # Again, we slice the labels to the annotated examples
            # We thresold the label likelihoods at 0.5 to get binary labels
            Y_train_inst_openmic = Y_train_openmic[train_inst, inst_num] >= 0.5
            Y_train_noninst_openmic = Y_train_openmic[train_inst, inst_num] < 0.5
            
            # Repeat the above slicing and dicing but for the test set
            X_test_inst_openmic = X_test_openmic[test_inst]
            Y_test_inst_openmic = Y_test_openmic[test_inst, inst_num] >= 0.5

            ###### IRMAS
            # get the training and testing labels for each instrument
            X_train_inst_irmas = X_train_irmas
            genre_train_inst_irmas = genre_train_irmas
            X_test_inst_irmas = X_test_irmas

            Y_train_inst_irmas = Y_train_irmas==instrument
            Y_train_noninst_irmas = Y_train_irmas!=instrument
            Y_test_inst_irmas = Y_test_irmas==instrument

            ###### classification ######
            X_train_inst_openmic_true = X_train_inst_openmic[Y_train_inst_openmic]
            X_train_inst_openmic_false = X_train_inst_openmic[Y_train_noninst_openmic]

            X_train_inst_irmas_true = X_train_inst_irmas[Y_train_inst_irmas]
            X_train_inst_irmas_false = X_train_inst_irmas[Y_train_noninst_irmas]

            genre_train_inst_openmic_true = genre_train_inst_openmic[Y_train_inst_openmic]
            genre_train_inst_openmic_false = genre_train_inst_openmic[Y_train_noninst_openmic]

            genre_train_inst_irmas_true = genre_train_inst_irmas[Y_train_inst_irmas]
            genre_train_inst_irmas_false = genre_train_inst_irmas[Y_train_noninst_irmas]

            dim_inst = min(X_train_inst_openmic_true.shape[0], X_train_inst_irmas_true.shape[0])
            dim_noninst = min(X_train_inst_openmic_false.shape[0], X_train_inst_irmas_false.shape[0])

            X_train_inst_openmic_true, genre_train_inst_openmic_true = \
                self.resample_data(X_train_inst_openmic_true, genre_train_inst_openmic_true, dim_inst)
            X_train_inst_irmas_true, genre_train_inst_irmas_true = \
                self.resample_data(X_train_inst_irmas_true, genre_train_inst_irmas_true, dim_inst)

            X_train_inst_openmic_false, genre_train_inst_openmic_false = \
                self.resample_data(X_train_inst_openmic_false, genre_train_inst_openmic_false, dim_noninst)
            X_train_inst_irmas_false, genre_train_inst_irmas_false = \
                self.resample_data(X_train_inst_irmas_false, genre_train_inst_irmas_false, dim_noninst)
            
            X_train_inst_irmas = np.vstack((X_train_inst_irmas_true, X_train_inst_irmas_false))
            Y_train_inst_irmas = np.array([[True] * len(X_train_inst_irmas_true) + [False] * len(X_train_inst_irmas_false)]).reshape(-1,)

            X_train_inst_openmic = np.vstack((X_train_inst_openmic_true, X_train_inst_openmic_false))
            Y_train_inst_openmic = np.array([[True] * len(X_train_inst_openmic_true) + [False] * len(X_train_inst_openmic_false)]).reshape(-1,)

            if 'k' in self.debias_method:

                X_all = np.vstack((X_train_inst_irmas, X_train_inst_openmic))
                # kernelize embedding with fastfood
                Sampler = Fastfood(n_components=4*X_all.shape[1], random_state=0,
                                                sigma=np.median(pairwise_distances(X_all, metric='l2')))
                X_all = Sampler.fit_transform(X_all)
                X_train_inst_irmas = X_all[:len(X_train_inst_irmas)]
                X_train_inst_openmic = X_all[len(X_train_inst_irmas):]
                X_train_inst_irmas_true = X_train_inst_irmas[:len(X_train_inst_irmas_true)]
                X_train_inst_openmic_true = X_train_inst_openmic[:len(X_train_inst_openmic_true)]

                X_test_inst_irmas = Sampler.transform(X_test_inst_irmas)
                X_test_inst_openmic = Sampler.transform(X_test_inst_openmic)

            if train_set == 'irmas' and test_set == 'irmas':
                X_train_clf, Y_train_clf = X_train_inst_irmas, Y_train_inst_irmas
                X_test_clf, Y_test_clf = X_test_inst_irmas, Y_test_inst_irmas
            elif train_set == 'irmas' and test_set == 'openmic':
                X_train_clf, Y_train_clf = X_train_inst_irmas, Y_train_inst_irmas
                X_test_clf, Y_test_clf = X_test_inst_openmic, Y_test_inst_openmic
            elif train_set == 'openmic' and test_set == 'openmic':
                X_train_clf, Y_train_clf = X_train_inst_openmic, Y_train_inst_openmic
                X_test_clf, Y_test_clf = X_test_inst_openmic, Y_test_inst_openmic
            else:
                X_train_clf, Y_train_clf = X_train_inst_openmic, Y_train_inst_openmic
                X_test_clf, Y_test_clf = X_test_inst_irmas, Y_test_inst_irmas

            ############### LDA ###############
            # project the separation direction of the instrument class
            X_train_conca = np.vstack((X_train_inst_irmas_true, X_train_inst_openmic_true))
            genre_train_conca = np.hstack((genre_train_inst_irmas_true, genre_train_inst_openmic_true))
            Y_A = np.zeros(len(X_train_inst_irmas_true))
            Y_B = np.ones(len(X_train_inst_openmic_true))
            Y_conca = np.hstack((Y_A, Y_B))

            if self.debias_method == '-lda':
                LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                LDA.fit(X_train_conca, Y_conca)

                v = LDA.coef_.copy()
                v /= np.sqrt(np.sum(v**2))
                A = np.outer(v, v)

                X_train_clf = X_train_clf.copy().dot(np.eye(len(A)) - A)
                X_test_clf = X_test_clf.copy().dot(np.eye(len(A)) - A)

                globals()['LDAcoef_' + train_set][instrument] = LDA.coef_.copy()  

            elif self.debias_method == '-mlda':
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
