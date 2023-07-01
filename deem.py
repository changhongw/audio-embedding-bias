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

from sklearn.preprocessing import StandardScaler
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.metrics import pairwise_distances

import warnings
warnings.filterwarnings('ignore')

class deem():
<<<<<<< Updated upstream
    """DEEM: DEbiasing pre-trained EMbeddings"""
=======
    """DEEM: DEbiasing pre-trained audio EMbeddings"""
>>>>>>> Stashed changes
    
    def __init__(self, embedding, debias_method, feature_dir, instrument_map, genre_map, param_grid, class_align=None):
        self.embedding = embedding
        self.debias_method = debias_method
        self.feature_dir = feature_dir
        self.base_dir = os.path.split(os.path.abspath(feature_dir))[0]
        self.instrument_map = instrument_map 
        self.genre_map = genre_map
        self.param_grid = param_grid
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
        """match the elements of two lists and return true when there is an intersection"""
        ele_A = set(map(str.lower, A))
        ele_B = set(map(str.lower, B))
        return len(ele_A.intersection(ele_B)) > 0


    def load_irmas(self):
        """load data of IRMAS dataset"""

        print('Load IRMAS data:')
        embeddings = h5py.File(self.feature_dir, "r")

        feature = np.array(embeddings["irmas"][self.embedding]["features"])  # (13410, )
        keys_ori = np.array(embeddings["irmas"][self.embedding]["keys"])
<<<<<<< Updated upstream
        # some machine needs the following line of code; please comment out if not the case for you
        keys_ori = np.array([str(k, 'utf-8') for k in keys_ori])  
=======
        # # some machine may need the following line of code; please comment out if not the case for you
        # keys_ori = np.array([str(k, 'utf-8') for k in keys_ori])  
>>>>>>> Stashed changes
        key_clip = np.unique(keys_ori)  # (6705, )

        feature_clip = []
        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys_ori[:]==key,:],axis=0))
            
        feature_clip = np.array(feature_clip)
        genre_clip = [item[item.rindex("[")+1:item.rindex("]")] for item in key_clip]
        genre_clip = np.array(genre_clip)

        key_train = set(pd.read_csv(os.path.join(self.base_dir, "data/irmas/irmas_train.csv"), header=None, squeeze=True))
        key_test = set(pd.read_csv(os.path.join(self.base_dir, "data/irmas/irmas_test.csv"), header=None, squeeze=True))

        # these loops go through all sample keys, and save their row numbers to either idx_train or idx_test
        idx_train, idx_test = [], []
        for k in range(len(key_clip)):
            if str(key_clip[k]) in key_train:
                idx_train.append(k)
            elif str(key_clip[k]) in key_test:
                idx_test.append(k)
            else:
                raise RuntimeError("Unknown sample key={}! Abort!".format(str(key_clip[k])))
                
        # cast the idx_* arrays to numpy structures
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        instruments = [key[key.index("[")+1:key.index("]")] for key in key_clip]

        for inst in self.class_align:   # update the label of one instrument at a time
            instruments = [inst if x in self.class_align[inst] else x for x in instruments]
        instruments = np.array(instruments)

        # use the split indices to partition the features, labels, and masks
        X_train_irmas = feature_clip[idx_train,:]
        X_test_irmas = feature_clip[idx_test]

        Y_train_irmas = instruments[idx_train]
        Y_test_irmas = instruments[idx_test]

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

        genre_train_irmas = np.array(genre_train_irmas)[genre_train_idx]   # 4997
        genre_test_irmas = np.array(genre_test_irmas)[genre_test_idx]      # 1666

        print("# Train: {},  # Test: {}".format(len(genre_train_irmas), len(genre_test_irmas)))

        return (X_train_irmas[genre_train_idx], Y_train_irmas[genre_train_idx]), \
            (X_test_irmas[genre_test_idx], Y_test_irmas[genre_test_idx]), (genre_train_irmas, genre_test_irmas)
    

    def load_openmic(self):
        """load data of OpenMIC dataset"""

        print('Load OpenMIC data:')
        with open(os.path.join(self.base_dir, "data/openmic-2018/openmic_classmap_10.json"), "r") as f: # only consider 10 classes of Openmic dataset
            self.openmic_class_map = json.load(f)

        embeddings = h5py.File(self.feature_dir, "r")

        feature = np.array(embeddings["openmic"][self.embedding]["features"])
        keys = np.array(embeddings["openmic"][self.embedding]["keys"])
<<<<<<< Updated upstream
        # some machine needs the following line of code; please comment out if not the case for you
        keys = np.array([str(k, 'utf-8') for k in keys])  
=======
        # # some machine may need the following line of code; please comment out if not the case for you
        # keys = np.array([str(k, 'utf-8') for k in keys])  
>>>>>>> Stashed changes
        key_clip = np.unique(keys)

        feature_clip = []
        for key in tqdm(key_clip):
            feature_clip.append(np.mean(feature[keys[:]==key,:],axis=0))
        feature_clip = np.array(feature_clip)   # (20000, )

        # key-label map using the information from the dataset source
        np_load_old = np.load   # save np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)   # modify the default parameters of np.load

        Ytrue = np.load(os.path.join(self.base_dir, "data/openmic-2018/openmic-2018.npz"))["Y_true"]
        Ymask = np.load(os.path.join(self.base_dir, "data/openmic-2018/openmic-2018.npz"))["Y_mask"]
        sample_key = np.load(os.path.join(self.base_dir, "data/openmic-2018/openmic-2018.npz"))["sample_key"]

        np.load = np_load_old   # restore np.load for future normal usage
        del(np_load_old)

        Y_true = []
        Y_mask = []

        for key in tqdm(key_clip):
            Y_true.append(Ytrue[sample_key==key])
            Y_mask.append(Ymask[sample_key==key])
            
        Y_true = np.squeeze(np.array(Y_true))
        Y_mask = np.squeeze(np.array(Y_mask))

        # train-test split
        train_set = set(pd.read_csv(os.path.join(self.base_dir, "data/openmic-2018/openmic2018_train.csv"), header=None, squeeze=True))
        test_set = set(pd.read_csv(os.path.join(self.base_dir, "data/openmic-2018/openmic2018_test.csv"), header=None, squeeze=True))

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

        meta = pd.read_csv(os.path.join(self.base_dir, "data/openmic-2018/openmic-2018-metadata.csv"))
        train_genre_meta = list(meta["track_genres"][idx_train])  # full genre meta: ID, title, url
        test_genre_meta = list(meta["track_genres"][idx_test])

        idx_genre_train = []
        genre_train_openmic = []
        for k in tqdm(range(len(train_genre_meta))):
            if isinstance(train_genre_meta[k], str):  # this track has genre information
                idx_genre_train.append(k)
                genre_excerpt = literal_eval(train_genre_meta[k])  # split multiple genre labels
                genre_train_openmic.append([item['genre_title']for item in genre_excerpt]) # append all genre labels
        idx_genre_train = np.array(idx_genre_train)  # all indices with genre annotation: 14581

        idx_genre_test = []
        genre_test_openmic = []
        for k in tqdm(range(len(test_genre_meta))):
            if isinstance(test_genre_meta[k], str):
                idx_genre_test.append(k)
                genre_excerpt = literal_eval(test_genre_meta[k])
                genre_test_openmic.append([item['genre_title']for item in genre_excerpt])
        idx_genre_test = np.array(idx_genre_test)   # 4911

        # transfer multiple genre labels into single genre label for each audio except
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

        genre_train_openmic, genre_test_openmic = genre_train_final, genre_test_final   # 14581, 4911

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

        genre_train_openmic = genre_train_openmic[genre_train_openmic!="other"]   # (6838, )
        genre_test_openmic = genre_test_openmic[genre_test_openmic!="other"]   # (2491, )

        print("# Train: {},  # Test: {}".format(len(genre_train_openmic), len(genre_test_openmic)))

        return (X_train_openmic, Y_train_openmic), (X_test_openmic, Y_test_openmic), (Y_mask_train, Y_mask_test), \
                (genre_train_openmic, genre_test_openmic)
    

    def instrument_classfication(self, train_set, test_set, irmas_feature, openmic_feature):
        """ 
        Binary classification of instruments
        
        Args
        ------
        train_set: string
            training set name, i.e. "irmas" or "openmic"
        test_set: string
            test set name, i.e. "irmas" or "openmic"
        irmas_feature: tensor
            pre-trained embedding feature on IRMAS dataset
        openmic_feature: tensor
            pre-trained embedding feature on openmic dataset
        """

        if train_set == test_set and (self.debias_method == '' or self.debias_method == '-k'):
            globals()['models_' + train_set] = dict()
        elif 'lda' in self.debias_method:
            globals()['LDAcoef_' + train_set] = dict()

        (X_train_irmas, Y_train_irmas), (X_test_irmas, Y_test_irmas), (genre_train_irmas, genre_test_irmas) = irmas_feature
        (X_train_openmic, Y_train_openmic), (X_test_openmic, Y_test_openmic), (Y_mask_train, Y_mask_test), \
                (genre_train_openmic, genre_test_openmic) = openmic_feature
        
        print('Train on {}, test on {}'.format(train_set, test_set))

        for instrument in tqdm(self.instrument_map):
            
            ###### OpenMIC data ###### (numbers are based on "cello" example)
            # Map the instrument name to its column number
            inst_num = self.openmic_class_map[instrument]
            
            # First, sub-sample the data: select down to the data for which we have annotations
            # This is what the mask arrays are for
            train_inst = Y_mask_train[:, inst_num]   # (6838, )
            test_inst = Y_mask_test[:, inst_num]     # (2491, )
            
            # Here, we're using the Y_mask_train array to slice out only the training examples
            # for which we have annotations for the given class
            X_train_inst_openmic = X_train_openmic[train_inst]   # (628, )
            genre_train_inst_openmic = genre_train_openmic[train_inst]
            
            # Again, we slice the labels to the annotated examples
            # We thresold the label likelihoods at 0.5 to get binary labels
            Y_train_inst_openmic = Y_train_openmic[train_inst, inst_num] >= 0.5   # (628, )
            Y_train_noninst_openmic = Y_train_openmic[train_inst, inst_num] < 0.5
            
            # Repeat the above slicing and dicing for the test set
            X_test_inst_openmic = X_test_openmic[test_inst]    # (217, )
            Y_test_inst_openmic = Y_test_openmic[test_inst, inst_num] >= 0.5

            ###### IRMAS data ###### (numbers are based on "cello" example)
            # get the training and testing data for each instrument
            X_train_inst_irmas = X_train_irmas     # (4997, )
            genre_train_inst_irmas = genre_train_irmas
            X_test_inst_irmas = X_test_irmas       # (1666, )

            Y_train_inst_irmas = Y_train_irmas==instrument    # (4997, )
            Y_train_noninst_irmas = Y_train_irmas!=instrument
            Y_test_inst_irmas = Y_test_irmas==instrument      # (1666, )
 
            ###### classification ######
            X_train_inst_openmic_true = X_train_inst_openmic[Y_train_inst_openmic]      # (290, )
            X_train_inst_openmic_false = X_train_inst_openmic[Y_train_noninst_openmic]  # (338, )

            X_train_inst_irmas_true = X_train_inst_irmas[Y_train_inst_irmas]            # (294, )
            X_train_inst_irmas_false = X_train_inst_irmas[Y_train_noninst_irmas]        # (4703, )

            genre_train_inst_openmic_true = genre_train_inst_openmic[Y_train_inst_openmic]
            genre_train_inst_openmic_false = genre_train_inst_openmic[Y_train_noninst_openmic]

            genre_train_inst_irmas_true = genre_train_inst_irmas[Y_train_inst_irmas]
            genre_train_inst_irmas_false = genre_train_inst_irmas[Y_train_noninst_irmas]

            dim_inst = min(X_train_inst_openmic_true.shape[0], X_train_inst_irmas_true.shape[0])  # 290
            dim_noninst = min(X_train_inst_openmic_false.shape[0], X_train_inst_irmas_false.shape[0])  # 338

            X_train_inst_openmic_true, genre_train_inst_openmic_true = \
                self.resample_data(X_train_inst_openmic_true, genre_train_inst_openmic_true, dim_inst)  # 290
            X_train_inst_irmas_true, genre_train_inst_irmas_true = \
                self.resample_data(X_train_inst_irmas_true, genre_train_inst_irmas_true, dim_inst)  # 288

            X_train_inst_openmic_false, genre_train_inst_openmic_false = \
                self.resample_data(X_train_inst_openmic_false, genre_train_inst_openmic_false, dim_noninst) # 338
            X_train_inst_irmas_false, genre_train_inst_irmas_false = \
                self.resample_data(X_train_inst_irmas_false, genre_train_inst_irmas_false, dim_noninst)  # 336
            
            X_train_inst_irmas = np.vstack((X_train_inst_irmas_true, X_train_inst_irmas_false))  # 624
            Y_train_inst_irmas = np.array([[True] * len(X_train_inst_irmas_true) + [False] * len(X_train_inst_irmas_false)]).reshape(-1,)
            genre_train_inst_irmas = np.hstack((genre_train_inst_irmas_true, genre_train_inst_irmas_false))

            X_train_inst_openmic = np.vstack((X_train_inst_openmic_true, X_train_inst_openmic_false))  # 628
            Y_train_inst_openmic = np.array([[True] * len(X_train_inst_openmic_true) + [False] * len(X_train_inst_openmic_false)]).reshape(-1,)
            genre_train_inst_openmic = np.hstack((genre_train_inst_openmic_true, genre_train_inst_openmic_false))

            if train_set == 'irmas' and test_set == 'irmas':
                X_train_clf, Y_train_clf = X_train_inst_irmas, Y_train_inst_irmas  # (624, )
                X_test_clf, Y_test_clf = X_test_inst_irmas, Y_test_inst_irmas      # (1666, )
            elif train_set == 'irmas' and test_set == 'openmic':
                X_train_clf, Y_train_clf = X_train_inst_irmas, Y_train_inst_irmas
                X_test_clf, Y_test_clf = X_test_inst_openmic, Y_test_inst_openmic  # (217, )
            elif train_set == 'openmic' and test_set == 'openmic':
                X_train_clf, Y_train_clf = X_train_inst_openmic, Y_train_inst_openmic  # (628, )
                X_test_clf, Y_test_clf = X_test_inst_openmic, Y_test_inst_openmic
            else: # train_set == 'openmic' and test_set == 'irmas':
                X_train_clf, Y_train_clf = X_train_inst_openmic, Y_train_inst_openmic
                X_test_clf, Y_test_clf = X_test_inst_irmas, Y_test_inst_irmas

            if 'k' in self.debias_method:
            
                scaler = StandardScaler()
                X_train_inst_irmas = scaler.fit_transform(X_train_inst_irmas)
                X_train_inst_openmic = scaler.fit_transform(X_train_inst_openmic)
                X_train_all = np.vstack((X_train_inst_irmas, X_train_inst_openmic))

                # kernelize embedding with fastfood
                Sampler = Fastfood(n_components=4*X_train_all.shape[1], random_state=self.param_grid['random_state'],
                                                sigma=np.median(pairwise_distances(X_train_all, metric='l2')))
                X_train_all = Sampler.fit_transform(X_train_all)   # (1252, 1024): 128 -> 1024
                X_train_inst_irmas = X_train_all[:len(X_train_inst_irmas)]
                X_train_inst_openmic  = X_train_all[len(X_train_inst_irmas):]

                scaler = StandardScaler()
                X_train_clf = scaler.fit_transform(X_train_clf)  # (628, 128)
                X_test_clf = scaler.transform(X_test_clf)        # (217, 128)
                X_test_clf = np.nan_to_num(X_test_clf, np.nan)  

                X_train_clf = Sampler.transform(X_train_clf)
                X_test_clf  = Sampler.transform(X_test_clf)

            ############### LDA ###############
            # For global bias correction
            X_train_conca = np.vstack((X_train_inst_irmas, X_train_inst_openmic))  # (1252, )
            genre_train_conca = np.hstack((genre_train_inst_irmas, genre_train_inst_openmic))   
            # # For class-wise bias correction
            # X_train_conca = np.vstack((X_train_inst_irmas_true, X_train_inst_openmic_true))  
            # genre_train_conca = np.hstack((genre_train_inst_irmas_true, genre_train_inst_openmic_true))

            Y_A = np.zeros(len(X_train_inst_irmas))
            Y_B = np.ones(len(X_train_inst_openmic))
            Y_conca = np.hstack((Y_A, Y_B))

            if 'lda' in self.debias_method and '-m' in self.debias_method:
                genre_LDAcoef = []
                for genre in self.genre_map:
                    
                    X_train_conca_sub = X_train_conca[genre_train_conca == genre] # 494
                    Y_conca_sub = Y_conca[genre_train_conca == genre]  # irmas 209, openmic 285

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
            
            model_auc = roc_auc_score(Y_test_clf, Y_pred_scores)
            model_ap = average_precision_score(Y_test_clf, Y_pred_scores)
            
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
            with open(os.path.join(self.base_dir, 'models/models_' + train_set + '_' + self.embedding + self.debias_method + '.pickle'), 'wb') as fdesc:
                pickle.dump(globals()['models_'+train_set], fdesc)
        elif train_set == test_set and 'lda' in self.debias_method:
            with open(os.path.join(self.base_dir, 'models/LDAcoef_' + train_set + '_' + self.embedding + self.debias_method + '.pickle'), 'wb') as fdesc:
                pickle.dump(globals()['LDAcoef_'+train_set], fdesc) 


    def resample_data(self, feature, genre, num):
        """
        Select "num" number of samples from original feature with the same genre distribution

        Args
        ------
        feature: tensor
            original pre-trained embedding feature
        genre: list
            original genre name
        num: int 
            target number of samples
        """
        random.seed(self.param_grid['random_state'])

        feature_all = feature[0,:]   # (294, )
        genre_all = genre[0]         # (294, ), num = 290
        ratio =  num / len(feature)  # ratio between target and original => using this ratio to get sample from each genre, 0.986
        
        for genre_item in self.genre_map:
            genre_target_bool = genre == genre_item  

            len_target = genre_target_bool.sum()  # number of samples of this specific genre, 84
            idx_shuffle = random.sample(range(len_target), len_target)  # shuffle the idx of target genre to select randomly, 84 
            idx_keep = idx_shuffle[:int(len_target * ratio)]  # select "total * ratio" samples from this genre, 82

            feature_new = feature[genre_target_bool][idx_keep]  # (82, )
            genre_new = genre[genre_target_bool][idx_keep]      # (82, ), same genre

            feature_all = np.vstack((feature_all, feature_new))  # (83, )
            genre_all = np.hstack((genre_all, genre_new))

        return feature_all[1:,], genre_all[1:]    # (288, )
