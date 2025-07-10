import sys,csv
from collections import Counter
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.metrics import make_scorer
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV)
from sklearn.utils import resample
import pandas as pd
import itertools
from random import sample
import time
import numpy as np


# import dataset
comments = pd.read_csv(r"Dataset/Data_ReadyForAnalysis_WithComments&MetaInfo.csv")

#create lsit for test variables (included only one - other variables were run in parallel with different scripts at once on SURF)
test_variables = ['LIBERAL_DUMMY'] 

# process comments; write to new dataframe column
comments = comments.dropna(subset = ['commentText'])
select = comments[test_variables+['ID']+['commentText']]  
select = select.drop_duplicates('ID').reset_index(drop = True)
len(select)

def combine_configuration():
    Vectorizers = [CountVectorizer, TfidfVectorizer]
    Classifiers = [MultinomialNB(), LogisticRegression(max_iter=1000, n_jobs=-1),
                   SVC(kernel='rbf', class_weight="balanced"), SVC(kernel='linear', class_weight="balanced")
                  ]
    config = [Vectorizers, Classifiers]
    configurations = list(itertools.product(*config))
    return configurations

def machine_learning(train, test, labels):
    acc = pd.DataFrame(columns = ['Vectorizer', 'Classifier','Parameters', 'F1_score','Recall','Precision','Accuracy','Ratio_resampled'])

    train_labels = train[labels]
    train_texts = train['commentText']
    test_labels = test[labels]
    test_texts = test['commentText']

    configurations = combine_configuration()
    
    for vectorizer, classifier in configurations:
        pipeline = Pipeline(steps = [
          ("vectorizer", vectorizer()), 
          ("classifier", classifier)])

        grid = {"vectorizer__ngram_range": [(1,1), (1,2)],
                "vectorizer__max_df": [0.5, 1.0],
                "vectorizer__min_df": [0, 5],
                "classifier__C": [0.01, 1, 100]
               }
        
        try:
            search=GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=grid, scoring='f1', cv=5)
            search.fit(train_texts, train_labels)
        except:
            #print('regularization is not applicable')
            grid.pop('classifier__C')
            search=GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=grid, scoring='f1', cv=5)
            search.fit(train_texts, train_labels)
        #print(search.cv_results_['split1_test_score'])
        y_pred = search.predict(test_texts)
        acc = pd.concat([acc, pd.DataFrame({'Vectorizer': [vectorizer],
                                    'Classifier': [classifier],
                                    'Parameters': [search.best_params_],
                                    'F1_score': [metrics.f1_score(test_labels, y_pred)],
                                    'Recall': [metrics.recall_score(test_labels, y_pred)],
                                    'Precision': [metrics.precision_score(test_labels, y_pred, zero_division = 0)],
                                    'Accuracy': [metrics.accuracy_score(test_labels, y_pred)],
                                    'Ratio_resampled': [Counter(train_labels)[1] / len(train_labels)],
                                    'Manual': [test_labels],
                                    'Prediction': [y_pred],
                                    'split0_test_score': [search.cv_results_['split0_test_score'].mean()],
                                    'split1_test_score': [search.cv_results_['split1_test_score'].mean()],
                                    'split2_test_score': [search.cv_results_['split2_test_score'].mean()],
                                    'split3_test_score': [search.cv_results_['split3_test_score'].mean()],
                                    'split4_test_score': [search.cv_results_['split4_test_score'].mean()],
                                    'split0_test_score_std': [search.cv_results_['split0_test_score'].std()],
                                    'split1_test_score_std': [search.cv_results_['split1_test_score'].std()],
                                    'split2_test_score_std': [search.cv_results_['split2_test_score'].std()],
                                    'split3_test_score_std': [search.cv_results_['split3_test_score'].std()],
                                    'split4_test_score_std': [search.cv_results_['split4_test_score'].std()]})], ignore_index=True)



                         
    best_classifier = acc[acc['F1_score'] == acc['F1_score'].max()].reset_index()
    #print('algorithm with maximum F1_score:', best_classifier)
    return acc, best_classifier['Prediction'][0]


# # Random (re)sampling
def impute(data): #impute missing values
    columns = ['F1_score', 'Precision', 'Recall']
    
    for column in columns:
        mean_value = data[data[column] != 0][column].mean()
        data[column] = np.where(data[column] == 0, mean_value, data[column])

    return data
    
def random_sampling(data, size: int):
    r_sample = data.sample(n=size) #do not create random seed for random resampling

    return r_sample

def random_on_sets(data):
    accuracy = pd.DataFrame(columns = ['Variable', 'Vectorizer', 'Classifier','Parameters', 'F1_score','Recall','Precision','Accuracy','Ratio_test','Ratio_resampled','Manual','Prediction', 'Sample Size'])

    for size in range(250,3862,250):
        train_set, test_set = sklearn.model_selection.train_test_split(random_sampling(data, size=size), test_size = 0.20, random_state= None)
        train_set = train_set[['LIBERAL_DUMMY','ID','commentText']]
        test_set = test_set[['LIBERAL_DUMMY','ID','commentText']]

        #for v in test_variables:
        v = 'LIBERAL_DUMMY'
        print(v)
        acc,prediction = machine_learning(train_set, test_set, v)

        print(f'size is {size}')

        acc['Variable'] = v
        acc['Ratio_test'] = test_set[v].mean()
        acc['Ratio_prediction'] = prediction.mean()
        acc['Sample Size'] = size

        #impute zero_division cells with mean
        acc = impute(acc)

        #concat
        accuracy = pd.concat([accuracy,acc],ignore_index=True)

    return accuracy

def monte_carlo(data):
    accuracy_all = pd.DataFrame()
    for mc_round in range (1, 201, 1):
        start_time = time.time()

        print(f"Monte Carlo round {mc_round}")
        accuracy = random_on_sets(data)
        accuracy['Monte_carlo_round'] = mc_round

        accuracy_all = pd.concat([accuracy, accuracy_all])

        end_time = time.time()
        elapsed = end_time-start_time

        print(f"Time elapsed:{elapsed}")

        accuracy_all.to_csv(f'outcome.csv') #temporary data saving

    return accuracy_all

#save csvs
outcome = monte_carlo(comments)
outcome.to_csv(f'outcome.csv') #final 


