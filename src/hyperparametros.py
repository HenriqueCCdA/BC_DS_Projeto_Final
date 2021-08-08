from scipy.stats import uniform
import numpy as np

parameters_dummy = {'strategy' : ['stratified', 'most_frequent', 'prior', 'uniform']}

parameters_logistic_regression = {'C': uniform(loc=0, scale=4)}

parameters_decision_tree = {'max_depth'       : np.arange(1, 21),
                            'criterion'       : ['gini', 'entropy'],
                            'min_samples_leaf': np.arange(1, 6),
                            'max_leaf_nodes'  : np.arange(2, 6)
                            }

parameters_random_forest = {'n_estimators'    : [100, 150, 200, 250, 300, 350, 400],
              'max_depth'       : np.arange(1, 21),
              'criterion'       : ['gini', 'entropy'],
              'min_samples_leaf': np.arange(1, 6),
              'max_leaf_nodes'  : np.arange(2, 11)
             }

parameters_svc = {'kernel'          : ['linear', 'poly', 'rbf', 'sigmoid'],
                  'C'               : uniform(loc=0, scale=2),
                  'gamma'           : ['scale', 'auto'],
                  'shrinking'       : [True, False]
                  }

parameters_KNeighbors = {'n_neighbors'     : np.arange(1, 11),
                         'p'               : [1, 2],
                         'weights'         : ['uniform', 'distance'],
                         'algorithm'       : ['auto', 'ball_tree', 'kd_tree', 'brute']
             }