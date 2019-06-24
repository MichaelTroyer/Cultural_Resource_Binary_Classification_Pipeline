parameters = {
       
    'K Nearest Neighbors':                  
        {
         'algorithm'                : ['ball_tree', 'kd_tree', 'brute'],
        #  'leaf_size'                : [1],
         'metric'                   : ['manhattan', 'euclidean', 'chebyshev', 'seuclidean', 'mahalanobis'],
         'n_neighbors'              : [1, 3, 5, 7, 9],
        #  'p'                        : [2],
         'weights'                  : ['distance', 'uniform']
         },  
    
    'Random Forest':
        {
         'bootstrap'                : [True, False], 
         'criterion'                : ['entropy', 'gini'],  
         'max_depth'                : [None, 10],
         'max_features'             : [None, 10],   
        #  'max_leaf_nodes'           : [None],
        #  'min_impurity_split'       : [1e-07],
        #  'min_samples_leaf'         : [1],
        #  'min_samples_split'        : [2], 
        #  'min_weight_fraction_leaf' : [0.0],
         'n_estimators'             : [10], 
         'oob_score'                : [True, False]
         },                                  
                     
    'Gradient Boosted Decision Trees':
        {
         'criterion'                : ['friedman_mse', 'mse', 'mae'], 
         'loss'                     : ['deviance', 'exponential'],    
         'learning_rate'            : [0.1, 1, 10],
         'max_depth'                : [None, 2, 4, 8],
         'max_features'             : [None, 2, 4, 8],   
        #  'max_leaf_nodes'           : [None, 1, 10],
        #  'min_impurity_split'       : [1e-07],
        #  'min_samples_leaf'         : [1],
        #  'min_samples_split'        : [2], 
        #  'min_weight_fraction_leaf' : [0.0],
         'n_estimators'             : [10, 100],
         'presort'                  : ['auto'],
        #  'subsample'                : [1]
         },
                     
    'Logistic Regression':
        {
         'C'                        : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
         'class_weight'             : [None, 'balanced'],
         'dual'                     : [True, False],
         'fit_intercept'            : [True, False],
         'intercept_scaling'        : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
         'max_iter'                 : [100],
         'penalty'                  : ['l1', 'l2'],
         'solver'                   : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
         },
    
    'Neural Network Multi-layer Perceptron':
        {
         'activation'               : ['identity', 'relu', 'logistic', 'tanh'],
         'alpha'                    : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
         'batch_size'               : ['auto'],
         'beta_1'                   : [0.9],   # adam
         'beta_2'                   : [0.999], # adam
         'early_stopping'           : [False], 
         'epsilon'                  : [1e-08],
         'hidden_layer_sizes'       : [(100,)], 
         'learning_rate'            : ['constant', 'invscaling', 'adaptive'],  # sgd
         'learning_rate_init'       : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
         'max_iter'                 : [200], 
         'momentum'                 : [0.9],         # sgd
         'nesterovs_momentum'       : [True],        # sgd
         'power_t'                  : [0.5],         # sgd
         'shuffle'                  : [True],
         'solver'                   : ['adam', 'lbfgs', 'sgd'],
         'tol'                      : [0.0001],
         'validation_fraction'      : [0.1]          # early-stopping
         },
    
    'Linear Support Vector Classifier':
       {
        'C'                         : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'class_weight'              : [None, 'balanced'],  
        'dual'                      : [True, False],  
        'fit_intercept'             : [True, False],  
        'intercept_scaling'         : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'loss'                      : ['hinge', 'squared_hinge'],
        'max_iter'                  : [100, 200, 400],
        'penalty'                   : ['l1', 'l2'],
        },
    
    'Support Vector Machine':
        {
         'C'                        : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
         'cache_size'               : [200],
         'coef0'                    : [0.0001, 0.001, 0.01, 0.1, 1],
         'class_weight'             : ['balanced', None],
         'degree'                   : range(1, 11), 
         'gamma'                    : ['auto'],
         'kernel'                   : ['linear', 'poly', 'rbf', 'sigmoid'],
         'max_iter'                 : [400],
         'probability'              : [True, False], 
         'shrinking'                : [True, False], 
         }
    }
