import hyperparameter_tuning as hyp_opt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


def classifier(X, y, X_test, y_test, pds, estimator, opt):
    rf_params = {
        'n_estimators': [10, 20, 30],
        'max_depth': [15, 20, 30, 50],
        # 'min_samples_leaf': [1,2,4,8],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    ada_params = {
        'n_estimators': [10, 50, 250, 500, 750, 1000],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 10],
    }

    xgb_params = {'max_depth': [2, 3, 6, 10],
                  'learning_rate': [0.01, 0.05, 0.1, 0.2],
                  'n_estimators': [10, 50, 100, 250, 500, 1000],
                  'colsample_bytree': [0.3, 0.7, 0.9, 1]}

    ann_params = {
        'hidden_layer_sizes': [(2, 2, 2), (4, 4), (8, 8), (32, 32), (64, 64,), (128, 128,),
                               (2, 2, 2), (4, 4, 4), (8, 8, 8), (32, 32, 32), (64, 64, 64), (128, 128, 128)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'learning_rate': ['constant', 'adaptive'],
    }

    gan_params = {}

    param_dict = {
        'Random Forest': [rf_params, RandomForestClassifier(random_state=0)],
        'Adaboost': [ada_params, AdaBoostClassifier(random_state=0)],
        'XGBoost': [xgb_params, XGBClassifier(num_class=2,
                                              verbosity=0,
                                              objective='multi:softmax',
                                              use_label_encoder=False)],
        'Neural Network': [ann_params, MLPClassifier(tol=1e-3,
                                                     max_iter=100,
                                                     early_stopping=True,
                                                     n_iter_no_change=4)],
        'GAN': gan_params
    }

    search_space = param_dict[estimator][0]
    clf = param_dict[estimator][1]

    if opt == 'Random_Search':
        return hyp_opt.RandomSearch(X, y, X_test, y_test, pds, clf, search_space, estimator)
    elif opt == 'Genetic_Search':
        return hyp_opt.GeneticSearch(X, y, X_test, y_test, pds, clf, search_space, estimator)
    else:
        return 'error'
