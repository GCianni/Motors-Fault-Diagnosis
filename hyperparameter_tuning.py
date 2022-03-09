import warnings
import optunity
import optunity.metrics

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from evolutionary_search import EvolutionaryAlgorithmSearchCV


warnings.filterwarnings("ignore")


def RandomSearch(X, y, X_test, y_test, pds, estimator, search_space_dict, estimator_type):
    clf = RandomizedSearchCV(estimator=estimator, cv=pds, param_distributions=search_space_dict, n_jobs=-1)
    clf.fit(X, y)
    print(f'\n{estimator_type}')
    print(f'Random Search Best param: {clf.best_params_}')
    print(f'Random Search Best Accuracy: {str(clf.best_score_)}')

    y_pred = clf.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f' Random Search Test Accuracy: {test_acc}')

    return [clf.best_params_, clf.best_score_, test_acc, y_pred]


def GeneticSearch(X, y, X_test, y_test, pds, estimator, search_space_dict, estimator_type):
    clf = EvolutionaryAlgorithmSearchCV(estimator=estimator, params=search_space_dict, scoring="accuracy", cv=pds,
                                        verbose=0,
                                        population_size=50,
                                        gene_mutation_prob=0.05,
                                        gene_crossover_prob=0.6,
                                        tournament_size=5,
                                        generations_number=10,
                                        n_jobs=1)
    clf.fit(X, y)

    print(f'\n{estimator_type}')
    print(f'Genetic Search Best param: {clf.best_params_}')
    print(f'Genetic Search Best Accuracy: {str(clf.best_score_)}')

    y_pred = clf.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Genetic Search  Test Accuracy: {test_acc}')

    return [clf.best_params_, clf.best_score_, test_acc, y_pred]
