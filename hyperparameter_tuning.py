from sklearn.model_selection import PredefinedSplit, GridSearchCV
    def GridSearch (X, y, X_test, y_test, pds, estimator,search_space_dict):
        clf =  GridSearchCV(estimator=estimator, cv=pds, param_grid=search_space_dict, n_jobs = -1)
        clf.fit(X,y)
        clf.best
    # Fit with all data
    clf.fit(X, y)