from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search_cv(model, params, cv):

    return GridSearchCV(model, params, cv=cv)

def randomized_search(model, params, num_iter):

    return RandomizedSearchCV(model, params, n_iter= num_iter)