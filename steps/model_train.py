from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Scaling, regression, and categorical encoding are present in xgboost

def get_model():
    params = {
        "gamma": range(0, 2),
        "max_depth": range(0, 5),
        "lambda": range(0, 5)
    }
    model = GridSearchCV(XGBRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5)
    return model


def get_best_param(model, x_train, y_train):
    model.fit(x_train, y_train)
    best_param = model.best_params_
    return best_param


def get_best_estimator(best_param):
    model = XGBRegressor(**best_param)
    return model
    

   
