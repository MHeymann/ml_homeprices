"""
The main method to run the machine learning task
"""
# System imports
import os
import sys

# Third party imports
import numpy as np
# import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer as Imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Local imports
from home_prices.common.utils import fetch_tgz_data, load_csv_data, \
    split_train_test_stratified
from home_prices.common.utils import CategoricalEncoder, DataFrameSelector

ROOMS_IX, BEDROOMS_IX, POPULATION_IX, HOUSEHOLD_IX = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom tranformer for adding useful combined attributes to
    our data.
    """
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):  # pylint: disable=invalid-name,unused-argument
        """
        Do nothing
        """
        return self

    def transform(self, X):  # pylint: disable=invalid-name
        """
        Add the required new combined fields
        """
        rooms_per_household = X[:, ROOMS_IX] / X[:, HOUSEHOLD_IX]
        population_per_household = X[:, POPULATION_IX] / X[:, HOUSEHOLD_IX]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, BEDROOMS_IX] / X[:, ROOMS_IX]
            ret_value = np.c_[X, rooms_per_household,
                              population_per_household, bedrooms_per_room]
        else:
            ret_value = np.c_[X, rooms_per_household,
                              population_per_household]
        return ret_value

def fetch_and_load_data():
    if not os.path.isfile("./datasets/housing/housing.csv"):
        fetch_tgz_data()

    # Load the csv file into a dataframe
    data = load_csv_data()
    return data

def setup_pipeline(num_attributes, cat_attributes):
    # Setup numerical and categorical pipelines
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('cat_encoder', CategoricalEncoder()),
        ])

    # Bring together with FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ])
    return full_pipeline

def display_scores(scores):
    """
    A method to display cross validation scores
    :param scores:  the cross validation scores of some model
    """
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


def display_model_performance(regressor, data_prepared, data_labels, 
                            model_name):
    predictions = regressor.predict(data_prepared)
    model_mse = mean_squared_error(data_labels, predictions)
    model_rmse = np.sqrt(model_mse)

    print(model_name + ":")
    print("Standard Error:", model_rmse)
    print()

    model_scores = cross_val_score(regressor,
                                  data_prepared,
                                  data_labels,
                                  scoring="neg_mean_squared_error",
                                  cv=10)
    model_rmse_scores = np.sqrt(-model_scores)

    print(model_name, "Scores:")
    display_scores(model_rmse_scores)


def fine_tune_model(reg_model, param_grid, housing_prepared, housing_labels):
    # Fine-tune Random Forest Regressor
    grid_search = GridSearchCV(reg_model, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)

    # Show the best parameters
    print()
    print("Random Forest Regressor best parameters:")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"],
            cvres["params"]):
        print(np.sqrt(-mean_score), params)
    final_model = grid_search.best_estimator_
    return final_model

def print_attribute_importances(attribute_weights, num_attributes, full_pipeline):
    extra_attributes = ["rooms_per_hhold", "pop_per_hhold",
                        "bedrooms_per_room"]
    cat_pipeline = dict(full_pipeline.transformer_list)["cat_pipeline"]
    cat_encoder = cat_pipeline.named_steps["cat_encoder"]
    cat_one_hot_attributes = list(cat_encoder.categories_[0])
    attributes = num_attributes + extra_attributes + cat_one_hot_attributes
    named_attribute_weights = sorted(zip(attribute_weights, attributes), reverse=True)
    for weight in named_attribute_weights:
        print(weight)
    
def main(regressor="random_forest"):
    """
    The main method
    """
    # Fetch data from internet
    data = fetch_and_load_data()

    # Process median_income into categories
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # Split data into training and testing sets
    train_data, test_data = split_train_test_stratified(data, "income_cat")

    # Extract labels and housing data
    housing_labels = train_data["median_house_value"].copy()
    housing = train_data.drop("median_house_value", axis=1)

    # split housing into categorical and numerical data
    # cat_attributes = ["ocean_proximity", "income_cat"]
    cat_attributes = ["ocean_proximity"]
    num_attributes = ['longitude', 'latitude', 'housing_median_age',
                      'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income']
 
    # Set up pipeline to prepare data with.
    full_pipeline = setup_pipeline(num_attributes, cat_attributes)

    # Prepare the data
    housing_prepared = full_pipeline.fit_transform(housing)

    print()

    # Select the appropriate regressor
    if regressor == "linear":
        reg_model = LinearRegression()
        reg_name = "Linear Regressor"
    elif regressor == "random_forest":
        reg_model = RandomForestRegressor()
        reg_name = "Random Forest Regressor"
    elif regressor == "decision_tree":
        reg_model = DecisionTreeRegressor()
        reg_name = "Decision Tree Regressor"
    elif regressor == "svr":
        reg_model = SVR(kernel="linear", gamma='auto')
        reg_name = "Support Vector Machine"
    else:
        error_mes = "Regressor '{regressor}' not recognised."
        raise ValueError(error_mes.format(regressor=regressor))

    # Train regression model
    reg_model.fit(housing_prepared, housing_labels)
    display_model_performance(reg_model,
                              housing_prepared,
                              housing_labels,
                              reg_name)

    if regressor == "random_forest":
        # Fine tune Random Forest
        param_grid = [
                {'n_estimators': [50, 100, 1000], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [50, 100, 1000], 'max_features': [2, 4, 6]}]
        final_model = fine_tune_model(RandomForestRegressor(),
                                      param_grid,
                                      housing_prepared,
                                      housing_labels)
        # Get the best model weights
        print()
        print("Attribute weights:")
        feature_importances = final_model.feature_importances_
        print_attribute_importances(feature_importances, num_attributes, full_pipeline)
    elif regressor == "linear":
        final_model = reg_model
        print("Coefficients used by linear model:")
        coeffs = final_model.coef_
        print_attribute_importances(coeffs, num_attributes, full_pipeline)
    elif regressor == "decision_tree":
        # Fine tune Decision Tree
        param_grid = [{'criterion': ["mse", "friedman_mse", "mae"]}]
        final_model = fine_tune_model(DecisionTreeRegressor(),
                                      param_grid,
                                      housing_prepared,
                                      housing_labels)
    elif regressor == "svr":
        param_grid = [
                {'kernel': ["linear"], "C": [10000, 100000]},
                {'kernel': ["rbf"], "C": [10000, 100000],
                    "gamma": [0.045, 0.05, 0.055]}]
        final_model = fine_tune_model(SVR(),
                                      param_grid,
                                      housing_prepared,
                                      housing_labels)
    else:
        final_model = reg_model

    print()

    # Evaluate on test set
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final Standard Error:", final_rmse)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Running with Random Forest Regressor Model, the default.")
        main("random_forest")
    elif sys.argv[1] == "linear":
        print("Running with Linear Regressor Model.")
        main("linear")
    elif sys.argv[1] == "random_forest":
        print("Running with Random Forest Regressor Model.")
        main("random_forest")
    elif sys.argv[1] == "decision_tree":
        print("Running with Decision Tree Regressor Model.")
        main("decision_tree")
    elif sys.argv[1] == "svr":
        print("Running with Support Vector Machine Regressor Model.")
        main("svr")
    else:
        print("Model " + sys.argv[1] + " not recognised.")
