"""
The main method to run the machine learning task
"""
# Third party imports
import numpy as np
# import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

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

def main():
    """
    The main method
    """
    # Fetch data from internet
    fetch_tgz_data()

    # Load the csv file into a dataframe
    data = load_csv_data()

    # Process median_income into categories
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # Split data into training and testing sets
    train_data, test_data = split_train_test_stratified(data, "income_cat")
    print("Test data:")
    print(test_data)

    # Extract labels and housing data
    housing_labels = train_data["median_house_value"].copy()
    housing = train_data.drop("median_house_value", axis=1)

    # split housing into categorical and numerical data
    # cat_attributes = ["ocean_proximity", "income_cat"]
    cat_attributes = ["ocean_proximity"]
    num_attributes = ['longitude', 'latitude', 'housing_median_age',
                      'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income']

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

    # Prepare the data
    housing_prepared = full_pipeline.fit_transform(housing)
    print("housing prepared shape", housing_prepared.shape)

    # Train a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing_prepared[0:5]
    some_labels = housing_labels.iloc[:5]
    print("Predictions:", lin_reg.predict(some_data))
    print("Labels:", list(some_labels))

if __name__ == '__main__':
    main()
