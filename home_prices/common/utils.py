"""A file with general use tools"""

# System imports
import os

# Third party imports
import tarfile
from zlib import crc32
import pandas as pd
import numpy as np
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import FeatureUnion

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_tgz_data(tgz_url=HOUSING_URL, data_path=HOUSING_PATH,
                   file_name="housing.tgz"):
    """
    A function to fetch and extract data
    """
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path, file_name)
    urllib.request.urlretrieve(tgz_url, tgz_path)
    tgz_file = tarfile.open(tgz_path)
    tgz_file.extractall(path=data_path)
    tgz_file.close()


def load_csv_data(csv_dir=HOUSING_PATH):
    """
    A function that loads csv data into a pandas dataframe
    :return: A pandas dataframe
    """
    csv_path = os.path.join(csv_dir, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data=None, test_ratio=0.2):
    """
    Take a dataset and split it into an ordered pair of test and train data

    :return: A tuple (<train_data>, <test_data>)
    """
    assert data is not None, "Data may not be None"
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def _test_set_check(identifier=None, test_ratio=0.2):
    """
    Helper function to decide if test set or train set
    """
    assert identifier is not None, "Identifier may not be None"
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data=None, test_ratio=0.2, id_column=""):
    """
    Split the data based on some hash value identifier
    """
    assert data is not None
    assert id_column != ""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: _test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def split_train_test_stratified(data, \
                                strat_column, \
                                test_ratio=0.2, \
                                random_state=42):
    """
    A function to split data by stratified categories
    """
    assert data is not None, "Data may not be None"
    assert strat_column is not None, "strat_column may not be None"
    assert strat_column != "", "Please provide a strat_column value"

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio,
                                   random_state=random_state)
    for train_index, test_index in split.split(data, data[strat_column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set


def convert_to_category(val, categories):
    """
    Take a value and a list of category names, and return the category represented by the value
    :param val:  the value to convert
    :param categories: the possible categories that val could represent
    :return:
    """
    i = np.where(val == 1)[0]
    i = i[0]
    return categories[i]


class _CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A Transformer to one hot-encode a given category of data
    """
    def __init__(self):
        self.unique_categories = None

    def fit(self, X, y=None):  # pylint: disable=unused-argument,invalid-name
        """
        Satisfy sklearn class design. Basically do nothing
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X):  # pylint: disable=invalid-name
        """
        Take a list of category strings and change it into corresponding tuples of zeroes and ones
        """
        assert X is not None, "X is None?"
        cats_encoded, unique_cats = pd.factorize(X)
        encoder = OneHotEncoder()
        cat_1hot = encoder.fit_transform(cats_encoded.reshape(-1, 1)).toarray()
        self.unique_categories = unique_cats
        return cat_1hot


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A Transformer to one hot-encode a given category of data
    """
    def __init__(self):
        self.categories_ = None
        self._one_hot_encoder = OneHotEncoder(sparse=False)
        self._ordinal_encoder = OrdinalEncoder()

    def fit(self, X, y=None):  # pylint: disable=invalid-name,unused-argument
        """
        Satisfy sklearn, do nothing
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X):  # pylint: disable=invalid-name,no-self-use
        """
        Encode each column of X as one hot lists
        :param X:
        :return:
        """
        ordinal_data = self._ordinal_encoder.fit_transform(X)
        self.categories_ = self._ordinal_encoder.categories_
        return self._one_hot_encoder.fit_transform(ordinal_data)

    def inverse_transform(self, X):  # pylint: disable=invalid-name,no-self-use
        """
        Return the original data, given transformed data
        """
        ordinal_data = self._one_hot_encoder.inverse_transform(X)
        return self._ordinal_encoder.inverse_transform(ordinal_data)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select subset of columns of a given dataframe
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):  # pylint: disable=invalid-name,unused-argument
        """
        Not applicable, do nothing
        """
        return self

    def transform(self, X):  # pylint: disable=invalid-name
        """
        Select the required columns
        """
        return X[self.attribute_names].to_numpy()
