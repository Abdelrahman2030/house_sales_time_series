import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def sort_data(df):
    """
    This function sorts the data by date_sold and then resets the index of the data

    :param df: DataFrame

    :return: DataFrame
    """
    # Sort the data by date_sold
    df = df.sort_values(by="datesold")

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    return df


def main(df, neural_network=False, tree_ensemble=False):
    """
    This function preprocesses the data for the different models.
    It changes the columns names, fills missing values, adds a new column with the outliers,
    adds new features of time, drops columns, label encodes the bedrooms column, one hot encodes the categorical columns,
    scales the data

    :param df: DataFrame
    :param neural_network: bool
    :param tree_ensemble: bool

    :return: DataFrame
    """
    # Change columns names
    df.rename(
        columns={
            "datesold": "date_sold",
            "postcode": "post_code",
            "bedrooms": "bedrooms",
            "propertyType": "property_type",
        },
        inplace=True,
    )

    # Fill missing values in bedrooms column with the median
    df["bedrooms"].replace(0, df["bedrooms"].median(), inplace=True)

    # Add a new column with the outliers
    price_zscore = stats.zscore(df["price"])
    df["outlier"] = np.where(np.abs(price_zscore) > 3, 1, 0)

    # Add new feautres of time
    df["date_sold"] = pd.to_datetime(df["date_sold"])
    df["day_of_week"] = df["date_sold"].dt.dayofweek
    df["year"] = df["date_sold"].dt.year
    df["month"] = df["date_sold"].dt.month

    # Order the index of the data
    df = df.set_index("date_sold").sort_index()

    # Drop index
    df.reset_index(drop=True, inplace=True)

    # Label encode the bedrooms column
    le = LabelEncoder()
    df["year"] = le.fit_transform(df["year"])

    # Create an if loop for the different models
    # If the model is a neural network, it will use one hot encoding
    # If the model is a tree ensemble, it will use label encoding

    if neural_network:
        # One hot encoding
        df = pd.get_dummies(df, columns=["property_type"], drop_first=True)
        df = pd.get_dummies(df, columns=["post_code"], drop_first=True)
        df = pd.get_dummies(df, columns=["day_of_week"], drop_first=True)
        df = pd.get_dummies(df, columns=["month"], drop_first=True)

        df = df.astype("int")

    elif tree_ensemble:
        # Label encoding
        df["property_type"] = le.fit_transform(df["property_type"])
        df["post_code"] = le.fit_transform(df["post_code"])

    # Create if loop for the different models
    # If the model is a neural network, it will scale the data
    # If the model is a tree ensemble, it will not scale the data

    if neural_network:
        # normalise the data
        df["price"] = np.log1p(df["price"])

        # Scale the data
        scaler = StandardScaler()
        df["price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))

    return df
