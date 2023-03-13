import pandas as pd
from typing import List, Optional

from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# function that averages rides from previous 7, 14, 21, 28 days
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    '''Adds a feature column calculated by averaging rides from
    - previous 7 days
    - previous 14 days
    - previous 21 days
    - previous 28 days
    '''

    X_ = X.copy()
    X_['average_rides_last_4_weeks'] = X_[[f'sales_lag_7', f'sales_lag_14', f'sales_lag_21', f'sales_lag_28']].mean(axis=1)

    return X_

# convert function to sklearn transformer
add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate=False)


def extract_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    '''Extracts temporal features from the datetime index'''

    X_ = X.copy()
    X_['date'] = pd.to_datetime(X_['date'])

    X_['year'] = X_['date'].dt.year
    X_['month'] = X_['date'].dt.month
    X_['day'] = X_['date'].dt.day
    X_['dayofweek'] = X_['date'].dt.dayofweek
    X_['weekofyear'] = X_['date'].dt.weekofyear
    X_['quarter'] = X_['date'].dt.quarter
    X_['weekend'] = X_['dayofweek'].isin([5,6])
    X_['payday'] = X_['day'].isin([15]) | X_['date'].dt.is_month_end
    
    return X_.drop('date', axis=1)

# convert function to sklearn transformer
add_temporal_features = FunctionTransformer(extract_temporal_features, validate=False)


class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X, y=None):
        X_ = X.copy()
        return pd.DataFrame(super().transform(X_), columns=self.columns)
    

class FourierFeatures():
    '''add Fourier features to the data'''
    def __init__(self, fourier_pairs):
        self.fourier_pairs = fourier_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        fourier = CalendarFourier(freq="A", order=self.fourier_pairs)  # 10 sin/cos pairs for Annual seasonality

        # since all store_families have the same length, we can get Fourier Features for one store_family and then concatenate them for all store_families

        dp = DeterministicProcess(
            index=X_['date'].unique(),
            constant=False,               # dummy feature for bias (y-intercept)
            order=0,                     # trend (order 1 means linear)
            additional_terms=[fourier],  # annual seasonality (fourier)
            drop=False,                   # drop terms to avoid collinearity
        )

        X_train = dp.in_sample()  # create features for dates in tunnel.index
        X_ = X_.merge(X_train, left_on='date', right_index=True, how='left')

        return X_


class CustomMedianImputer(BaseEstimator, TransformerMixin):
    '''group by group_var and impute using median'''
    def __init__(self, columns, group, group_var=None):
        self.columns = columns
        self.group = group
        self.group_var = group_var

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df = df.astype('float64', errors='ignore')
        col = self.columns

        if self.group == True:
            df[col] = df[col].groupby(self.group_var).transform(lambda x: x.fillna(x.median()))

        elif self.group == False:
            df[col] = df[col].transform(lambda x: x.fillna(x.median()))

        return X