import os
import math
import warnings
import glob

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer


warnings.simplefilter(action="ignore", category=FutureWarning)

sns.set_theme(style="whitegrid")  # opsional, biar plot lebih rapi
plt.rcParams["figure.figsize"] = (8,5)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit


def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def preprocess(df):
  # dropna
  df.dropna(subset="Customer ID",axis=0,inplace=True)
  # removing canceled product
  df = df[~df.Invoice.str.contains('C',na=False)]
  # drop duplicates
  df = df.drop_duplicates()
  # Convert InvoiceDate to datetime
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
  # removing outlier
  replace_with_threshold(df,"Quantity")
  replace_with_threshold(df,"Price")

  return df
