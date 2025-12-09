import warnings
from datetime import timedelta

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    print("Removing missing values...")
    df.dropna(subset="Customer ID",axis=0,inplace=True)
    # removing canceled product
    print("Removing canceled transactions...")
    df = df[~df.Invoice.str.contains('C',na=False)]
    # drop duplicates
    print("Removing duplicate records...")
    df = df.drop_duplicates()
    # Convert InvoiceDate to datetime
    print("Converting InvoiceDate to datetime...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Handling negative values
    print("Removing negative values in Quantity and Price...")
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    # Data type validation
    print("Validating and converting data types...")
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df = df.dropna(subset=['Price', 'Quantity'])
    # Remove empty strings and whitespace
    print("Removing empty strings and whitespace in key columns...")
    df['Description'] = df['Description'].str.strip()
    df = df[df['Description'].str.len() > 0]
    # Remove zero values in critical columns
    print("Removing zero values in Quantity and Price...")
    df = df[(df['Quantity'] != 0) & (df['Price'] != 0)]
    # removing outlier
    print("Handling outliers in Quantity and Price columns...")
    replace_with_threshold(df,"Quantity")
    replace_with_threshold(df,"Price")
    # Feature engineering: create Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


def rfm_featuring(data: pd.DataFrame):
    """
    Creates RFM features from the input data.

    Parameters:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Dataframe with RFM features.
    """
    # Mengatur tanggal terakhir 2011-12-10 sebagai invoice date terakhir menjadi 2011-12-09.
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    latest_date = data["InvoiceDate"].max() + timedelta(days=1)
    # Membuat RFM features berdasarkan subset dari customerID
    RFM = data.groupby("Customer ID").agg(
        {
            "InvoiceDate": lambda x: (latest_date - x.max()).days, # hari sejak terakhir pembelian
            "Invoice": lambda x: x.nunique(),
            "Revenue": lambda x: x.sum()
        }
    )

    RFM["InvoiceDate"] = RFM["InvoiceDate"].astype(int)
    RFM.rename(columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "Revenue": "Monetary"}, inplace=True)
    Shopping_Cycle = data.groupby("Customer ID").agg({"InvoiceDate": lambda x: ((x.max() - x.min()).days)})
    RFM["Shopping_Cycle"] = Shopping_Cycle
    RFM["Interpurchase_Time"] = RFM["Shopping_Cycle"] // RFM["Frequency"]
    
    return RFM[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]]


def main():
    # Load data
    print("Starting data preprocessing...")
    DATA_PATH = './uci-retail_raw.csv'
    print("Loading data from:", DATA_PATH)
    df = pd.read_csv('./uci-retail_raw.csv')

    # Preprocess data
    df_cleaned = preprocess(df)
    # Feature engineering: RFM features
    print("Creating RFM features...")
    df_cleaned = rfm_featuring(df_cleaned)

    print("Saving cleaned data to './preprocessing/uci-retail_preprocessing.csv'...")
    df_cleaned.to_csv('./preprocessing/uci-retail_preprocessing.csv', index=False)


if __name__ == "__main__":
    main()