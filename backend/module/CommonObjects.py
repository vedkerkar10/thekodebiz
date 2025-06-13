from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN

import pandas as pd
import numpy as np
from pandas._libs.parsers import STR_NA_VALUES


def categorical_to_numeric(d):
    """
    Convert columns having categorical values to numerical values
    :param d: DataFrame containing columns that need to be converted to numerical
    :return: Returns DataFrame with columns having numerical values along with encoders
    """
    encoders = {}
    label_encoder = LabelEncoder()
    col_numeric = d.select_dtypes(include=np.number).columns.tolist()
    col_numeric.sort()
    print(f'Columns having "Numeric" values are: {col_numeric}')
    for col in d.columns:
        if d[col].dtype == 'object':  # Check if the column is categorical
            print(f'Started converting values in column {col} to numerical')
            d[col] = d[col].astype(str)
            encoders[col] = label_encoder.fit(d[col])
            d[col] = encoders[col].transform(d[col])
            print(f'Completed converting values in column {col} to numerical')
    col_dates = d.select_dtypes(include=['datetime64[ns]']).columns.to_list()
    print(f'Columns having "Date" values are: {col_dates}')
    # d.drop(columns=col_dates, inplace=True)

    return d, encoders, col_dates, col_numeric


def eda(d):  # Call this function on upload of first excel
    """
    Perform basic data cleansing operation. Remove duplicate rows and rows containing null values
    :param d: excel data in DataFrame format
    :return: returns cleaned data in json dict format and cleaned data with numerical values in json format along with encoders
    """

    # !!! Change to read from session storage Later !!!
    # Reading File From POST Request
    # file = request.files['data']
    # data = pd.read_excel(file)
    # data = pd.DataFrame.from_dict(d)

    data = d.copy()

    n_rows, n_cols, duplicate_rows, null_values = data.shape[0], data.shape[
        1], data.duplicated().sum(), data.isnull().sum()
    print(f'Total rows is: {n_rows} and total columns is {n_cols}')
    print(f'Duplicate rows is: {duplicate_rows}.')
    data_analysis = f'Total rows is: {n_rows} and total columns is {n_cols}\n'
    data_analysis += f'Duplicate rows is: {duplicate_rows}.\n'
    if duplicate_rows > 0:
        print('\nDeleting duplicate rows')
        data.drop_duplicates(inplace=True)
        print(
            f'Number of duplicate rows after removing duplicates is: {data.duplicated().sum()}')
        data_analysis += f'Number of duplicate rows after removing duplicates is: {data.duplicated().sum()}\n'
    if null_values.sum() > 0:
        print('\nThe columns having null values are as below:')
        data_analysis += f'\nThe columns having null values are as below:\n'
        col_names = []
        col_nulls = []
        for cols in data.columns:
            if null_values[cols] > 0:
                col_names.append(cols)
                col_nulls.append(null_values[cols])
        col_data = {'Column Name': col_names, 'Null Values': col_nulls}
        col_data_df = pd.DataFrame(col_data)
        print(col_data_df)
        data_analysis += col_data_df.to_string()
        data.dropna(inplace=True)
        null_vals = data.isnull().sum()
        print(
            f'\nNumber of rows with null values after removing nulls is: {str(null_vals.sum())}')
        data_analysis += f'\nNumber of rows with null values after removing nulls is: {str(null_vals.sum())}\n'
    else:
        print('There are no null values in the data')
        data_analysis += f'There are no null values in the data\n'
    data.reset_index(drop=True, inplace=True)
    print(
        f'Total rows after removing duplicates and rows with null values is: {data.shape[0]} and total columns is {data.shape[1]}')
    data_analysis += f'Total rows after removing duplicates and rows with null values is: {data.shape[0]} and total columns is {data.shape[1]}\n'
    data1 = data.copy()
    data_num, encoder, date_cols, numeric_columns = categorical_to_numeric(data1)
    # data_num, encoder, date_cols = categorical_to_numeric(data)
    return data, data_num, encoder, date_cols, data_analysis, numeric_columns


def read_data(file_name):
    """
    The function gets request.file as input and returns dataframe
    :param file_name: Read data from request.file. If format is csv - convert possible date columns to datetime.
    :return: DataFrame of data read from the file.
    """
    # file_ext = file_name.filename.split(".")[-1]
    # file_ext = file_name.split(".")[-1]
    file_ext = file_name.filename.split(".")[-1]
    print(f'File extension is: {file_ext}')

    accepted_na_values = STR_NA_VALUES - {'NA'}

    if file_ext == 'csv':
        df = pd.read_csv(file_name, keep_default_na=False, na_values=accepted_na_values)
        obj_cols = df.select_dtypes(include=['object']).columns.to_list()
        # df[obj_cols] = df[obj_cols].apply(pd.to_datetime, errors='ignore')
        for col in obj_cols:
            try:
                df[col] = pd.to_datetime(df[col], format='mixed')
                print(f'Converted "{col}" to date column')
            except:
                print(f'Column "{col}" is not a date column')
    else:
        df = pd.read_excel(file_name, keep_default_na=False, na_values=accepted_na_values)
    return df


def get_unique_values(d, features):
    values = d[features].to_dict(orient='list')
    unique_values = {k: list(set(v)) for k, v in values.items()}
    return unique_values


def colsTest(pred_df, train_df, target, features):
    # print()
    # =
    # # x.remove(target)

    return set(features.keys()).issubset(set(pred_df.columns))


def featuresTest(features, df):
    """this function checks if the unique values in the dataframe are chosen by the user during training

    Args:
        features:({key:Array<number>})
        df:(pandas.dataframe) dataframe holding the prediction file

    Returns:
        {key:OK|Error}
    """
    res = True
    result = {}
    for target in features.keys():
        is_subset = set(df[target].unique()).issuperset(set(features[target]))
        result[target] = False if not is_subset else True
        res = False if not is_subset else res
    return res


def aggregatorTest(df, aggregator):
    '''
    df:(pandas.dataframe) dataframe holding the prediction file
    aggregator:([]) list of aggregators
    '''
    date_columns = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col])
                date_columns.append(col)
            except ValueError:
                pass
    date_columns.extend(aggregator)
    return False if not set(date_columns).issubset(set(df.columns)) else True
