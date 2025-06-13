from sklearn.preprocessing import LabelEncoder


def categorical_to_numeric(d):
    """
    Convert columns having categorical values to numerical values
    :param d: DataFrame containing columns that need to be converted to numerical
    :return: Returns DataFrame with columns having numerical values
    """
    label_encoder = LabelEncoder()
    for col in d.columns:
        if d[col].dtype == 'object':  # Check if the column is categorical
            d[col] = label_encoder.fit_transform(d[col])
    return d
