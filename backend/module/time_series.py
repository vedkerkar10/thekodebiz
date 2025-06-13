import json
import os
import itertools
import datetime
import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout
from keras.layers import Input
from keras.optimizers import Adam
import tensorflow as tf
import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Blueprint, request, json
from darts.models import Prophet
from darts import TimeSeries
import module.CommonObjects as co
from .config import public

timeSeries = Blueprint("timeSeries", __name__)
target_data = []
test_plot_data = []
testing_data = []

testing_df_data = []
training_df_data = []

models = ["arima", "lstm", "gru", "Prophet"]
# main()
os.environ["NIXTLA_ID_AS_COL"] = "True"


def date_transform(df, granularity):
    """
    Adds additional date columns to the input DataFrame according to granularity

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'date' column.
        granularity (str): Granularity level for date transformation ('D', 'M', 'Y', 'Q', 'H').

    Returns:
        pd.DataFrame: Transformed DataFrame with additional columns based on the specified granularity.
    """
    df[aggregate_date[0]] = pd.to_datetime(df[aggregate_date[0]])
    if granularity == "D":
        df["day_of_week"] = df[aggregate_date[0]].dt.dayofweek
        df["day_of_month"] = df[aggregate_date[0]].dt.day
        df["M"] = df[aggregate_date[0]].dt.month
        df["Y"] = df[aggregate_date[0]].dt.year
    elif granularity == "M":
        df["M"] = df[aggregate_date[0]].dt.month
        df["Y"] = df[aggregate_date[0]].dt.year
    elif granularity == "Y":
        df["Y"] = df[aggregate_date[0]].dt.year
    elif granularity == "Q":
        df["Q"] = df[aggregate_date[0]].dt.quarter
        df["Y"] = df[aggregate_date[0]].dt.year
    elif granularity == "H":
        df["H"] = df[aggregate_date[0]].apply(lambda x: 1 if x.quarter in (1, 2) else 2)
        df["Y"] = df[aggregate_date[0]].dt.year
    return df


def granularity_control_df(df, granularity, feature, date_list):
    """
    Aggregates target based on the granularity.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        granularity (str): Granularity level for controlling the DataFrame ('D', 'M', 'Y', 'Q', 'H').
        feature (list): List of features to be grouped by.
        date_list (list): List of date columns.

    Returns:
        list: List of DataFrames with controlled granularity.
    """

    dfs = []
    unique_values = df[feature].apply(lambda x: tuple(x), axis=1).unique().tolist()
    for i, val in enumerate(unique_values):
        conditions = [(df[f] == val[feature.index(f)]) for f in feature]
        non_gran_df = df[np.logical_and.reduce(conditions)]
        dfs.append(
            non_gran_df.groupby(date_list)
            .agg(
                {
                    target: "sum",
                    aggregate_date[0]: "last",
                    **{f: "last" for f in feature},
                }
            )
            .reset_index()
        )

    return dfs


def df_split(dfs, date_list, feature, forecast):
    """
    Splits the input DataFrames into training and testing sets in diff formats for later functions.

    Parameters:
        dfs (list): List of DataFrames.
        date_list (list): List of date-related columns.
        feature (list): List of features.
        forecast (bool): Whether forecasting is performed.
    Returns:
        tuple: Forecast training and testing data, training and testing DataFrame data, training and testing data, test plot data.
    """
    df_split_training_data = []
    df_split_testing_data = []
    df_split_testing_df_data = []
    df_split_training_df_data = []
    for ddfs in dfs:
        if forecast:
            train_set = ddfs
            test_set = ddfs
        else:
            train_set, test_set = train_test_split(ddfs, train_size=0.80, shuffle=False)
        df_split_testing_df_data.append(test_set)
        # data frame format for arima model
        df_split_training_df_data.append(train_set)
        df_split_testing_data.append(test_set[date_list + feature + [target]].values)
        df_split_training_data.append(train_set[date_list + feature + [target]].values)
    df_split_forecast_training_data = np.concatenate(
        df_split_training_data, axis=0
    )  # numpy array
    df_split_forecast_testing_data = np.concatenate(
        df_split_testing_data, axis=0
    )  # numpy array
    return (
        df_split_forecast_training_data,
        df_split_forecast_testing_data,
        df_split_training_df_data,
        df_split_testing_df_data,
        df_split_training_data,
    )


MODEL = ""


@timeSeries.route("/start", methods=["POST"])
def train():
    print("/start")
    print("train called")
    global MODEL, epochs
    data = request.files["file"]
    features = json.loads(request.form.get("features"))
    frequency = request.form.get("frequency")
    target = request.form.get("target")
    granularity = request.form.get("granularity")
    agg_date = json.loads(request.form.get("agg_date"))
    plot_vals = json.loads(request.form.get("uniqueValues"))
    epochs = json.loads(request.form.get("epochs"))

    df = co.read_data(data)

    # plot_vals = {'store': [1, 2, 3], 'item': [1, 2, 3, 4]}

    print("Exploratory data analysis")
    df_agg_gran, training_df_data, testing_df_data = eda_ts(
        df, granularity, frequency, features, [], agg_date, target, False, plot_vals
    )

    # #no 2
    print("Building Model")
    eval_df = build_model_ts(
        copy.deepcopy(forecast_training_data),
        copy.deepcopy(forecast_testing_data),
        copy.deepcopy(training_df_data),
        copy.deepcopy(testing_df_data),
        copy.deepcopy(training_data),
    )
    print("Model Built")
    # eval_df.to_csv('eval_df.csv')
    # no 3
    # plot_results(models,feature_scaler,target_fitted_scaler,num_features,plot_values)
    MODEL = eval_df["Model Name"][0].lower()
    graphs = plot_results(
        models, feature_scaler, target_fitted_scaler, num_features, plot_values
    )
    # 'graphs':[list(row) for row in zip(*graphs)],'graphs_df':graph_dfs.to_json(orient='table',index=False)
    print("returning train")
    return {
        "EVAL": eval_df.to_json(orient="table", index=False),
        "graphs": json.dumps(graphs["graphs"]),
        "graphs_df": graphs["graphs_df"],
    }


@timeSeries.route("/Predict", methods=["POST"])
def Predict_TS():
    print("/Predict")
    print("Predict_TS called")
    global target, aggregate_date, combinations, encoders

    train_file = request.files["trainingFile"]
    predict_file = request.files["predictionFile"]

    train_df = co.read_data(train_file)  # pd.read_csv('train.csv')
    test_df = co.read_data(predict_file)  # pd.read_csv('predict_sales.csv')

    test_df[target] = 0
    print("Calling predict_values_ts")
    pred_df = predict_values_ts(train_df, test_df, MODEL)

    pred_df.to_csv(f"predicted_{target}.csv", index=False)

    print(pd.read_csv(f"predicted_{target}.csv"))
    """schema `graphs`
        [
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
        ]
    """

    print("pred_df", pred_df)
    print("aggregate_date", aggregate_date)
    print("target", target)
    print("combinations", combinations)
    print("encoders", encoders)
    graphs = [
        # [{
        #     "y": pred_df[target].to_list(),
        #     "x": pred_df[aggregate_date[0]].dt.strftime('%Y-%m-%d').to_list(),
        #     "label": "Prediction"
        # }]
    ]

    print("Generating Combination Display")
    selected_dfs = []
    print("Combinations", combinations)
    for comb in combinations:
        if len(encoders.keys()) > 0:
            # print('pred_df',pred_df['item'])
            conditions = [
                (pred_df[key] == encoders[key].inverse_transform(value).tolist()[0])
                for key, value in comb.items()
            ]
        else:
            conditions = [(pred_df[key] != key) for key, value in comb.items()]
        print(conditions)
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition
        selected_df = pred_df[combined_condition]
        selected_dfs.append(selected_df)
    print(selected_df)

    print("Generating Graph Display")
    for sel_df in selected_dfs:
        if not sel_df.empty:
            print(sel_df[agg_features])
            result = ""
            for i, header in enumerate(agg_features):
                value = sel_df[agg_features].iloc[1][header]
                result += f"{header} : {value} "
            result = result.strip()
            graphs.append(
                [
                    {
                        "y": sel_df[target].to_list(),
                        "x": sel_df[aggregate_date[0]]
                        .dt.strftime("%Y-%m-%d")
                        .to_list(),
                        "label": "Prediction",
                        "values": result,
                    }
                ]
            )

    # pred_df[target] = pred_df[target].apply(lambda x: [x])
    pred_df[target] = pred_df[target]
    column_names = aggregate_date + agg_features + [target]
    print("Returning")
    return {
        "Final": pred_df[column_names].to_json(orient="table", index=False),
        "graphs": json.dumps(graphs),
    }


# @timeSeries.route('/plot')
def plot_forecast(
    all_predicted_targets,
    all_actual_sales,
    models,
    num_steps_forward,
    plot_dates,
    combination_values,
):
    """
    Plots the forecasted and actual sales.

    Parameters:
        all_predicted_targets (list): List of predicted targets.
        all_actual_sales (np.ndarray): Array of actual sales.
        models (list): List of model names.
        num_steps_forward (int): Number of steps forward to plot.
    """
    """schema `graphs`
        [
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
            [
                {x:[],label:'a'},
                {x:[],label:'b'}
            ],
        ]
    """
    # print('plot_forecast called')
    graphs = []
    # graph_dfs = []
    list_graph_df = []
    for arima_preds, lstm_preds, gru_preds, prophet_preds, combination_value, all_actual_sale in zip(
        all_predicted_targets[0],
        all_predicted_targets[1],
        all_predicted_targets[2],
        all_predicted_targets[3],
        combination_values,
        all_actual_sales,
    ):
        temp_x = plot_dates
        arima_temp = np.array(arima_preds).tolist()
        lstm_temp = np.array(lstm_preds).tolist()
        gru_temp = np.array(gru_preds).tolist()
        prophet_temp = np.array(prophet_preds).tolist()

        graph_df = pd.DataFrame(
            list(
                zip(temp_x, arima_temp, lstm_temp, gru_temp, prophet_temp, np.array(all_actual_sale))
            ),
            columns=[
                "Date",
                f"ARIMA Predicted {target}",
                f"LSTM Predicted {target}",
                f"GRU Predicted {target}",
                f"Prophet Predicted {target}",
                f"Actual {target}",
            ],
        )

        for key, value in combination_value.items():
            if key in cat_feats:
                graph_df[key] = encoders[key].inverse_transform([value]).tolist()[0]
            else:
                graph_df[key] = value
        list_graph_df.append(graph_df)

    stat_df = pd.concat(list_graph_df, axis=0)
    column_list = list(combination_value.keys())
    column_list = (
        ["Date"]
        + column_list
        + [
            f"Actual {target}",
            f"ARIMA Predicted {target}",
            f"LSTM Predicted {target}",
            f"GRU Predicted {target}",
            f"Prophet Predicted {target}",
        ]
    )
    stat_df = stat_df[column_list]
    stat_df.to_csv("statdf.csv")

    print(stat_df.to_string())
    for n, (predicted_target, model_name) in enumerate(
        zip(all_predicted_targets, models)
    ):
        graph = []
        temp_x = []
        temp_y = []
        for m, predicted_target_feature in enumerate(zip(predicted_target)):
            temp_x.extend(plot_dates)
            temp_y.extend(np.array(predicted_target_feature).reshape(-1, 1).tolist())

            final_list = []
            for key, value in combination_values[m].items():
                if key in cat_feats:
                    final_list.append(
                        f"{key} : {encoders[key].inverse_transform([value])[0]}"
                    )
                else:
                    final_list.append(f"{key} : {value}")
            graph.append(
                {
                    "y": np.array(predicted_target_feature).reshape(-1, 1).tolist(),
                    "x": plot_dates,
                    "label": f"{model_name} Predicted {target}",
                    "values": final_list,
                }
            )
        # graph_dfs.append(pd.DataFrame(list(zip([x[0] for x in temp_y], temp_x, np.array(all_actual_sales[:num_steps_forward])[0])), columns=[
        #     f'{model_name} Predicted {target}', 'Date', 'Actual']))
        """Features(eg store,item) Aggregator Actual ...Predictions(arima lstm gru)"""
        graphs.append(graph)

    graphs.append(
        [
            {
                "y": np.array(all_actual_sales[:num_steps_forward]).tolist()[i],
                "x": plot_dates,
                "label": f"Actual {target}",
                # 'values': combination_values[m]
            }
            for i, _ in enumerate(zip(predicted_target))
        ]
    )
    # merged_df = [pd.concat(graph_dfs,axis=0).to_json(orient='table', index=False)]
    return {
        "graphs": [list(row) for row in zip(*graphs)],
        "graphs_df": stat_df.to_json(orient="table", index=False),
    }


def eda_ts(
    df,
    GRANULARITY="D",
    gran="",
    feature=[],
    other_feats=[],
    agg_date="",
    target_var="",
    forecast=False,
    plot_vals={},
):
    global \
        date_list, \
        arima_sequence_length, \
        nn_sequence_length, \
        num_features, \
        def_gran, \
        granularity, \
        target, \
        plot_values, \
        aggregate_date, \
        other_features, \
        agg_features, \
        cat_feats
    global \
        forecast_training_data, \
        forecast_testing_data, \
        training_df_data, \
        testing_df_data, \
        training_data, \
        test_plot_data, \
        valid_test_feats, \
        encoders

    granularity = GRANULARITY
    def_gran = gran
    agg_features = feature
    other_features = other_feats
    aggregate_date = agg_date
    target = target_var
    plot_values = plot_vals
    cat_feats = []
    encoders = {}
    for col in agg_features:
        print(f"{col} : type --> {df[col].dtype}")
        if df[col].dtype == "object":
            cat_feats.append(col)
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col].values)
            encoders[col] = label_encoder
            print(f'Assigned Encoder for "{col}"')

    valid_test_feats = aggregate_date + agg_features + other_feats

    if granularity == "D":
        date_list = ["Y", "M", "day_of_month", "day_of_week"]
        with open("seasonality.json") as f:
            json_data = json.load(f)
        arima_sequence_length = json_data["Daily"]["ARIMA"]
        nn_sequence_length = json_data["Daily"]["LSTM"]
    elif granularity == "M":
        date_list = ["Y", "M"]
        arima_sequence_length = 12
        nn_sequence_length = 12
    elif granularity == "Y":
        date_list = ["Y"]
        arima_sequence_length = 5
        nn_sequence_length = 5
    elif granularity == "Q":
        date_list = ["Y", "Q"]
        arima_sequence_length = 4
        nn_sequence_length = 4
    elif granularity == "H":
        date_list = ["Y", "H"]
        arima_sequence_length = 2
        nn_sequence_length = 2

    num_features = len(date_list) + len(agg_features) + 1

    for col, value in plot_values.items():
        if col in cat_feats:
            df = df[df[col].isin(encoders[col].transform(value))]
        else:
            df = df[df[col].isin(value)]
    aggregate_fn_list = aggregate_date + feature

    start_time = datetime.datetime.now()
    # print("Start Time for feature aggregation:", start_time.time().strftime("%M:%S:%f"))
    df_aggregated = df.groupby(aggregate_fn_list).agg({target_var: "sum"}).reset_index()
    end_time = datetime.datetime.now()
    # print("End Time for feature aggregation:", end_time.time().strftime("%M:%S:%f"))

    start_time = datetime.datetime.now()
    # print("Start Time for date transform:", start_time.time().strftime("%M:%S:%f"))
    df_agg_columns = date_transform(df_aggregated, granularity)
    end_time = datetime.datetime.now()
    # print("End Time for date transform:", end_time.time().strftime("%M:%S:%f"))

    start_time = datetime.datetime.now()
    # print("Start Time for granularity:", start_time.time().strftime("%M:%S:%f"))
    df_agg_gran = granularity_control_df(
        df_agg_columns, granularity, feature, date_list
    )
    end_time = datetime.datetime.now()
    # print("End Time for granularity:", end_time.time().strftime("%M:%S:%f"))

    start_time = datetime.datetime.now()
    # print("Start Time for df_split:", start_time.time().strftime("%M:%S:%f"))
    (
        forecast_training_data,
        forecast_testing_data,
        training_df_data,
        testing_df_data,
        training_data,
    ) = df_split(df_agg_gran, date_list, feature, False)
    end_time = datetime.datetime.now()
    # print("End Time for df_split:", end_time.time().strftime("%M:%S:%f"))
    # return "done"
    return df_agg_gran, training_df_data, testing_df_data


# function 2


def build_model_ts(
    forecast_training_data,
    forecast_testing_data,
    training_df_data,
    testing_df_data,
    training_data,
    models=["LSTM", "GRU", "ARIMA", "Prophet"],
):
    """
    Builds a time series model with eval df.

    Parameters:
        forecast_training_data (np.ndarray): Forecast training data.
        forecast_testing_data (np.ndarray): Forecast testing data.
        training_df_data (list): List of training DataFrame data.
        testing_df_data (list): List of testing DataFrame data.
        training_data (list): List of training data.
        models (list): List of models to use ('LSTM', 'GRU', 'ARIMA').
        forecast (bool): Whether forecasting is performed.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]]: Evaluation DataFrame if forecast is False,
        otherwise tuple containing feature scaler and target fitted scaler.
    """
    global final_sequence, feature_scaler, target_fitted_scaler

    forecast_training_data_scaled, forecast_testing_data_scaled, feature_scaler = (
        normalise(forecast_training_data, forecast_testing_data)
    )
    temp = np.concatenate((forecast_training_data_scaled, forecast_testing_data_scaled))

    train_sequences = create_sequences(
        forecast_training_data_scaled, nn_sequence_length
    )
    test_sequences = create_sequences(temp, nn_sequence_length)

    np.random.shuffle(train_sequences)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_fitted_scaler = target_scaler.fit(
        forecast_training_data[:, -1].reshape(-1, 1)
    )

    final_sequence = end_sequence(feature_scaler, training_data, nn_sequence_length)
    train_data, test_data = train_sequences, test_sequences

    X_train, y_train = train_data[:, :-1, :], train_data[:, -1:, -1]
    test_length = -len(testing_df_data[0])
    X_test, y_test = test_data[test_length:, :-1, :], test_data[test_length:, -1:, -1]

    metrics = [
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
        tf.keras.metrics.CosineSimilarity(name="cosine_similarity"),
        tf.keras.metrics.LogCoshError(name="logcosh"),
    ]
    print("Building LSTM")
    lstm_metrics = build_dl_model(
        nn_sequence_length,
        num_features,
        X_train,
        y_train,
        X_test,
        y_test,
        metrics,
        "lstm",
    )

    print("Building GRU")
    gru_metrics = build_dl_model(
        nn_sequence_length,
        num_features,
        X_train,
        y_train,
        X_test,
        y_test,
        metrics,
        "gru",
    )

    print("Building ARIMA")
    arima_metrics = build_arima(
        arima_sequence_length,
        num_features,
        training_df_data,
        testing_df_data,
        metrics,
        target_fitted_scaler,
    )
    
    print("Building Prophet")
    prophet_metrics = build_prophet(
        training_df_data,
        testing_df_data,
        metrics,
        target_fitted_scaler,
    )

    metrics_lists = [lstm_metrics, gru_metrics, arima_metrics, prophet_metrics]
    eval_df = pd.DataFrame()

    for model_name, metrics_list in zip(models, metrics_lists):
        metrics_dict = {
            "Model Name": model_name,
            "MeanSquaredError": metrics_list[0],
            "RootMeanSquaredError": metrics_list[1],
            "MeanAbsoluteError": metrics_list[2],
            "MeanAbsolutePercentageError": metrics_list[3],
            "MeanSquaredLogarithmicError": metrics_list[4],
            "LogCoshError": metrics_list[5],
        }

        new_row = pd.DataFrame(metrics_dict, index=[0])
        eval_df = pd.concat([eval_df, new_row], ignore_index=True)

    eval_df = eval_df.sort_values(
        by="RootMeanSquaredError", ascending=True
    ).reset_index(drop=True)
    eval_df["rank"] = eval_df.index + 1
    return eval_df


# function 3
# @timeSeries.route('/plot')


def plot_results(
    models, feature_scaler, target_fitted_scaler, num_features, plot_values
):
    """
    Plots the results of the models.

    Parameters:
        models (list): List of model names.
        feature_scaler (MinMaxScaler): Feature scaler.
        target_fitted_scaler (MinMaxScaler): Target fitted scaler.
        num_features (int): Number of features.
        plot_values (dict): Dictionary containing values for plotting.
    """
    global num_steps_forward, combinations
    model_predictions = []
    all_actual = []
    keys = list(plot_values.keys())
    values = plot_values.values()
    combinations = []
    for val in itertools.product(*values):
        combination = {}
        for i in range(len(keys)):
            if keys[i] in cat_feats:
                combination[keys[i]] = encoders[keys[i]].transform([val[i]])
            else:
                combination[keys[i]] = val[i]
        combinations.append(combination)
    combination_values = []
    for model in models:
        feature_predictions = []
        actual = []
        # plot_dates=[]
        for val, actual_target, test_df, train_df in zip(
            final_sequence, testing_df_data, testing_df_data, training_df_data
        ):
            if (
                dict(zip(agg_features, train_df[agg_features].values[0]))
                in combinations
            ):
                num_steps_forward = len(test_df)
                predicted_target = forecast(
                    val[1:],
                    num_steps_forward,
                    actual_target[target].values,
                    test_df,
                    train_df,
                    model,
                    feature_scaler,
                    target_fitted_scaler,
                    num_features,
                )
                feature_predictions.append(predicted_target)
                plot_dates = [
                    pd.Timestamp(date).to_pydatetime().strftime("%Y-%m-%d")
                    for date in test_df[aggregate_date[0]].values
                ]
                combination_values.append(
                    dict(zip(agg_features, train_df[agg_features].values[0]))
                )
                actual.append(actual_target[target].values)
        model_predictions.append(feature_predictions)

    return plot_forecast(
        model_predictions,
        actual,
        models,
        num_steps_forward,
        plot_dates,
        combination_values,
    )


def predict_values_ts(train_df, test_df, model):
    """
    Predicts values on actual test set.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        model: Model for forecasting.

    Returns:
        Tuple[list, pd.DataFrame]: Predictions and concatenated DataFrame.
    """
    aggregate_fn_list = aggregate_date + agg_features

    # for col, value in plot_values.items():
    #     if(not test_df[col].isin(value).any()):
    #         print("values not present")
    #         return False

    for col, value in plot_values.items():
        train_df = train_df[train_df[col].isin(value)]
        test_df = test_df[test_df[col].isin(value)]

    for col in cat_feats:
        train_df[col] = encoders[col].transform(train_df[col].values)
        test_df[col] = encoders[col].transform(test_df[col].values)

    # if not set(valid_test_feats).issubset(set(test_df.columns.tolist())):
    #     print(set(test_df.columns.tolist()),set(valid_test_feats))
    #     print('columns not present')
    #     return False
    test_df[target] = 0

    # df_aggregated = train_df.groupby(aggregate_fn_list).agg({target: 'sum'}).reset_index()
    forecast_df_agg_gran_columns = date_transform(train_df, granularity)
    forecast_df_agg_gran = granularity_control_df(
        forecast_df_agg_gran_columns, granularity, agg_features, date_list
    )

    test_df_agg_gran_columns = date_transform(test_df, granularity)
    test_df_agg_gran = granularity_control_df(
        test_df_agg_gran_columns, granularity, agg_features, date_list
    )

    # global forecasted_testing_df_data

    (
        pred_forecasted_training_data,
        _,
        pred_forecasted_training_df_data,
        _,
        pred_forecast_training_data,
    ) = df_split(forecast_df_agg_gran, date_list, agg_features, True)
    _, pred_forecasted_testing_data, _, pred_forecasted_testing_df_data, _ = df_split(
        test_df_agg_gran, date_list, agg_features, True
    )

    # global final_sequence
    forecast_training_data_scaled, forecast_testing_data_scaled, feature_scaler = (
        normalise(pred_forecasted_training_data, pred_forecasted_testing_data)
    )
    temp = np.concatenate((forecast_training_data_scaled, forecast_testing_data_scaled))

    train_sequences = create_sequences(
        forecast_training_data_scaled, nn_sequence_length
    )
    test_sequences = create_sequences(temp, nn_sequence_length)

    np.random.shuffle(train_sequences)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_fitted_scaler = target_scaler.fit(
        pred_forecasted_training_data[:, -1].reshape(-1, 1)
    )

    final_sequence = end_sequence(
        feature_scaler, pred_forecast_training_data, nn_sequence_length
    )
    train_data, test_data = train_sequences, test_sequences

    X_train, y_train = train_data[:, :-1, :], train_data[:, -1:, -1]
    test_length = -len(pred_forecasted_testing_df_data[0])
    X_test, y_test = test_data[test_length:, :-1, :], test_data[test_length:, -1:, -1]
    if model == "lstm":
        forecast_dl(
            nn_sequence_length,
            num_features,
            X_train,
            y_train,
            target_fitted_scaler,
            X_test,
            "lstm",
        )
    elif model == "gru":
        forecast_dl(
            nn_sequence_length,
            num_features,
            X_train,
            y_train,
            target_fitted_scaler,
            X_test,
            "gru",
        )
    elif model == "arima":
        forecast_arima(
            arima_sequence_length,
            num_features,
            pred_forecasted_training_df_data,
            pred_forecasted_testing_df_data,
        )
    elif model == "Prophet":
        forecast_prophet(
            pred_forecasted_training_df_data,
            pred_forecasted_testing_df_data,
        )

    preds = []
    global num_steps_forward
    keys = list(plot_values.keys())
    values = plot_values.values()

    combinations = []
    for val in itertools.product(*values):
        combination = {}
        for i in range(len(keys)):
            if keys[i] in cat_feats:
                combination[keys[i]] = encoders[keys[i]].transform([val[i]])
            else:
                combination[keys[i]] = val[i]
        combinations.append(combination)

    for val, actual_target, test_df, train_df in zip(
        final_sequence,
        pred_forecasted_testing_df_data,
        pred_forecasted_testing_df_data,
        pred_forecasted_training_df_data,
    ):
        if dict(zip(agg_features, train_df[agg_features].values[0])) in combinations:
            num_steps_forward = len(test_df)
            predictions = forecast(
                val[1:],
                num_steps_forward,
                actual_target[target].values,
                test_df,
                train_df,
                model,
                feature_scaler,
                target_fitted_scaler,
                num_features,
            )
            preds.append(predictions)
    if def_gran == "D":
        def_date_list = ["Y", "M", "day_of_month", "day_of_week"]
    elif def_gran == "M":
        def_date_list = ["Y", "M"]
    elif def_gran == "Y":
        def_date_list = ["Y"]
    elif def_gran == "Q":
        def_date_list = ["Y", "Q"]
    elif def_gran == "H":
        def_date_list = ["Y", "H"]

    pred_dfs = []
    for pred, single_test_df_agg_gran in zip(preds, test_df_agg_gran):
        temp_df = date_transform(single_test_df_agg_gran, def_gran)
        temp_df[target] = pred
        pred_dfs.append(
            temp_df.groupby(def_date_list)
            .agg(
                {
                    target: "sum",
                    aggregate_date[0]: "last",
                    **{f: "last" for f in agg_features},
                }
            )
            .reset_index()
        )

    concat_df = pd.concat(pred_dfs, ignore_index=True)

    for col in cat_feats:
        concat_df[col] = encoders[col].inverse_transform(concat_df[col].values)

    concat_df.to_csv("concated_predicted_sales.csv", index=False)

    return concat_df


# sub functions 2
def normalise(forecast_training_data, forecast_testing_data):
    """
    Normalizes the forecast training and testing data.

    Parameters:
        forecast_training_data (np.ndarray): Forecast training data.
        forecast_testing_data (np.ndarray): Forecast testing data.

    Returns:
        np.ndarray, np.ndarray, MinMaxScaler: Normalized forecast training and testing data, and feature scaler.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    forecast_training_data_scaled = feature_scaler.fit_transform(forecast_training_data)
    forecast_testing_data_scaled = feature_scaler.transform(forecast_testing_data)
    return forecast_training_data_scaled, forecast_testing_data_scaled, feature_scaler


def create_sequences(data, seq_length):
    """
    Creates sequences from the input data.

    Parameters:
        data (np.ndarray): Input data.
        seq_length (int): Length of each sequence.

    Returns:
        np.ndarray: Sequences of data.
    """
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


def end_sequence(feature_scaler, training_data, nn_sequence_length):
    """
    Generates the final sequence where train set ends.

    Parameters:
        feature_scaler (MinMaxScaler): Feature scaler.
        training_data (list): List of training data.
        sequence_length (int): Length of each sequence.

    Returns:
        list: Final sequence.
    """
    final_sequence = []
    for td in training_data:
        td_scaled = feature_scaler.transform(td)
        sequenced_td = create_sequences(td_scaled, nn_sequence_length)
        final_sequence.append(sequenced_td[-1])
    return final_sequence


def build_dl_model(
    nn_sequence_length,
    num_features,
    X_train,
    y_train,
    X_test,
    y_test,
    metrics,
    model_name,
):
    """
    Builds deep learning models.

    Parameters:
        sequence_length (int): Length of each sequence.
        num_features (int): Number of features.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        X_test (np.ndarray): Testing input data.
        y_test (np.ndarray): Testing target data.
        metrics (list): List of metrics.
        model_name (str): Name of the model.

    Returns:
        list: List of evaluation metrics.
    """
    global epochs
    model_checkpoint = ModelCheckpoint(
        filepath=f"{public}/models/{model_name}"
        + "_".join(
            [f"{key}{''.join(map(str, value))}" for key, value in plot_values.items()]
        )
        + ".keras",
        monitor="val_mape",
        save_best_only=True,
        mode="min",
        verbose=0,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
    lr_reducer_callback = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=1e-7, verbose=0
    )
    model = Sequential()
    if model_name == "gru":
        model.add(Input((nn_sequence_length - 1, num_features)))
        model.add(GRU(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=64, return_sequences=False))
    elif model_name == "lstm":
        model.add(Input((nn_sequence_length - 1, num_features)))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=8))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer=Adam(0.01), loss="mean_squared_error", metrics=metrics)
    # NOTE have to replace epochs back to original 100
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint, early_stopping_callback, lr_reducer_callback],
    )

    metrics = model.evaluate(X_test, y_test, verbose=0)
    return metrics[1:]


def build_arima(
    arima_sequence_length,
    num_features,
    training_df_data,
    testing_df_data,
    metrics,
    target_fitted_scaler,
):
    """
    Builds an ARIMA model.

    Parameters:
        sequence_length (int): Length of each sequence.
        num_features (int): Number of features.
        training_df_data (list): List of training DataFrame data.
        testing_df_data (list): List of testing DataFrame data.
        metrics (list): List of metrics.
        target_fitted_scaler (MinMaxScaler): Target fitted scaler.

    Returns:
        list: List of evaluation metrics.
    """
    seasonality = arima_sequence_length
    duration_dict = {"D": "D", "M": "ME", "Y": "Y", "Q": "3ME", "H": "6ME"}
    duration = duration_dict[granularity]
    # NOTE Set season_length to seasonality
    sf = StatsForecast(models=[AutoARIMA(season_length=seasonality)], freq=duration)

    # for df in training_df_data:
    #     df['ds'] = pd.to_datetime(df[aggregate_date[0]])
    #     df['unique_id'] = df[agg_features[0]]
    #     df['y'] = df[target]

    concatenated_df = pd.concat(training_df_data, ignore_index=True)
    concatenated_df["unique_id"] = concatenated_df[agg_features].apply(
        lambda row: " ".join(map(str, row)), axis=1
    )
    concatenated_df["ds"] = pd.to_datetime(concatenated_df[aggregate_date[0]])
    concatenated_df["y"] = concatenated_df[target]
    sf.fit(concatenated_df[["ds", "unique_id", "y"]])

    with open(
        f"{public}/models/arima"
        + "_".join(
            [f"{key}{''.join(map(str, value))}" for key, value in plot_values.items()]
        )
        + ".pickle",
        "wb",
    ) as file:
        pickle.dump(sf, file)

    mse_results = []
    rmse_results = []
    mae_results = []
    mape_results = []
    msle_results = []
    cosine_similarity_results = []
    logcosh_results = []
    for train_df, test_df in zip(training_df_data, testing_df_data):
        train_df["ds"] = pd.to_datetime(train_df[aggregate_date[0]])
        train_df["unique_id"] = train_df[agg_features].apply(
            lambda row: " ".join(map(str, row)), axis=1
        )
        train_df["y"] = train_df[target]
        predictions = sf.forecast(
            df=train_df[["ds", "unique_id", "y"]], h=len(test_df), level=[95]
        )["AutoARIMA"].values
        actual = test_df[target].values
        metric_results = []
        scaled_actual = target_fitted_scaler.transform(actual.reshape(-1, 1)).flatten()
        scaled_predictions = target_fitted_scaler.transform(
            predictions.reshape(-1, 1)
        ).flatten()
        mse = tf.keras.metrics.MeanSquaredError()
        mse.update_state(scaled_actual, scaled_predictions)
        mse_results.append(mse.result().numpy())

        # rmse = keras.metrics.RootMeanSquaredError()
        # rmse.update_state(scaled_actual, scaled_predictions)
        rmse_results.append(np.sqrt(mse.result().numpy()))
        mae = tf.keras.metrics.MeanAbsoluteError()
        mae.update_state(scaled_actual, scaled_predictions)
        mae_results.append(mae.result().numpy())

        mape = tf.keras.metrics.MeanAbsolutePercentageError()
        mape.update_state(scaled_actual, scaled_predictions)
        mape_results.append(mape.result().numpy())

        msle = tf.keras.metrics.MeanSquaredLogarithmicError()
        msle.update_state(scaled_actual, scaled_predictions)
        msle_results.append(msle.result().numpy())

        cosine_similarity = tf.keras.metrics.CosineSimilarity()
        cosine_similarity.update_state(scaled_actual, scaled_predictions)
        cosine_similarity_results.append(cosine_similarity.result().numpy())

        logcosh = tf.keras.metrics.LogCoshError()
        logcosh.update_state(scaled_actual, scaled_predictions)
        logcosh_results.append(logcosh.result().numpy())

    avg_mse = sum(mse_results) / (len(mse_results) * 1)
    avg_rmse = sum(rmse_results) / (len(rmse_results) * 1)
    avg_mae = sum(mae_results) / (len(mae_results) * 1)
    avg_mape = sum(mape_results) / (len(mape_results) * 1)
    avg_msle = sum(msle_results) / (len(msle_results) * 1)
    avg_cosine_similarity = sum(cosine_similarity_results) / (
        len(cosine_similarity_results) * 1
    )
    avg_logcosh = sum(logcosh_results) / (len(logcosh_results) * 1)
    return [
        avg_mse,
        avg_rmse,
        avg_mae,
        avg_mape,
        avg_msle,
        avg_cosine_similarity,
        avg_logcosh,
    ]

def build_prophet(
    training_df_data,
    testing_df_data,
    metrics,
    target_fitted_scaler,
):

    actual = []
    predictions = []
    for filtered_train_data, filtered_test_data in zip(training_df_data, testing_df_data):
        ts = TimeSeries.from_dataframe(filtered_train_data, time_col='ds', value_cols='y')
        unique_id = filtered_train_data['unique_id'].values[0]

        model = Prophet()
        model.fit(ts)
        model.save(f"{public}/models/prophet_" + unique_id + ".pkl")

        filtered_forecast = model.predict_raw(len(filtered_test_data))
        filtered_predictions = filtered_forecast['yhat'].values
        filtered_actual = filtered_test_data[target].values
        
        actual.extend(filtered_actual)
        predictions.extend(filtered_predictions)
    
    actual = np.array(actual)
    predictions = np.array(predictions)
    
    metric_results = []
    scaled_actual = target_fitted_scaler.transform(actual.reshape(-1, 1)).flatten()
    scaled_predictions = target_fitted_scaler.transform(predictions.reshape(-1, 1)).flatten()
    mse = tf.keras.metrics.MeanSquaredError()
    mse.update_state(scaled_actual, scaled_predictions)
    metric_results.append(mse.result().numpy())
    
    rmse = tf.keras.metrics.RootMeanSquaredError()
    rmse.update_state(scaled_actual, scaled_predictions)
    metric_results.append(rmse.result().numpy())
    
    mae = tf.keras.metrics.MeanAbsoluteError()
    mae.update_state(scaled_actual, scaled_predictions)
    metric_results.append(mae.result().numpy())
    
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    mape.update_state(scaled_actual, scaled_predictions)
    metric_results.append(mape.result().numpy())
    
    msle = tf.keras.metrics.MeanSquaredLogarithmicError()
    msle.update_state(scaled_actual, scaled_predictions)
    metric_results.append(msle.result().numpy())
    
    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    cosine_similarity.update_state(scaled_actual, scaled_predictions)
    metric_results.append(cosine_similarity.result().numpy())
    
    logcosh = tf.keras.metrics.LogCoshError()
    logcosh.update_state(scaled_actual, scaled_predictions)
    metric_results.append(logcosh.result().numpy())
    
    return metric_results

def forecast_dl(
    nn_sequence_length,
    num_features,
    X_train,
    y_train,
    target_fitted_scaler,
    X_test,
    model,
):
    """
    Forecasts using deep learning model.

    Parameters:
        sequence_length (int): Length of each sequence.
        num_features (int): Number of features.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        target_fitted_scaler (MinMaxScaler): Target fitted scaler.
        X_test (np.ndarray): Testing input data.
        model (str): Model type ('gru' or 'lstm').

    Returns:
        np.ndarray: Forecasted values.
    """
    global epochs
    dl_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f"{public}/models/{model}"
        + "_".join(
            [f"{key}{''.join(map(str, value))}" for key, value in plot_values.items()]
        )
        + ".keras",
        monitor="val_mape",
        save_best_only=True,
        mode="min",
        verbose=0,
    )

    dl_model = Sequential()
    if model == "gru":
        dl_model.add(Input((nn_sequence_length - 1, num_features)))
        dl_model.add(GRU(units=128, return_sequences=True))
        dl_model.add(Dropout(0.2))
        dl_model.add(GRU(units=64, return_sequences=False))
    elif model == "lstm":
        dl_model.add(Input((nn_sequence_length - 1, num_features)))
        dl_model.add(LSTM(units=128, return_sequences=True))
        dl_model.add(Dropout(0.2))
        dl_model.add(LSTM(units=64, return_sequences=False))

    dl_model.add(Dense(units=8))
    dl_model.add(Dense(units=1, activation="sigmoid"))

    dl_model.compile(
        optimizer=Adam(0.01),
        loss="mean_squared_error",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
            tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
            tf.keras.metrics.CosineSimilarity(name="cosine_similarity"),
            tf.keras.metrics.LogCoshError(name="logcosh"),
        ],
    )
    # NOTE get Epochs Baack to 100
    history = dl_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        verbose=0,
        validation_split=0.1,
        callbacks=[dl_model_checkpoint],
    )

    # predictions = dl_model.predict(X_test,verbose=0)
    # scaled_predictions = target_fitted_scaler.inverse_transform(predictions)
    # return scaled_predictions


def forecast_arima(
    arima_sequence_length, num_features, training_df_data, testing_df_data
):
    """
    Forecasts using ARIMA model.

    Parameters:
        sequence_length (int): Length of each sequence.
        num_features (int): Number of features.
        training_df_data (list): List of training DataFrame data.
        testing_df_data (list): List of testing DataFrame data.

    Returns:
        np.ndarray: Forecasted values.
    """
    seasonality = arima_sequence_length

    duration_dict = {"D": "D", "M": "ME", "Y": "Y", "Q": "3ME", "H": "6ME"}
    duration = duration_dict[granularity]
    sf = StatsForecast(models=[AutoARIMA(season_length=seasonality)], freq=duration)

    # for df in training_df_data:
    #     df['ds'] = pd.to_datetime(df[aggregate_date[0]])
    #     df['unique_id'] = df[agg_features[0]]
    #     df['y'] = df[target]

    concatenated_df = pd.concat(training_df_data, ignore_index=True)
    concatenated_df["unique_id"] = concatenated_df[agg_features].apply(
        lambda row: " ".join(map(str, row)), axis=1
    )
    concatenated_df["ds"] = pd.to_datetime(concatenated_df[aggregate_date[0]])
    concatenated_df["y"] = concatenated_df[target]
    sf.fit(concatenated_df[["ds", "unique_id", "y"]])

    # concatenated_df = pd.concat(training_df_data, ignore_index=True)
    # sf.fit(concatenated_df[['ds', 'unique_id', 'y']])

    with open(
        f"{public}/models/arima"
        + "_".join(
            [f"{key}{''.join(map(str, value))}" for key, value in plot_values.items()]
        )
        + ".pickle",
        "wb",
    ) as file:
        pickle.dump(sf, file)

def forecast_prophet(training_df_data, testing_df_data):
        
    for filtered_train_data in training_df_data:
        filtered_train_data['unique_id']  = filtered_train_data[agg_features].apply(lambda row: " ".join(map(str, row)), axis=1)
        unique_id = filtered_train_data['unique_id'].values[0]
        print(filtered_train_data)
        ts = TimeSeries.from_dataframe(filtered_train_data, time_col=aggregate_date[0], value_cols=target)
        model = Prophet()
        model.fit(ts)
        model.save(f"{public}/models/prophet_" + unique_id + ".pkl")    

def forecast(
    current_sequence,
    num_steps_forward,
    actual_sales_list,
    test_df,
    train_df,
    model,
    feature_scaler,
    target_fitted_scaler,
    num_features,
):
    """
    Forecasts future values.

    Parameters:
        current_sequence (np.ndarray): Current sequence.
        num_steps_forward (int): Number of steps forward to forecast.
        actual_sales_list (list): List of actual sales.
        test_df (pd.DataFrame): Test DataFrame.
        train_df (pd.DataFrame): Train DataFrame.
        model (str): Model type ('gru', 'lstm', 'arima').
        feature_scaler (MinMaxScaler): Feature scaler.
        target_fitted_scaler (MinMaxScaler): Target fitted scaler.
        num_features (int): Number of features.

    Returns:
        list: Forecasted values.
    """
    prev_date = train_df[-1:][aggregate_date[0]].values[0]
    dates = []
    predicted_target = []
    test = []
    values = feature_scaler.inverse_transform(
        current_sequence[-1].reshape(-1, num_features)
    )
    values = np.round(values)

    data = pd.DataFrame(
        {**{f: values[0, -len(agg_features) - 1] for i, f in enumerate(agg_features)}},
        index=[0],
    )
    if model == "arima":
        with open(
            f"{public}/models/arima"
            + "_".join(
                [
                    f"{key}{''.join(map(str, value))}"
                    for key, value in plot_values.items()
                ]
            )
            + ".pickle",
            "rb",
        ) as file:
            sf = pickle.load(file)
        train_df["ds"] = pd.to_datetime(train_df[aggregate_date[0]])
        train_df["unique_id"] = train_df[agg_features].apply(
            lambda row: " ".join(map(str, row)), axis=1
        )
        train_df["y"] = train_df[target]
        duration_dict = {"D": "D", "M": "M", "Y": "Y", "Q": "3M", "H": "6M"}
        duration = duration_dict[granularity]
        predicted_target = (
            sf.forecast(
                df=train_df[["ds", "unique_id", "y"]], h=len(test_df), level=[95]
            )["AutoARIMA"]
            .values.flatten()
            .tolist()
        )

    elif model == "Prophet":
        test_df["unique_id"] = test_df[agg_features].apply(lambda row: " ".join(map(str, row)), axis=1)

        testing_df_data = [test_df[test_df["unique_id"] == uid] for uid in test_df["unique_id"].unique()]
        
        predicted_target = []
        for filtered_test_data in testing_df_data:
            unique_id = filtered_test_data['unique_id'].values[0]
            model = Prophet.load(f"{public}/models/prophet_" + unique_id + ".pkl")
            filtered_forecast = model.predict_raw(len(filtered_test_data))
            filtered_predictions = filtered_forecast['yhat'].values
            predicted_target.extend(filtered_predictions)

    else:
        if model == "lstm":
            ml_model = load_model(
                f"{public}/models/lstm"
                + "_".join(
                    [
                        f"{key}{''.join(map(str, value))}"
                        for key, value in plot_values.items()
                    ]
                )
                + ".keras"
            )
        elif model == "gru":
            ml_model = load_model(
                f"{public}/models/gru"
                + "_".join(
                    [
                        f"{key}{''.join(map(str, value))}"
                        for key, value in plot_values.items()
                    ]
                )
                + ".keras"
            )
        for i in range(num_steps_forward):
            test.append(current_sequence)

            next_prediction = ml_model.predict(
                current_sequence.reshape(1, -1, num_features), verbose=0
            )
            values = feature_scaler.inverse_transform(
                current_sequence[-1].reshape(-1, num_features)
            )
            values = np.round(values)
            next_prediction = target_fitted_scaler.inverse_transform(next_prediction)

            if granularity == "D":
                data[aggregate_date[0]] = prev_date + pd.DateOffset(days=1)
                prev_date = pd.to_datetime(data[aggregate_date[0]])
                dates.append(prev_date)
                data["M"] = prev_date.dt.month.values[0]
                data["day_of_week"] = prev_date.dt.dayofweek.values[0]
                data["day"] = prev_date.dt.day.values[0]
                data["Y"] = prev_date.dt.year.values[0]

                predicted_target.append(float(next_prediction[0, 0]))
                next_timestep = np.array(
                    [
                        data["Y"].values[0],
                        data["M"].values[0],
                        data["day_of_week"].values[0],
                        data["day"].values[0],
                        *data[agg_features].values[0],
                        float(next_prediction[0, 0]),
                    ]
                ).reshape(1, -1)

            elif granularity == "M":
                data[aggregate_date[0]] = prev_date + pd.DateOffset(months=1)
                prev_date = pd.to_datetime(data[aggregate_date[0]])
                dates.append(prev_date)
                data["M"] = prev_date.dt.month.values[0]
                data["Y"] = prev_date.dt.year.values[0]
                predicted_target.append(float(next_prediction[0, 0]))
                next_timestep = np.array(
                    [
                        data["Y"].values[0],
                        data["M"].values[0],
                        *data[agg_features].values[0],
                        float(next_prediction[0, 0]),
                    ]
                ).reshape(1, -1)

            elif granularity == "Y":
                data[aggregate_date[0]] = prev_date + pd.DateOffset(years=1)
                prev_date = pd.to_datetime(data[aggregate_date[0]])
                dates.append(prev_date)
                data["Y"] = prev_date.dt.year.values[0]
                predicted_target.append(float(next_prediction[0, 0]))
                next_timestep = np.array(
                    [
                        data["Y"].values[0],
                        *data[agg_features].values[0],
                        float(next_prediction[0, 0]),
                    ]
                ).reshape(1, -1)

            elif granularity == "Q":
                data[aggregate_date[0]] = prev_date + pd.DateOffset(months=3)
                prev_date = pd.to_datetime(data[aggregate_date[0]])
                dates.append(prev_date)
                data["Q"] = prev_date.dt.month.values[0]
                data["Y"] = prev_date.dt.year.values[0]
                predicted_target.append(float(next_prediction[0, 0]))
                next_timestep = np.array(
                    [
                        data["Y"].values[0],
                        data["Q"].values[0],
                        *data[agg_features].values[0],
                        float(next_prediction[0, 0]),
                    ]
                ).reshape(1, -1)

            elif granularity == "H":
                data[aggregate_date[0]] = prev_date + pd.DateOffset(months=6)
                prev_date = pd.to_datetime(data[aggregate_date[0]])
                dates.append(prev_date)
                data["H"] = prev_date.dt.month.values[0]
                data["Y"] = prev_date.dt.year.values[0]
                predicted_target.append(float(next_prediction[0, 0]))
                next_timestep = np.array(
                    [
                        data["Y"].values[0],
                        data["H"].values[0],
                        *data[agg_features].values[0],
                        float(next_prediction[0, 0]),
                    ]
                ).reshape(1, -1)
            next_timestep_1 = feature_scaler.transform(next_timestep).reshape(1, 1, -1)
            current_sequence = np.append(
                current_sequence.reshape(1, -1, num_features)[:, 1:, :],
                next_timestep_1,
                axis=1,
            )

    actual_sales_values = actual_sales_list[:num_steps_forward].reshape(-1, 1)

    return predicted_target


def main():
    global epochs
    epochs = 2
    # no 1
    # df_agg_gran, training_df_data, testing_df_data = eda_ts(pd.read_csv("train.csv"), 'M', 'M', [
    #                                                         'store', 'item'], [], [aggregate_date[0]], 'sales', False, {'store': [1, 2, 3], 'item': [1, 2]})
    df_agg_gran, training_df_data, testing_df_data = eda_ts(
        pd.read_csv("SalesTraining.csv"),
        "M",
        "M",
        ["Store Cd", "Item Cd"],
        [],
        ["Inv Date"],
        "Inv Amt",
        False,
        {
            "Store Cd": ["Store -1", "Store -2"],
            "Item Cd": ["Item - 1", "Item - 2", "Item - 3"],
        },
    )
    # no 2
    eval_df = build_model_ts(
        copy.deepcopy(forecast_training_data),
        copy.deepcopy(forecast_testing_data),
        copy.deepcopy(training_df_data),
        copy.deepcopy(testing_df_data),
        copy.deepcopy(training_data),
    )
    eval_df.to_csv("eval_df.csv")

    # no 3
    plot_results(
        models, feature_scaler, target_fitted_scaler, num_features, plot_values
    )

    model = eval_df["Model Name"][0].lower()
    train_df = pd.read_csv("SalesTraining.csv")
    test_df = pd.read_csv("new.csv")
    test_df[target] = 0

    # no 4
    pred_df = predict_values_ts(train_df, test_df, model)

    pred_df.to_csv("predicted_sales.csv", index=False)

    print(pd.read_csv("predicted_sales.csv"))
