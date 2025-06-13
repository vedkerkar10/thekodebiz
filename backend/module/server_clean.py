import io
from flask import Flask, request, send_file, Response
from flask import json as JSON
from flask_cors import CORS
import inspect
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
import xlsxwriter
import tempfile
import zipfile

from keras.src.ops import Correlate
from scipy.signal import correlate
from scipy import stats
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    TunedThresholdClassifierCV,
    FixedThresholdClassifier,
    StratifiedKFold,
)
from sklearn.feature_selection import RFE, RFECV, mutual_info_classif
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SMOTENC,
    RandomOverSampler,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)
from sklearn.preprocessing import (
    LabelEncoder,
    PolynomialFeatures,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    LogisticRegressionCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from prince import MCA

from lightgbm import LGBMClassifier

import module.CommonObjects as co
from time import time, strftime
from datetime import datetime
from module.time_series import timeSeries
from module.test import test

app = Flask(__name__)
app.register_blueprint(timeSeries, url_prefix="/time_series")
app.register_blueprint(test, url_prefix="/test")

cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config["SECRET_KEY"] = "32fe2d5e-c37e-49ca-a101-ba3cdf2280dc"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# algos = {}

with open("algos.json", "r") as f:
    # global algos
    algos = json.load(f)

GlobalVariables = {}
training_data = pd.DataFrame()
training_clean_data = pd.DataFrame()
training_data_num = pd.DataFrame()
training_clean_data_copy = pd.DataFrame()
training_data_num_copy = pd.DataFrame()
training_encoders = {}
training_features_data = pd.DataFrame()
training_analysis = pd.DataFrame()
training_data_analysis = ""
training_file_path = Path.cwd()
correlation_target = pd.DataFrame()
correlation_all = pd.DataFrame()
predict_data = pd.DataFrame()
predict_clean_data = pd.DataFrame()
predict_data_num = pd.DataFrame()
predict_clean_data_copy = pd.DataFrame()
predict_data_num_copy = pd.DataFrame()
predict_data_analysis = ""
predict_encoders = {}
predicted_data_final = pd.DataFrame()
date_columns = []
numeric_columns = []
categorical_columns = []
best_model_index = -1
probability_index = -1
pct_complete = 0.0
best_parameters = pd.DataFrame()
g_confusion_matrix = pd.DataFrame()
g_classification_report = pd.DataFrame()
scaler = {}
outlier_columns = {}
sample_str = ""
feature_selection_algos = []
perform_pca = False
pca_features = []
pca_algo = 'PCA(n_components=3, random_state=42)'
perform_mca = False
mca_features = []
mca_algo = 'MCA(n_components=3, random_state=42)'

sse_clients = []


@app.route("/events")
def events():
    def stream():
        while True:
            if sse_clients:
                client = sse_clients.pop(0)
                yield f"data: {json.dumps(client)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


def alertFrontend(message: str, type: str = "DIALOG"):
    if type in ["DIALOG", "TOAST"]:
        sse_clients.append({"message": message, "type": type})
        print("Alerted Frontend")
    else:
        raise ValueError("Invalid type. Valid types are 'DIALOG' and 'TOAST'.")


def toggleRefetchOutlierAnalysisData():
    sse_clients.append({"message": 'Refetch Data Analysis Data', "type": 'RefetchOutlierAnalysisData'})
    print("Alerted Frontend")


@app.route("/reload_algos", methods=["GET"])
def loadAlgos():
    global algos
    try:
        with open("algos.json", "r") as f:
            algos = json.load(f)
            print("Reloaded Algos")
            print(algos)
            return {
                "status": 200,
                "message": "Algos.json Reloaded",
                "Algos.json": algos,
            }
    except Exception as e:
        return {"status": 500, "message": str(e)}


@app.route("/GetData", methods=["POST"])
def get_file_data():  # Call this function on upload of first excel
    """
    Get data from excel and store in DataFrame.
    Clean data convert cleaned data to numerical values and store in DataFrame
    :param file_str: The filename including file path
    :param file_type: Specify whether file is for training or Prediction. Valid values "Train" or "Predict"
    :return: None

    """
    global \
        training_data, \
        training_clean_data, \
        training_data_num, \
        training_clean_data_copy, \
        training_data_num_copy, \
        training_encoders, \
        training_data_analysis, \
        training_file_path, \
        predict_data, \
        predict_clean_data, \
        predict_data_num, \
        predict_clean_data_copy, \
        predict_data_num_copy, \
        predict_encoders, \
        predict_data_analysis, \
        date_columns, \
        numeric_columns, \
        categorical_columns

    print(strftime("%Y-%m-%d %H:%M:%S"))
    # Reading File From POST Request
    file = request.files["data"]
    file_name = file.filename
    file_type = request.form["file_type"]
    target_features = []
    file_path = file.filename  # Properly get the filename from the Flask file object
    if not file_name:
        raise ValueError("No filename provided in the uploaded file.")
    file_ext = file_name.split(".")[-1]
    # alertFrontend(
    #     "Reading Data from File "
    #     + str(file_name)
    #     + " "
    #     + str(strftime("%Y-%m-%d %H:%M:%S")),
    #     "TOAST",
    # )
    if file_type == "Train":
        print(f"Reading data from Training file : {file_name}")
        training_file_path = file_path
        training_data = co.read_data(file)
        (
            training_clean_data,
            training_data_num,
            training_encoders,
            date_columns,
            data_analysis,
            numeric_columns,
        ) = co.eda(training_data)
        categorical_columns = list(set(training_clean_data.columns.tolist()) - set(numeric_columns) - set(date_columns))
        categorical_columns.sort()
        training_clean_data_copy = training_clean_data.copy()
        training_data_num_copy = training_data_num.copy()
        alertFrontend(
            "Data Analysis " + str(data_analysis),
            "DIALOG",
        )
        target_features = training_data_num.select_dtypes(
            exclude=["datetime64[ns]"]
        ).columns.to_list()
        target_features.sort()
        training_data_analysis = data_analysis
        df_20 = training_clean_data.head(20)
        # training_clean_data.reset_index(inplace=True)
    else:
        print(f"Reading data from Prediction file: {file_path}")
        predict_data = co.read_data(file)
        (
            predict_clean_data,
            predict_data_num,
            predict_encoders,
            date_columns,
            data_analysis,
            numeric_columns,
        ) = co.eda(
            predict_data,
        )
        predict_clean_data_copy = predict_clean_data.copy()
        predict_data_num_copy = predict_data_num.copy()
        predict_data_analysis = data_analysis
        df_20 = predict_clean_data.head(20)
    print(strftime("%Y-%m-%d %H:%M:%S"))
    print(df_20.to_string())
    print("data_analysis")
    print(data_analysis)

    return {
        "status": 200,
        "target_features": target_features,
        "date_columns": date_columns,
        "data_analysis": data_analysis,
        "display_data": df_20.to_json(orient="table", index=False),
    }


@app.route("/SetOptions", methods=["POST"])
def get_corr():  # Call this function after selecting Target and Features
    """
    Find the correlation of features with target
    :param target: target column name as string
    :param feature_list: features column names as a List
    :return: json object with correlation of Features with Target column
    """

    global training_data_num, correlation_target, correlation_all
    target = request.form["target"]
    feature_list = json.loads(request.form["features"])
    problem_type = json.loads(request.form["algorithm"])

    # if (
    #         "remove_outliers" in algos[problem_type]
    #         and algos[problem_type]["remove_outliers"]
    # ):
    #     remove_outliers(problem_type)
    #
    # if (
    #         'scaling' in algos[problem_type]
    #         and algos[problem_type]['scaling']
    # ):
    #     data_scaling(problem_type, training_data_num)

    print(f"Correlation started at: {strftime('%Y-%m-%d %H:%M:%S')}")
    data = training_data_num.copy()
    selected_cols = feature_list + [target]
    data = data[selected_cols]
    c_all = data.corr()
    c_all.index.name = "Features"
    c = c_all.copy()
    c = c[[target]]
    c["Corelation (Abs. Value)"] = abs(c[target])
    c = c.sort_values(by="Corelation (Abs. Value)", ascending=False)
    # c = c.sort_values(by=target, ascending=True)
    col_name = "Correlation with " + target + ""
    c.rename(columns={target: col_name}, inplace=True)
    c = c.drop(target)
    c = c.reset_index().rename(columns={"index": "Features"})
    correlation_all = c_all
    # correlation_target = c
    print(f"Correlation completed at: {strftime('%Y-%m-%d %H:%M:%S')}")

    if problem_type == 'Classification':
        print(f"Mutual Information started at: {strftime('%Y-%m-%d %H:%M:%S')}")
        X = data[feature_list]
        y = data[target]
        mi_scores = mutual_info_classif(X, y, discrete_features="auto")
        mi_scores_df = pd.DataFrame(
            {"Features": X.columns, "MI Score": mi_scores}
        ).sort_values(by="MI Score", ascending=False)
        mi_scores_df.reset_index(drop=True, inplace=True)
        print("Mutual Information Scores are as below:")
        print(
            "____________________________________________________________________________________________________"
            "____________"
        )
        print("")
        print(mi_scores_df)
        print(
            "____________________________________________________________________________________________________"
            "____________"
        )
        print("")
        print(f"Mutual Information completed at: {strftime('%Y-%m-%d %H:%M:%S')}")
        c = c.merge(right=mi_scores_df, on="Features")
    correlation_target = c
    # if (feature_selection_algos or
    #         ("feature_selection" in algos[problem_type]
    #          and algos[problem_type]["feature_selection"]
    #         )):
    #     if sample_str or ("sampling" in algos[problem_type] and algos[problem_type]["sampling"]):
    #         X, y = sampling(problem_type, target, X, y)
    #     feature_selection(problem_type=problem_type, feature_list=feature_list, target=target)
    return {
        "c1": c.to_json(orient="table", index=False),
        "c2": c_all.to_json(orient="table", index=True),
    }


# Call this function on click of Train button


@app.route("/Train", methods=["POST"])
def build_models():
    """
    Evaluation of algorithms to identify the most optimal predictive model and rank these models
    :param problem_type: Algorithm Category
    :param target: target column name as string
    :param feature_list: features column names as a List
    :return: model_results - Evaluated models with their rank and accuracy
            df_predicted - predicted values - this is no longer required
            best_pos - position of the best algorithm
            prob_pos - position of algorithm that has probability prediction
            err - Boolean - True if there is error, False if there is no error
            e - Error message
            build_errors - Error details during model building - in json table format
            prob_errors - Error details during probability phase - in json table format
    """

    target = GlobalVariables["target"]
    feature_list = GlobalVariables["feature_list"]
    problem_type = GlobalVariables["algorithm"]

    print(f"Target selected is: {target}")
    print(f"Features selected are: {feature_list}")
    print(f"Algorithm selected is: {problem_type}")
    global \
        training_clean_data, \
        training_data_num, \
        training_encoders, \
        training_analysis, \
        best_model_index, \
        probability_index, \
        pct_complete, \
        algos, \
        sample_str

    print(
        f'Training of algorithms for "{problem_type}" started at: {strftime("%Y-%m-%d %H:%M:%S")}'
    )
    data = training_data_num.copy()

    X = data.loc[:, feature_list]
    y = data.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if sample_str or (
            "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
    ):
        X_train, y_train = sampling(problem_type, target, X_train, y_train)

    model_results = {}  # stores the results of all the models
    build_errors = {}  # stores the error messages by model during build phase

    results_columns = algos[problem_type]["col_names"]
    if 'tune_threshold' in algos[problem_type] and algos[problem_type]['tune_threshold']:
        results_columns += ['Best Threshold']
    sort_col_num = algos[problem_type]["sort_by"]
    sort_by = algos[problem_type]["col_names"][sort_col_num]

    df_predicted = "Blank"

    num_of_models = len(algos[problem_type]["names"])
    counter = -1

    class_labels = None
    if "positive_target" in algos[problem_type]:
        class_labels = get_class_labels(problem_type=problem_type, target=target)
    #     class_labels = algos[problem_type]['positive_target']
    #     if target in training_encoders.keys():
    #         class_labels = training_encoders[target].transform(class_labels).tolist()
    #
    # print(f'Class Labels are: {class_labels}')
    if perform_pca or perform_mca:
        X_train = perform_pca_mca(feature_list, X_train.copy())
        X_test = perform_pca_mca(feature_list, X_test.copy())

    # for model in algos[problem_type]["names"]:
    for model_str in algos[problem_type]["names"]:
        model = eval(model_str)
        counter += 1
        pct_complete = counter / num_of_models * 100
        # socketio.emit("progress",pct_complete)
        print(f"{pct_complete: .2f}% training completed")
        # model_name = str(model).split("(")[0]
        # model_name = model_str.split("(")[0]
        model_name = model_str
        mod_idx = algos[problem_type]["names"].index(model_str)
        err_details = []
        start_time = time()
        print(f"Training Algorithm {counter + 1} of {num_of_models}")
        print(
            f"Started training using {model_name} at {strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if 'tune_threshold' in algos[problem_type] and algos[problem_type]['tune_threshold']:
                print(f"Started tuning threshold using {model_name} at {strftime('%Y-%m-%d %H:%M:%S')}")
                tuned_model = TunedThresholdClassifierCV(
                    estimator=model.fit(X_train, y_train),
                    scoring=algos[problem_type]['tune_score'],
                    random_state=0
                )  # scoring=make_scorer(eval(algos[problem_type]['scores'][sort_col_num-1])),
                tuned_model.fit(X_train, y_train)
                y_pred = tuned_model.predict(X_test)
                best_threshold = tuned_model.best_threshold_
                print(f"Completed tuning threshold using {model_name} at {strftime('%Y-%m-%d %H:%M:%S')}")
                print(f'Best Threshold is {best_threshold}')
        except Exception as e:
            err_details.append(type(e).__name__)
            err_details.append(str(e))
            build_errors[model_name] = err_details
            continue
        score = [0]
        # for scores in algos[problem_type]["scores"]:
        for scores_str in algos[problem_type]["scores"]:
            scores = eval(scores_str)
            params = inspect.signature(scores)
            if "average" in params.parameters:
                if scores.__name__ in ["precision_score", "recall_score", "f1_score"]:
                    score.append(
                        # scores(y_test, y_pred, average="macro", zero_division=np.nan)
                        scores(
                            y_test,
                            y_pred,
                            average="micro",
                            zero_division=np.nan,
                            labels=class_labels,
                        )
                    )
                else:
                    score.append(scores(y_test, y_pred, average="micro"))
            else:
                if scores.__name__ in ["precision_score", "recall_score", "f1_score"]:
                    score.append(scores(y_test, y_pred, zero_division=np.nan))
                else:
                    score.append(scores(y_test, y_pred))

        # score.append(mod_idx)

        if algos[problem_type]["probability"]:
            try:
                prob = model.predict_proba(X_test)
                prob_flag = True
                score.append(prob_flag)
            except Exception as excp:
                prob_flag = False
                score.append(prob_flag)
        else:
            prob_flag = False
            score.append(prob_flag)

        finish_time = time()
        time_taken = finish_time - start_time
        score.append(time_taken)
        score.append(mod_idx)
        if 'tune_threshold' in algos[problem_type] and algos[problem_type]['tune_threshold']:
            print(f'Score before best threshold is {score}')
            score.append(best_threshold)
            print(f'Score after best threshold is {score}')
        model_results[model_name] = score
        print(f"{model_name} training completed in {time_taken: .2f} seconds")

    model_results = pd.DataFrame.from_dict(model_results).T
    print(model_results)
    model_results.columns = results_columns
    model_results.index.name = "Algorithm Name"
    model_results = model_results.sort_values(sort_by, ascending=False)
    model_results.iloc[:, 0] = np.arange(1, len(model_results) + 1)
    # model_results.iloc[:, 0] = model_results.iloc[:, 0].astype('int64')
    model_results = model_results.infer_objects()
    best_pos = int(model_results["Position"].iloc[0])
    build_errors = pd.DataFrame.from_dict(build_errors).T
    if not build_errors.empty:
        build_errors.columns = ["Error Type", "Error Details"]
        build_errors.index.name = "Algorithm Name"
        print(f"Build Errors are as below: {build_errors}")

    err = False
    e = "None"
    prob_pos = -1
    prob_target = None

    prob_errors = {}  # stores the error messages by model during finding probability phase

    if algos[problem_type]["probability"]:
        err_details = []
        model_with_prob = model_results[model_results["Probability Flag"]]
        if model_with_prob.empty:
            err = True
            e = "No model has probability values"
            err_details.append("All")
            err_details.append(e)
            prob_errors["All"] = err_details
            prob_errors = pd.DataFrame.from_dict(prob_errors).T
            prob_errors.columns = ["Error Type", "Error Details"]
            prob_errors.index.name = "Algorithm Name"
            print(strftime("%Y-%m-%d %H:%M:%S"))
            return (
                model_results,
                df_predicted,
                best_pos,
                prob_pos,
                err,
                e,
                build_errors,
                prob_errors,
            )
        prob_pos = model_with_prob.Position.iloc[0]

    if isinstance(build_errors, pd.DataFrame):
        build_errors = pd.DataFrame.to_dict(build_errors)
    if isinstance(prob_errors, pd.DataFrame):
        prob_errors = pd.DataFrame.to_dict(prob_errors)
    if algos[problem_type]["probability"]:
        cols_to_skip = 1
    else:
        cols_to_skip = 1

    best_model_index = best_pos
    probability_index = prob_pos

    cols_to_display = algos[problem_type]["col_display"]
    if 'tune_threshold' in algos[problem_type] and algos[problem_type]['tune_threshold']:
        cols_to_display += ['Best Threshold']
    pct_complete = 100.00
    print(f"{pct_complete: .2f}% training completed")
    print(
        f'Training of algorithms for "{problem_type}" completed at: {strftime("%Y-%m-%d %H:%M:%S")}'
    )
    print(f'Analysis of Training for "{problem_type}" is as below:')
    print(model_results.to_string())
    training_analysis = model_results.reset_index()

    return json.dumps(
        {
            "model_result": model_results.to_json(orient="table", index=True),
            "df_predicted": df_predicted,
            "best_pos": best_pos,
            "prob_pos": prob_pos,
            "err": err,
            "e": e,
            "build_errors": build_errors,
            "prob_errors": prob_errors,
            "cols_to_display": cols_to_display,
        },
        cls=NpEncoder,
    )


@app.route("/SetValue", methods=["POST"])
def SetValues():
    global GlobalVariables
    attribute = request.args.get("attr")
    value = json.loads(request.form["value"])
    GlobalVariables[attribute] = value

    print(list([attribute, value]))
    return "Done"


# Call this function on click of Predict button
@app.route("/Predict", methods=["POST"])
def predict_values():
    """
    Predict the value of Target using the best model
    :param problem_type: Algorithm Category
    :param target: target column name as string
    :param feature_list: features column names as a List
    : param model_position: position of selected model
    : param model_probability_flag: Boolean value indicating if an algo has probability feature or not
    :return: final_predict - predicted Target values for data 'd' along with probability, if applicable
             predict_errors - Error details during model building - in json table format
    """
    global \
        training_data, \
        training_clean_data, \
        training_data_num, \
        training_encoders, \
        training_analysis, \
        predict_data, \
        predict_clean_data, \
        predict_data_num, \
        predicted_data_final, \
        best_model_index, \
        probability_index, \
        algos, \
        scalar

    print(f"Prediction started at {strftime('%Y-%m-%d %H:%M:%S')}")
    problem_type = request.form["algorithm"]
    target = request.form["target"]
    feature_list = json.loads(request.form["features"])
    model_position = int(request.form["model_position"])  # json.loads(request.form[''])
    model_probability_flag = (
        True if request.form["model_probability_flag"] == "true" else False
    )  # json.loads(request.form[''])

    print(model_position)
    print(model_probability_flag)

    if model_position is None:
        best_pos = 0
        # prob_pos = probability_index
    else:
        best_pos = model_position

    if model_probability_flag is None:
        if hasattr(algos[problem_type]["names"][best_pos], "predict_proba"):
            prob_pos = best_pos
        else:
            prob_pos = -1
    elif model_probability_flag:
        prob_pos = best_pos
    else:
        prob_pos = -1

    print(f"Target selected is: {target}")
    print(f"Features selected are: {feature_list}")
    print(
        f"Algorithm selected for prediction is: {algos[problem_type]['names'][best_pos]}"
    )

    if scaler:
        data_scaling(problem_type, predict_data_num, scaler)
    pdf = predict_clean_data.copy()
    pdf_n = predict_data_num.copy()
    df_train = training_data_num.copy()

    num_of_periods = 1
    split_rows = False

    if "Periods To Predict" in pdf_n.columns:
        split_rows = True
        num_of_periods = pdf_n["Periods To Predict"].max()
    X_train = df_train.loc[:, feature_list]
    if perform_pca or perform_mca:
        X_train = perform_pca_mca(feature_list, X_train.copy())
    else:
        X_train = X_train.to_numpy()
    y_train = df_train.loc[:, target]
    final_predict = pd.DataFrame()
    predict_errors = {}  # stores the error messages for best model during predict phase
    err_details = []
    prob_target = None
    predicted_target = "Predicted " + str(target)
    # best_model = eval(algos[problem_type]["names"][best_pos])
    # model_name = str(best_model).split("(")[0]
    best_threshold: float = -1.0
    for i in range(num_of_periods):
        best_model = eval(algos[problem_type]["names"][best_pos])
        model_name = str(best_model).split("(")[0]
        predicted_df = pd.DataFrame()
        if split_rows:
            pdfi_n = pdf_n.loc[pdf_n["Periods To Predict"] == (i + 1)]
            X = pdfi_n.loc[:, feature_list]
        else:
            X = pdf_n.loc[:, feature_list]
        if perform_pca or perform_mca:
            X = perform_pca_mca(feature_list, X.copy())
        else:
            X = X.to_numpy()
        print(
            f"Training started for period {i + 1} of {num_of_periods} at {strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if sample_str or (
                "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
        ):
            X_train, y_train = sampling(problem_type, target, X_train, y_train)
        print(
            f"Number of rows in training and prediction are: {X_train.shape[0]} and {y_train.shape[0]}"
        )
        if 'tune_threshold' in algos[problem_type] and  algos[problem_type]['tune_threshold']:
            print(f'Best Threshold default value is: {best_threshold}')
            if not training_analysis.empty:
                # print(training_analysis.to_string())
                # print(f'Model position is {best_pos}')
                best_threshold = training_analysis[training_analysis['Position'] == best_pos]['Best Threshold'].values[
                    0]
                print(f'Best Threshold from trained model is: {best_threshold}')
            if best_threshold == -1.0:
                best_model = TunedThresholdClassifierCV(
                    estimator=best_model.fit(X_train, y_train),
                    scoring=algos[problem_type]['tune_score'],
                    random_state=0
                ).fit(X_train, y_train)
                best_threshold = best_model.best_threshold_
                print(f'Best Threshold from training model is: {best_threshold}')
            else:
                best_model = FixedThresholdClassifier(
                    estimator=best_model.fit(X_train, y_train),
                    threshold=best_threshold
                ).fit(X_train, y_train)
        else:
            best_model.fit(X_train, y_train)
        # best_model.fit(X_train, y_train)
        try:
            print(
                f"Prediction started for period {i + 1} of {num_of_periods} at {strftime('%Y-%m-%d %H:%M:%S')}"
            )
            y_predicted = best_model.predict(X)
        except Exception as e:
            err_details.append(type(e).__name__)
            err_details.append(str(e))
            predict_errors[model_name] = err_details
            predict_errors = pd.DataFrame.from_dict(predict_errors).T
            if not predict_errors.empty:
                predict_errors.columns = ["Error Type", "Error Details"]
                predict_errors.index.name = "Algorithm Name"
            print(f"Error in prediction: {predict_errors}")
            break

        if split_rows:
            predicted_df = pdf.loc[pdf["Periods To Predict"] == (i + 1)].copy()
        else:
            predicted_df = pdf.copy()
        predicted_df[predicted_target] = y_predicted
        if prob_pos != -1:
            if best_pos != prob_pos:
                # mod_for_prob = algos[problem_type]["names"][prob_pos]
                mod_for_prob = eval(algos[problem_type]["names"][prob_pos])
                print(
                    f'Training started for algorithm with probability "{mod_for_prob}" for period {i + 1} of {num_of_periods} at {strftime("%Y-%m-%d %H:%M:%S")}'
                )
                mod_for_prob.fit(X_train, y_train)
                print(
                    f'Prediction started for algorithm with probability "{mod_for_prob}" for period {i + 1} of {num_of_periods} at {strftime("%Y-%m-%d %H:%M:%S")}'
                )
                y_prob_predict = mod_for_prob.predict(X)
                # prob_target = 'Predicted ' + str(target) + ' for ' + str(mod_for_prob).split("(")[0]
                prob_target = "Predicted for Probability " + str(target)
                predicted_df[prob_target] = y_prob_predict
            else:
                mod_for_prob = best_model
            try:
                prob = pd.DataFrame()
                print(
                    f"Probaility of classification started for period {i + 1} of {num_of_periods} at {strftime('%Y-%m-%d %H:%M:%S')}"
                )
                probabilities = mod_for_prob.predict_proba(X)
                algo_name = str(mod_for_prob).split("(")[0]
                # cols = [f'Probability for {i} (%) by {algo_name}' for i in mod_for_prob.classes_]
                if target in training_encoders.keys():
                    col_list = (
                        training_encoders[target]
                        .inverse_transform(mod_for_prob.classes_)
                        .tolist()
                    )
                else:
                    col_list = mod_for_prob.classes_.tolist()
                cols = [f"Probability for {i} (%)" for i in col_list]
                prob = pd.DataFrame(probabilities * 100, columns=cols)
                prob.reset_index(drop=True, inplace=True)
                predicted_df.reset_index(drop=True, inplace=True)
                predicted_df = predicted_df.merge(
                    prob, left_index=True, right_index=True
                )
            except Exception as e:
                e = str(e)
        final_predict = pd.concat([final_predict, predicted_df], ignore_index=True)
        X_train = np.concatenate((X_train, X))
        y_train = np.concatenate((y_train, y_predicted))

    if target in training_encoders.keys():
        final_predict[predicted_target] = training_encoders[target].inverse_transform(
            final_predict[predicted_target]
        )
    if prob_target is not None and target in training_encoders.keys():
        final_predict[prob_target] = training_encoders[target].inverse_transform(
            final_predict[prob_target]
        )

    if isinstance(predict_errors, pd.DataFrame):
        predict_errors = pd.DataFrame.to_dict(predict_errors)

    predicted_data_final = final_predict
    df_20 = predicted_data_final.head(20)
    print(f"Prediction completed at {strftime('%Y-%m-%d %H:%M:%S')}")
    return {
        "Final": df_20.to_json(orient="table", index=False),
        "Errors": predict_errors,
    }


@app.route("/Start", methods=["GET"])
def get_algorithm():
    """
    This function is called at the beginning of the program
    :return: Returns list of values in Algorithm Category select box
    """
    algo_list = list(algos.keys())
    algo_list.sort()
    print(algo_list)
    return {"Algorithms": algo_list}


@app.route("/Reset", methods=["GET"])
def reset_data():
    """
    This function is called to reset variables on change of Training file
    :return: None
    """
    global \
        training_data, \
        training_clean_data, \
        training_clean_data_copy, \
        training_data_num_copy, \
        training_data_num, \
        training_encoders, \
        training_features_data, \
        training_data_analysis, \
        training_file_path, \
        training_analysis, \
        correlation_target, \
        correlation_all, \
        predict_data, \
        predict_clean_data, \
        predict_data_num, \
        predict_clean_data_copy, \
        predict_data_num_copy, \
        predict_data_analysis, \
        predict_encoders, \
        predicted_data_final, \
        date_columns, \
        numeric_columns, \
        categorical_columns, \
        best_model_index, \
        probability_index, \
        pct_complete, \
        best_parameters, \
        g_confusion_matrix, \
        g_classification_report, \
        scaler, \
        outlier_columns, \
        sample_str, \
        feature_selection_algos, \
        perform_pca, \
        pca_features, \
        pca_algo, \
        perform_mca, \
        mca_features, \
        mca_algo

    training_data = pd.DataFrame()
    training_clean_data = pd.DataFrame()
    training_data_num = pd.DataFrame()
    training_clean_data_copy = pd.DataFrame()
    training_data_num_copy = pd.DataFrame()
    training_encoders = {}
    training_features_data = pd.DataFrame()
    training_analysis = pd.DataFrame()
    training_data_analysis = ""
    training_file_path = Path.cwd()
    correlation_target = pd.DataFrame()
    correlation_all = pd.DataFrame()
    predict_data = pd.DataFrame()
    predict_clean_data = pd.DataFrame()
    predict_data_num = pd.DataFrame()
    predict_clean_data_copy = pd.DataFrame()
    predict_data_num_copy = pd.DataFrame()
    predict_data_analysis = ""
    predict_encoders = {}
    predicted_data_final = pd.DataFrame()
    date_columns = []
    numeric_columns = []
    categorical_columns = []
    best_model_index = -1
    probability_index = -1
    pct_complete = 0.0
    best_parameters = pd.DataFrame()
    g_confusion_matrix = pd.DataFrame()
    g_classification_report = pd.DataFrame()
    scaler = {}
    outlier_columns = {}
    sample_str = ""
    feature_selection_algos = []
    perform_pca = False
    pca_features = []
    pca_algo = 'PCA(n_components=3, random_state=42)'
    perform_mca = False
    mca_features = []
    mca_algo = 'MCA(n_components=3, random_state=42)'

    return {"status": 200}


@app.route("/Reset_Predict", methods=["GET"])
def reset_predict_data():
    """
    This function is called to reset variables on change of Predict file
    :return: None
    """
    global \
        predict_data, \
        predict_clean_data, \
        predict_data_num, \
        predict_clean_data_copy, \
        predict_data_num_copy, \
        predict_encoders, \
        predict_data_analysis, \
        predicted_data_final
    predict_data = pd.DataFrame()
    predict_clean_data = pd.DataFrame()
    predict_data_num = pd.DataFrame()
    predict_clean_data_copy = pd.DataFrame()
    predict_data_num_copy = pd.DataFrame()
    predict_encoders = {}
    predict_data_analysis = ""
    predicted_data_final = pd.DataFrame()

    return {"status": 200}


@app.route("/Set_Features_Data", methods=["GET", "POST"])
def set_unique_features():
    """This function should be called whenever Features are changed as well as the first time after Target is selected
     and all Features are selected by default. The function creates a global DataFrame of unique values of Features

    Returns: None

    """
    features = json.loads(request.form.get("features"))
    global training_clean_data, training_features_data
    training_features_data = training_clean_data[features]
    training_features_data = training_features_data.drop_duplicates(ignore_index=True)
    return {"status": 200}


@app.route("/Get_Unique_Values", methods=["GET", "POST"])
def get_unique_values():
    """This function takes in a parameter in form of a dict e.g. {'Region': 'South', 'Loan Tenure': 240} -
    here the key value is the feature name and value is the value selected by the user. This function needs to be
    called every time user changes the input value in any of the dropdowns. The dict that needs to be sent should be
    all key value pairs and not just the changed selection. E.g. user selects Region as South -
    then send {'Region': 'South'} then user selects loan tenure 240 - then send {'Region': 'South', 'Loan Tenure': 240}
    and so on. The function would return a dict of unique values of the remaining Feature columns

    Args:
        condition_values: Dict in the form {'Region': 'South', 'Loan Tenure': 240}

    Returns: The function would return a dict of unique values of the remaining Feature columns e.g.
    {'Cust Id': ['C101001', 'C101002', 'C101006', 'C101007', 'C101021'],
    'Gender': ['Female', 'Male'], 'Segment': ['Agriculturist', 'HUF', 'NRI', 'Salaried',
    'Self Employed Non-Professional','Self Employed Professional']
    and so on }

    """
    condition_values = {}
    if request.method == "POST":
        condition_values = request.form.get("condition_values")
    global training_features_data
    X = training_features_data.copy()
    if not (set(condition_values.keys()).issubset(set(X.columns))):
        return "Drop Down keys are not matching the Feature names selected"
    if len(condition_values) != 0:
        X = X[np.logical_and.reduce([X[k] == v for k, v in condition_values.items()])]
    col_values = X.to_dict(orient="list")
    col_values = {k: sorted(list(set(v))) for k, v in col_values.items()}
    col_values = {
        k: v for k, v in col_values.items() if k not in list(condition_values.keys())
    }
    print(col_values)
    return JSON.dumps({"unique": JSON.dumps(col_values)})


@app.route("/Get_Algo_Position", methods=["GET", "POST"])
def get_algo_position():
    """
    This function returns a json dataframe with algorithm name, position and probability flag of algorithm
    :params problem_type: Algorithm Category
    :returns: json dataframe with Algorithm Name, Position and Probability Flag
    """
    global algos
    problem_type = request.form["algorithm"]
    algo_result = {}
    for algo in algos[problem_type]["names"]:
        prob_flag = False
        algo_name = algo
        algo_idx = algos[problem_type]["names"].index(algo)
        # if hasattr(algo, 'predict_proba'):
        if hasattr(eval(algo), "predict_proba"):
            prob_flag = True
        algo_result[algo_name] = [algo_idx, prob_flag]

    algo_results = pd.DataFrame.from_dict(algo_result).T
    algo_results.columns = ["Position", "Probability Flag"]
    algo_results.index.name = "Algorithm Name"
    algo_results.reset_index(inplace=True)
    return json.dumps(
        {"algo_results": algo_results.to_json(orient="table", index=False)}
    )


@app.route("/Tune_Hyperparameters", methods=["GET", "POST"])
def tune_hyperparameters():
    """
    This function is called on selecting Tune Hyperparameters value in front end drop down
    : param problem_type: Algorithm Category
    : param target: target column name as string
    : param feature_list: features column names as a List
    : param hyper_position: position of selected hyper algorithm
    : returns: hyper_df: Dataframe containing best parameters, confusion matrix and classification report
            best_params: string containing dictionary of best parameter values
            best_params_df: Dataframe containing dictionary of best parameter values
            confusion_matrix -  Confusion Matrix for best parameters
            classification_report - Classification Report for best parameters
    """
    print("Inside Function Tune Hyper Parameters")
    problem_type = request.form["algorithm"]
    target = request.form["target"]
    feature_list = json.loads(request.form["features"])
    hyper_position = int(request.form["hyper_position"])  # json.loads(request.form[''])

    print(f"Target selected is: {target}")
    print(f"Features selected are: {feature_list}")
    print(f"Algorithm selected is: {problem_type}")
    global \
        training_clean_data, \
        training_data_num, \
        training_encoders, \
        best_model_index, \
        probability_index, \
        pct_complete, \
        best_parameters, \
        g_confusion_matrix, \
        g_classification_report, \
        algos
    print(f"Hyper tuning parameters started at: {strftime('%Y-%m-%d %H:%M:%S')}")
    data = training_data_num.copy()

    X = data.loc[:, feature_list]
    y = data.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if sample_str or (
            "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
    ):
        X_train, y_train = sampling(problem_type, target, X_train, y_train)

    if perform_pca or perform_mca:
        X_train = perform_pca_mca(feature_list, X_train.copy())
    if perform_pca or perform_mca:
        X = perform_pca_mca(feature_list, X.copy())
    class_labels = None
    if "positive_target" in algos[problem_type]:
        class_labels = get_class_labels(problem_type=problem_type, target=target)
    #     class_labels = algos[problem_type]['positive_target']
    #     if target in training_encoders.keys():
    #         class_labels = training_encoders[target].transform(class_labels).tolist()
    # print(f'Class Labels are: {class_labels}')

    param_grid = algos[problem_type]["hyper_algos"][hyper_position][1]["param_grid"]
    param_grid = {
        k: eval(v) if isinstance(v, str) else v for k, v in param_grid.items()
    }
    algos[problem_type]["hyper_algos"][hyper_position][1]["param_grid"] = param_grid
    model_str = (
            algos[problem_type]["hyper_algos"][hyper_position][0]
            + "("
            + ", ".join(
        [
            f"{k}={v}"
            for k, v in algos[problem_type]["hyper_algos"][hyper_position][
            1
        ].items()
        ]
    )
            + ")"
    )

    print(f"Hyper Algo selected is {model_str}")
    make_score = algos[problem_type]["make_scorer"]
    model_str = model_str.replace("make_score", make_score)
    model = eval(model_str)
    print(strftime("%Y-%m-%d %H:%M:%S"))
    model.fit(X_train, y_train)
    predictions = model.best_estimator_.predict(X)
    if target in training_encoders.keys():
        predictions = training_encoders[target].inverse_transform(predictions)
    print(strftime("%Y-%m-%d %H:%M:%S"))
    best_params = model.best_params_
    best_params_str = "("
    for k, v in best_params.items():
        best_params_str = best_params_str + k + "="
        if v is None:
            best_params_str += "None"
        elif isinstance(v, str):
            best_params_str = best_params_str + "'" + v + "'"
        else:
            best_params_str += str(v)
        best_params_str += ", "
    best_params_str = best_params_str[:-2]
    best_params_str += ")"
    print(f"Best Parameters are: {best_params_str}")
    best_score = model.best_score_
    print(f"Best Score is: {best_score}")
    print(f"Scorer function used is: {model.scorer_}")
    if hasattr(model.best_estimator_, "feature_importances_"):
        print(f"Names of features used are: {model.feature_names_in_}")
    report = classification_report(
        training_clean_data.loc[:, target], predictions, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    cm = confusion_matrix(training_clean_data.loc[:, target], predictions)
    if target in training_encoders.keys():
        labels = list(training_encoders[target].classes_)
    else:
        labels = y.unique().tolist()
        labels.sort()

    actual_labels = ["Actual " + label for label in labels]
    prediction_labels = ["Predicted " + label for label in labels]
    best_params_df = pd.DataFrame(best_params, index=["Values"]).T
    best_params_df.index.names = ["Parameters"]
    best_params_df.reset_index(inplace=True)
    temp_df = pd.DataFrame(
        {"Parameters": ["Best Parameters"], "Values": [best_params_str]}
    )
    best_params_df = pd.concat([temp_df, best_params_df], ignore_index=True)
    cm_df = pd.DataFrame(cm, index=actual_labels, columns=prediction_labels)
    cm_df.reset_index(inplace=True)
    cm_df = cm_df.rename(columns={"index": "Confusion Matrix"})
    report_df = report_df.loc[labels]
    report_df.reset_index(inplace=True)
    report_df = report_df.rename(columns={"index": "Metrics"})
    print(f"Best Parameters")
    print(best_params_df.to_string())
    print("Confusion Matrix")
    print(cm_df.to_string())
    print("Classification Report")
    print(report_df.to_string())
    hyper_df = pd.DataFrame()
    features_imp_df = pd.DataFrame()
    if hasattr(model.best_estimator_, "feature_importances_"):
        features_imp = {
            "Feature Name": model.best_estimator_.feature_names_in_.tolist(),
            "Feature Importance": model.best_estimator_.feature_importances_.tolist(),
        }
        features_imp_df = pd.DataFrame(features_imp)
        features_imp_df.sort_values(
            by="Feature Importance", ascending=False, ignore_index=True, inplace=True
        )
        if features_imp_df.empty:
            print(
                f"The Algorithm selected - {str(model.best_estimator_)} - does not have feature importance attribute"
            )
        else:
            print("Feature Importance Matrix:")
            print(features_imp_df.to_string())
    else:
        print(
            f"The Algorithm selected - {str(model.best_estimator_)} - does not have feature importance attribute"
        )

    best_parameters = best_params_df
    g_confusion_matrix = cm_df
    g_classification_report = report_df
    return {
        "hyper": hyper_df.to_json(orient="table", index=False),
        "best_params": str(best_params_str),
        "best_params_df": best_params_df.to_json(orient="table", index=False),
        "confusion_matrix": cm_df.to_json(orient="table", index=False),
        "classification_report": report_df.to_json(orient="table", index=False),
    }


@app.route("/Get_Hyper_Algo_Position", methods=["GET", "POST"])
def get_hyper_algo_position():
    """
    This function returns a json dataframe with Hyperparameters algorithm name and position of Hyper algorithm
    :params problem_type: Algorithm Category
    :returns: json dataframe with Hyper Algorithm Name, Position and Hyper Parameters
    """
    print("Inside function Get Hyper Algo Position")
    global algos
    problem_type = request.form["algorithm"]
    hyper_algo_result = {}
    for idx, hyper_algo in enumerate(algos[problem_type]["hyper_algos"]):
        hyper_algo_name = hyper_algo[1]["estimator"]
        hyper_algo_idx = idx
        hyper_parameters = hyper_algo[1]["param_grid"]
        hyper_parameters = {
            k: eval(v) if isinstance(v, str) else v for k, v in hyper_parameters.items()
        }
        hyper_algo_result[hyper_algo_name] = [hyper_algo_idx, hyper_parameters]

    hyper_algo_results = pd.DataFrame.from_dict(hyper_algo_result).T
    hyper_algo_results.columns = ["Position", "Hyper Parameters"]
    hyper_algo_results.index.name = "Hyper Algorithm Name"
    hyper_algo_results.reset_index(inplace=True)
    return json.dumps(
        {"hyper_algo_results": hyper_algo_results.to_json(orient="table", index=False)}
    )


@app.route("/download", methods=["POST"])
def file_download():
    """
    This function is called to download the results in an excel file.
    :params sheets: list of tabs to be downloaded in excel sheet.
    :returns: excel file as a temporary file.
    """
    global \
        training_clean_data, \
        correlation_target, \
        correlation_all, \
        training_analysis, \
        best_parameters, \
        g_confusion_matrix, \
        g_classification_report, \
        predicted_data_final

    # filename = "Output_" + strftime("%Y%m%d_%H%M%S") + ".xlsx"
    zip_filename = "Output_" + strftime("%Y%m%d_%H%M%S") + ".zip"

    # Parse sheets into a list
    sheets = JSON.loads(request.form["sheets"])
    print(sheets)
    print(type(sheets))
    if isinstance(sheets, str):  # If it's a string, convert to list
        sheets = sheets.split(",")

    download_sheets = {
        "Original Sheet": ["Original Sheet", training_clean_data],
        "Correlation 1": ["Correlation with Target", correlation_target],
        "Correlation 2": ["Correlation All", correlation_all],
        "Analysis": ["Training Analysis", training_analysis],
        "Best Params": ["Best Parameters", best_parameters],
        "Confusion Matrix": ["Confusion Matrix", g_confusion_matrix],
        "Classification Report": ["Classification Report", g_classification_report],
        "Prediction Sheet": ["Prediction Sheet", predicted_data_final],
    }
    print(f'Columns in training_clean_data are: {training_clean_data.columns}')
    print(f'Zip File generation started in python at: {strftime("%H:%M:%S")}')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for sheet in sheets:
            if sheet in download_sheets:
                csv_buffer = io.StringIO()
                download_sheets[sheet][1].to_csv(csv_buffer, index=False)
                zip_file.writestr(f'{download_sheets[sheet][0]}.csv', csv_buffer.getvalue())
    zip_buffer.seek(0)
    print(f'Zip File generation completed in python at: {strftime("%H:%M:%S")}')
    print(f'Name of zip file is {zip_filename}')
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=zip_filename,
        mimetype="application/zip",
    )

    # Create a temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
    #     with pd.ExcelWriter(temp_file.name, engine="xlsxwriter") as writer:
    #         for sheet in sheets:
    #             if (
    #                     sheet in download_sheets
    #             ):  # Check if the sheet exists in the dictionary
    #                 download_sheets[sheet][1].to_excel(
    #                     writer, sheet_name=download_sheets[sheet][0], index=False
    #                 )
    #     # Send the file as an attachment
    #     return send_file(
    #         temp_file.name,
    #         as_attachment=True,
    #         download_name=filename,
    #         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #     )


@app.route("/Hypertune_Estimate", methods=["GET", "POST"])
def hypertune_time_estimate():
    """
    This function is called on selecting Tune Hyperparameters value in front end drop down
    : param problem_type: Algorithm Category
    : param target: target column name as string
    : param feature_list: features column names as a List
    : param hyper_position: position of selected hyper algorithm
    : returns: no_of_fits: Number of fits the hypertuning shall run for
            estimated_time: Estimated total time in minutes the hypertuning shall run for
    """
    print("Inside Function Hypertune - Number of Fits and Time Estimator")
    problem_type = request.form["algorithm"]
    target = request.form["target"]
    feature_list = json.loads(request.form["features"])
    hyper_position = int(request.form["hyper_position"])  # json.loads(request.form[''])

    global training_data_num, algos
    data = training_data_num.copy()
    X = data.loc[:, feature_list]
    y = data.loc[:, target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if sample_str or (
            "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
    ):
        X_train, y_train = sampling(problem_type, target, X_train, y_train)
    if perform_pca or perform_mca:
        X_train = perform_pca_mca(feature_list, X_train.copy())
        X_test = perform_pca_mca(feature_list, X_test.copy())
    param_grid = algos[problem_type]["hyper_algos"][hyper_position][1]["param_grid"]
    param_grid = {
        k: eval(v) if isinstance(v, str) else v for k, v in param_grid.items()
    }
    try:
        cv = algos[problem_type]["hyper_algos"][hyper_position][1]["cv"]
        print('Found "cv" for selected hyper algo')
    except:
        print(
            'Did not find "cv" for selected hyper algo. Defaulting value of "cv" to 1'
        )
        cv = 1
    no_of_combinations = np.prod([len(v) for v in param_grid.values()])
    no_of_fits = no_of_combinations * cv
    model_str = algos[problem_type]["hyper_algos"][hyper_position][1]["estimator"]
    make_score = algos[problem_type]["make_scorer"]
    model_str = model_str.replace("make_score", make_score)
    print(f"Hyper Algo selected is {model_str}")
    model = eval(model_str)
    print(f"Number of fits is: {no_of_fits}")
    start_time = time()
    for i in range(5):
        print(f"Running algo {model_str}: {i + 1} of 5 times")
        model.fit(X_train, y_train)
    end_time = time()
    print(f"Time taken (in seconds) to fit algo 5 times is: {end_time - start_time}")
    estimated_time = (end_time - start_time) * no_of_fits / 5 / 60
    print(
        f"Number of fits is: {no_of_fits} and estimated time in minutes to run all the fits is: {estimated_time}"
    )
    hyper_df = pd.DataFrame()
    best_params_str = ""
    best_params_df = pd.DataFrame()
    cm_df = pd.DataFrame()
    report_df = pd.DataFrame()
    return json.dumps(
        {"no_of_fits": no_of_fits.item(), "estimated_time": estimated_time.item()}
    )


@app.route("/Summarise_Data", methods=["GET", "POST"])
def summarise_data():
    """
    This function is called on selecting feature value in front end drop down in visualisation screen
    : param target: target column name as string
    : param feature_selected: selected feature names string
    : returns: summary_data: Pivot table of count with rows as selected Feature columns as distinct values
    in Target
    """
    target = request.form["target"]
    feature_selected = request.form["feature_selected"]
    global training_clean_data
    df_pivot = pd.pivot_table(
        training_clean_data,
        values=training_clean_data.columns[0],
        index=feature_selected,
        columns=target,
        aggfunc="count",
    )
    df_pivot.reset_index(inplace=True)
    df_dict = df_pivot.to_dict()
    df_pivot = pd.DataFrame(df_dict).fillna(0)
    print(df_pivot.to_dict(orient="records"))
    return json.dumps(df_pivot.to_dict(orient="records"))


@app.route("/atomic_Summarise_Data", methods=["POST"])
def atomic_summarise_data():
    """
    Process an uploaded file and return summarized data for all columns in the desired format.
    """
    try:
        # Load the file into a Pandas DataFrame
        file = request.files["file"]
        df = co.read_data(file)
        print(df)

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).head()
        print(numeric_cols)

        if numeric_cols.empty:  # Check if DataFrame is empty
            return json.dumps({"error": "No numeric columns found in the file"}), 400

        # Convert to dictionary for JSON response
        result = numeric_cols.to_dict(orient="records")
        print(json.dumps(result))
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)}), 400


def sampling(problem_type: str, target: str, X_train, y_train):
    """
    This function accepts data that needs to be sampled and returns sampled data
    if "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
    """
    global algos, training_encoders, sample_str
    # if "sampling" not in algos[problem_type]:
    #     print(
    #         f'To sample data, set key: "sampling" to True in algos.json file under key "{problem_type}"'
    #     )
    #     print("Example entry is:")
    #     print('"sampling": true')
    #     print("Data has not been sampled")
    #     return X_train, y_train
    if not sample_str:
        if "sampling_algo" not in algos[problem_type]:
            print(
                f'To sample data, set key: "sampling_algo" as a string with function name of sampling algo '
                f'to be used in algos.json file under key "{problem_type}"'
            )
            print("Example entry is:")
            print('"sampling_algo": "SMOTE(random_state=2)"')
            print("Data has not been sampled")
            return X_train, y_train
        sample_str = algos[problem_type]["sampling_algo"]
    try:
        sample = eval(sample_str)
    except Exception as e:
        print(f"Unable to perform sampling due to below error")
        print(f"Error Type is: {type(e)}")
        print(f"Error Details are: {str(e)}")
        print("Data has not been sampled")
        error_message = f'Error During Sampling\nUnable to perform sampling due to below error'
        error_message += f"\nError Type is: {type(e)}\nError Details are: {str(e)}\nData has not been sampled"
        alertFrontend(error_message)
        return X_train, y_train
    print(f"Before Sampling, number of records in {target} column are as below:")
    frontend_message = f"SAMPLING UPDATE\n"
    frontend_message += f"Before Sampling, number of records in {target} column are as below:\n"
    target_mapping = {}
    if target in training_encoders.keys():
        target_mapping = dict(
            zip(
                training_encoders[target].classes_,
                range(len(training_encoders[target].classes_)),
            )
        )
    if target_mapping:
        for k, v in target_mapping.items():
            print(f'Number of records with value: "{k}" are: {sum(y_train == v)}')
            frontend_message += f'Number of records with value: "{k}" are: {sum(y_train == v)}\n'
    else:
        for val in sorted(y_train.unique()):
            print(f'Number of records with value: "{val}" are: {sum(y_train == val)}')
            frontend_message += f'Number of records with value: "{val}" are: {sum(y_train == val)}\n'
    try:
        X_train, y_train = sample.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"Unable to perform sampling due to below error")
        print(f"Error Type is: {type(e)}")
        print(f"Error Details are: {str(e)}")
        print("Data has not been sampled")
        error_message = f'Error During Sampling\nUnable to perform sampling due to below error'
        error_message += f"\nError Type is: {type(e)}\nError Details are: {str(e)}\nData has not been sampled"
        alertFrontend(error_message)
        return X_train, y_train
    print(f"After Sampling, number of records in {target} column are as below:")
    frontend_message += f"After Sampling, number of records in {target} column are as below:\n"
    if target_mapping:
        for k, v in target_mapping.items():
            print(f'Number of records with value: "{k}" are: {sum(y_train == v)}')
            frontend_message += f'Number of records with value: "{k}" are: {sum(y_train == v)}\n'
    else:
        for val in sorted(y_train.unique()):
            print(f'Number of records with value: "{val}" are: {sum(y_train == val)}')
            frontend_message += f'Number of records with value: "{val}" are: {sum(y_train == val)}\n'
    alertFrontend(frontend_message)
    return X_train, y_train


def get_class_labels(problem_type: str, target: str):
    """
    This function returns the class_labels set in algos.json file. If the target is encoded, it returns the
    encoded class_labels
    """
    global algos
    cls_labels = algos[problem_type]["positive_target"]
    if target in training_encoders.keys():
        cls_labels = training_encoders[target].transform(cls_labels).tolist()
    print(f"Class Labels are: {cls_labels}")
    return cls_labels


def feature_selection(problem_type: str, target: str, feature_list: list):
    """
    This function accepts the data with Features and target and returns the optimal selected Features
    if (
            "feature_selection" in algos[problem_type]
            and algos[problem_type]["feature_selection"]
    ):
    """
    global algos, training_encoders, training_data_num
    print(f"Feature Selection function started at: {strftime('%H:%M:%S')}")
    # if "feature_selection_algos" not in algos[problem_type]:
    #     print(
    #         f'To select optimal feature selection, set key: "feature_selection_algos" as a nested list with function '
    #         f'names of feature selection algos and parameters as dictionary in algos.json file under key "'
    #         f'{problem_type}"'
    #     )
    #     print("Example entry is:")
    #     print(
    #         '"feature_selection_algos": [["RFECV",{"estimator": "RandomForestClassifier(random_state=42)",'
    #         '"cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)","scoring":"make_scorer(recall_score, '
    #         'average=\'\\"micro\\"\', labels=class_labels)","verbose":0}]]"'
    #     )
    #     print("Feature Selection function has not been run")
    #     return
    X = training_data_num[feature_list]
    y = training_data_num[target]
    if sample_str or (
            "sampling" in algos[problem_type] and algos[problem_type]["sampling"]
    ):
        X, y = sampling(problem_type, target, X, y)
    feature_summary = []
    if "positive_target" in algos[problem_type]:
        class_labels = get_class_labels(problem_type=problem_type, target=target)
    if not feature_selection_algos:
        feature_algos = algos[problem_type]["feature_selection_algos"]
    else:
        feature_algos = feature_selection_algos
    for item in feature_algos:
        try:
            params = item[1]
        except:
            print(
                f'To select optimal feature selection, set key: "feature_selection_algos" as a nested list with '
                f"function names of feature selection algos and parameters as dictionary in algos.json file under key "
                f'"{problem_type}"'
            )
            print("Example entry is:")
            print(
                '"feature_selection_algos": [["RFECV",{"estimator": "RandomForestClassifier(random_state=42)",'
                '"cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)","scoring":"make_scorer('
                'recall_score, average=\'\\"micro\\"\', labels=class_labels)","verbose":0}]]"'
            )
            print("Feature Selection function has not been run")
            continue
        feature_estimator = params["estimator"] if "estimator" in params else params
        print(f"Feature selection algo is: {feature_estimator}")
        try:
            params = {
                k: eval(v) if isinstance(v, str) else v for k, v in params.items()
            }
        except Exception as e:
            print(f"Unable to perform feature selection due to below error")
            print(f"Error Type is: {type(e)}")
            print(f"Error Details are: {str(e)}")
            continue
        feature_selection_algo_str = (
                item[0] + "(" + ", ".join([f"{k}={v}" for k, v in params.items()]) + ")"
        )
        try:
            feature_selection_algo = eval(feature_selection_algo_str)
        except Exception as e:
            print(f"Unable to perform feature selection due to below error")
            print(f"Error Type is: {type(e)}")
            print(f"Error Details are: {str(e)}")
            continue
        try:
            print(
                f"Feature selection algo : {feature_estimator} started at: {strftime('%H:%M:%S')}"
            )
            start_time = time()
            feature_selection_algo.fit(X, y)
            end_time = time()
            print(
                f"Feature selection algo : {feature_estimator} completed at: {strftime('%H:%M:%S')}"
            )
        except Exception as e:
            print(f"Unable to perform feature selection due to below error")
            print(f"Error Type is: {type(e)}")
            print(f"Error Details are: {str(e)}")
            continue
        ranking_linear = pd.Series(
            feature_selection_algo.ranking_, index=X.columns
        ).sort_values()
        # print("Optimal number of features:", feature_selection_algo.n_features_)
        # print("Selected features:", list(X.columns[feature_selection_algo.support_]))
        # print("Maximum score is:", max(feature_selection_algo.cv_results_['mean_test_score']))
        # print('Feature Ranking')
        # print(ranking_linear)
        selected_features = list(X.columns[feature_selection_algo.support_])
        selected_features.sort()
        feature_summary.append(
            {
                "Feature Selection Algo": feature_estimator,
                "Optimal number of Features": feature_selection_algo.n_features_,
                "Selected Features": selected_features,
                "Max Score": round(
                    max(feature_selection_algo.cv_results_["mean_test_score"]), 4
                ),
                "Feature Ranking": ranking_linear,
                "Duration in Seconds": round(end_time - start_time, 2),
            }
        )
    # feature_summary_df = pd.DataFrame.from_dict(feature_summary)
    # print(feature_summary)
    print(
        "____________________________________________________________________________________________________"
        "____________"
    )
    print("")
    combined_features = []
    frontend_message = f'Feature Selection Summary\n'
    for feature_result in feature_summary:
        for k, v in feature_result.items():
            if k != "Feature Ranking":
                print(f"{k}: {v}")
                frontend_message += f"{k}: {v}\n"
            else:
                print(f"{k}:")
                frontend_message += f"{k}:\n"
                print(f"{v}")
                frontend_message += f"{v}\n"
            if k == "Selected Features":
                combined_features += v
        print(
            "______________________________________________________________________________________________________"
            "__________"
        )
        print("")
    combined_features = list(set(combined_features))
    combined_features.sort()
    combined = {
        "Combined number of Features": len(combined_features),
        "Combined Feature List": combined_features,
    }
    for k, v in combined.items():
        print(f"{k}: {v}")
        frontend_message += f"{k}: {v}\n"
    print(
        "____________________________________________________________________________________________________"
        "____________"
    )
    print("")
    print(f"Feature Selection function completed at: {strftime('%H:%M:%S')}")
    # print(feature_summary_df.to_string())
    alertFrontend(frontend_message)
    feature_summary_df = pd.DataFrame(feature_summary)
    feature_summary_df.drop(columns=['Feature Ranking'], inplace=True)
    feature_summary_df.loc[len(feature_summary_df)] = ["Combined Features", len(combined_features), combined_features,
                                                       feature_summary_df['Max Score'].max(),
                                                       feature_summary_df['Duration in Seconds'].sum()]
    return feature_summary_df


@app.route("/Outlier_Analysis", methods=["GET", "POST"])
def outlier_analysis():
    """
    This function is called after clicking on Outlier Analysis button in front end drop down
        : param feature_list: list of all selected feature in left frame
        : returns: counts: Array of tables with each table containing distinct values of a feature and corresponding
        number of records for that value
    """
    global training_data_num
    feature_list = json.loads(request.form["features"])
    counts = []
    for column in feature_list:
        counts.append(
            training_data_num[column]
            .value_counts()
            .reset_index()
            .rename(columns={"index": column, 0: "Count"})
            .sort_values(by=column)
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
    return counts


def iqr(q1=0.25, q3=0.75, outlier_coefficient=1.5, **kwargs):
    """
    This function calculates the outliers for given feature using Inter Quartile Range.
    The data is passed from outlier_removal function as key value pair
    key - data sends the training_data_num and
    key - column sends the feature name set in algos.json
    """
    df = kwargs["data"]
    column = kwargs["column"]
    Q1 = df[column].quantile(q1)
    Q3 = df[column].quantile(q3)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_coefficient * IQR
    upper_bound = Q3 + outlier_coefficient * IQR
    outlier_index = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    print(
        f"Number of rows that will be removed for feature {column} is {len(outlier_index)}"
    )
    frontend_message = 'Outlier Removal - IQR\n'
    frontend_message += f"Number of rows that will be removed for feature {column} is {len(outlier_index)}\n"
    alertFrontend(frontend_message)
    return outlier_index


def zscore(threshold=3, **kwargs):
    """
    This function calculates the outliers for given feature using z_score method.
    The data is passed from outlier_removal function as key value pair
    key - data sends the training_data_num and
    key - column sends the feature name set in algos.json
    """
    df = kwargs["data"]
    column = kwargs["column"]
    z_scores = np.abs(stats.zscore(df[column]))
    outlier_index = df[z_scores > threshold].index
    print(
        f"Number of rows that will be removed for feature {column} is {len(outlier_index)}"
    )
    frontend_message = 'Outlier Removal - zscore\n'
    frontend_message += f"Number of rows that will be removed for feature {column} is {len(outlier_index)}\n"
    alertFrontend(frontend_message)
    return outlier_index


def remove_percentile(lower_percentile=1, upper_percentile=99, **kwargs):
    """
    This function calculates the outliers for given feature using percentile method.
    The data is passed from outlier_removal function as key value pair
    key - data sends the training_data_num and
    key - column sends the feature name set in algos.json
    """
    df = kwargs["data"]
    column = kwargs["column"]
    lower_bound = np.percentile(df[column], lower_percentile)
    upper_bound = np.percentile(df[column], upper_percentile)
    outlier_index = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    print(
        f"Number of rows that will be removed for feature {column} is {len(outlier_index)}"
    )
    frontend_message = 'Outlier Removal - remove_percentile\n'
    frontend_message += f"Number of rows that will be removed for feature {column} is {len(outlier_index)}\n"
    alertFrontend(frontend_message)
    return outlier_index


def remove_outliers(problem_type, outlier_removal=None):
    """
    This function reads the algos.json for feature name and method to calculate the outlier for that feature
    The feature name and data is passed to relevant function as key value pair
    """
    if outlier_removal is None:
        outlier_removal = {}
    global algos, training_data_num, training_clean_data, outlier_columns
    print(f"Outlier Removal started at: {strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Count of rows in training data before outlier removal are: {len(training_data_num)}"
    )
    frontend_message = 'Outlier Removal\n'
    frontend_message += f"Count of rows in training data before outlier removal are: {len(training_data_num)}\n"
    if not outlier_removal:
        if "outlier_columns" not in algos[problem_type]:
            print(
                f'To remove outliers, set key: "outlier_columns"  in algos.json file under key {problem_type}'
            )
            print("Example entry is:")
            print(
                '"outlier_columns": {"Age": "IQR(q1=0.25, q3=0.75, outlier_coefficient=1.5)", "EMI Amount": "zscore('
                'threshold=3)", "Pincode": "remove_percentile(lower_percentile=1, upper_percentile=99)"}'
            )
            print("Outlier Removal not performed")
            return
        outlier_columns = algos[problem_type]["outlier_columns"]
    outlier_columns = outlier_removal
    outlier_indices = set()
    for column, method_str in outlier_columns.items():
        if column not in training_data_num:
            print(f"Feature: {column} not found in data")
            print(f"Skipping Outlier Removal for feature: {column}")
            continue
        method_str = method_str.replace(
            ")", f', data=training_data_num, column="{column}")'
        )
        print(method_str)
        try:
            method = eval(method_str)
            outlier_indices.update(method)
        except Exception as e:
            print(
                f"Unable to perform Outlier Removal for feature: {column} due to below error"
            )
            print(f"Error Type is: {type(e)}")
            print(f"Error Details are: {str(e)}")
            print(f"Skipping Outlier Removal for feature: {column}")
            continue
    training_data_num = training_data_num.drop(index=list(outlier_indices)).reset_index(
        drop=True
    )
    training_clean_data = training_clean_data.drop(
        index=list(outlier_indices)
    ).reset_index(drop=True)
    print(
        f"Count of rows in training data after outlier removal are: {len(training_data_num)}"
    )
    frontend_message += f"Count of rows in training data after outlier removal are: {len(training_data_num)}\n"
    print(f"Outlier Removal completed at: {strftime('%Y-%m-%d %H:%M:%S')}")
    alertFrontend(frontend_message)
    return


def data_scaling(problem_type: str, scaled_data: pd.DataFrame = pd.DataFrame(), scaling=None):
    """
    This function scales the data as per scaling_algos set in algos.json file
    """
    global algos, scaler, training_data_num
    print(f"Scaling Data started at: {strftime('%Y-%m-%d %H:%M:%S')}")
    frontend_message = 'Data Scaling\n'
    frontend_message += f"Scaling Data started at: {strftime('%Y-%m-%d %H:%M:%S')}\n"
    if scaled_data.empty:
        scaled_data = training_data_num
    if scaling is None:
        scaling = {}
    if not scaling:
        if "scaling_algos" not in algos[problem_type]:
            print(
                f'To scale data , set key: "scaling_algos" as a dict with key as scaling function '
                f'names and values as list of columns that need to be scaled in algos.json file under key "'
                f'{problem_type}"'
            )
            print("Example entry is:")
            print(
                '"scaling_algos":'
                "{"
                '"MinMaxScaler()": ["Age", "Region"],'
                '"StandardScaler()": ["EMI Amount"]'
                "}"
            )
            print("Data Scaling has not been run")
            return
        scaler = algos[problem_type]["scaling_algos"]
    scaler = scaling
    data_columns = scaled_data.columns.tolist()
    for scaling_function_str, columns in scaler.items():
        if not set(columns) & set(data_columns) == set(columns):
            print(f"Some or all of Features: {columns} not found in data")
            print(f"Skipping scaling for: {scaling_function_str}")
            continue
        try:
            scaling_function = eval(scaling_function_str)
            scaled_data[columns] = scaling_function.fit_transform(scaled_data[columns])
        except Exception as e:
            print(f"Unable to perform scaling due to below error")
            print(f"Error Type is: {type(e)}")
            print(f"Error Details are: {str(e)}")
            continue

    print(f"Scaling Data completed at: {strftime('%Y-%m-%d %H:%M:%S')}")
    frontend_message += f"Scaling Data completed at: {strftime('%Y-%m-%d %H:%M:%S')}"
    alertFrontend(frontend_message)

    return


@app.route("/DynamicSidebarConfig", methods=["GET", "POST"])
def dynamic_sidebar_config():
    config = [
        {
            "name": "Remove Outliers",
            "sidebarName": "Remove Outliers",
            "buttonText": "Remove Outliers",
            "modes": ["outlier_removal"],
        },
        {
            "name": "Perform Scaling",
            "sidebarName": "Scaling",
            "buttonText": "Perform Scaling",
            "modes": ["scaling"],
        },

        {
            "name": "Perform Feature Selection",
            "sidebarName": "Feature Selection",
            "buttonText": "Top Features Selection",
            "modes": ["feature_selection"],
        },
        {
            "name": "Perform Sampling",
            "sidebarName": "Sampling",
            "buttonText": "Perform Sampling",
            "modes": ["sampling"],
        },
        {
            "name": "Perform PCA",
            "sidebarName": "PCA",
            "buttonText": "Perform PCA",
            "modes": ["perform_pca"],
        },
        {
            "name": "Perform MCA",
            "sidebarName": "MCA",
            "buttonText": "Perform MCA",
            "modes": ["perform_mca"],
        },
        {
            "name": "Reset Preprocess Options",
            "sidebarName": "Reset All PreProcess Options",
            "buttonText": "Reset All PreProcess Options",
            "modes": ["reset_preprocess"],
        },
    ]
    return JSON.dumps({"data": config})


@app.route("/PreProcessConfig", methods=["POST"])
def configure_pre_process():
    """
    This function is called to prepare the configuration of screens for pre-processing.
        : param: problem_type: this is the algorithm selected
        : param: feature_list: this is the list of features selected
    """
    problem_type = request.form["algorithm"]
    feature_list = json.loads(request.form["features"])
    global algos, numeric_columns, categorical_columns
    numeric_features = list(set(feature_list) & set(numeric_columns))
    numeric_features.sort()
    categorical_features = list(set(feature_list) & set(categorical_columns))
    print(f'Numeric Features are: {numeric_features}')
    pre_process_config = algos[problem_type]["pre_process"]
    for k, v in pre_process_config.items():
        if k == 'reset_preprocess':
            continue
        if k == 'sampling' or k == 'feature_selection':
            v['dropdown'] = False
            v["feature_names"] = ['All Selected Features']
        elif k == 'perform_mca':
            v['dropdown'] = True
            v["feature_names"] = categorical_features
        else:
            v['dropdown'] = True
            v["feature_names"] = numeric_features

    output_format = {
        "outlier_removal": [
            {
                "feature_names": ["Age"],
                "function_name": "iqr",
                "parameters": {"q1": 0.25, "q3": 0.75, "outlier_coefficient": 1.5},
            },
            {
                "feature_names": ["EMI Amount"],
                "function_name": "zscore",
                "parameters": {"threshold": 3},
            },
            {
                "feature_names": ["Pincode", "Loan Amount"],
                "function_name": "remove_percentile",
                "parameters": {"lower_percentile": 1, "upper_percentile": 99},
            },
        ],
        "sampling": [
            {
                "feature_names": [
                    "Age",
                    "EMI Amount",
                    "Pincode",
                    "Region",
                    "Gross Disbursal",
                    "Loan Amount",
                ],
                "function_name": "SMOTE()",
                "parameters": {"random_state": 42, "sampling_strategy": "'auto'"},
            }
        ],
        "scaling": [
            {
                "feature_names": ["Age", "Region"],
                "function_name": "MinMaxScaler()",
                "parameters": {"copy": True},
            },
            {
                "feature_names": ["EMI Amount"],
                "function_name": "StandardScaler()",
                "parameters": {"copy": True},
            },
        ],
        "feature_selection": [
            [
                "RFECV",
                {
                    "estimator": "RandomForestClassifier(random_state=42)",
                    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
                    "scoring": "make_scorer(recall_score, average='\"micro\"', labels=class_labels)",
                    "verbose": 1,
                },
            ],
            [
                "RFECV",
                {
                    "estimator": "LogisticRegression(penalty='l2', solver='liblinear', random_state=42)",
                    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
                    "scoring": "make_scorer(recall_score, average='\"micro\"', labels=class_labels)",
                    "verbose": 1,
                },
            ],
        ],
    }

    return JSON.dumps(
        {"preprocess_config": pre_process_config, "output_format": output_format}
    )


def reset_preprocess():
    """
    This function is called to reset data that was changed in preprocessing
    """
    global scaler, outlier_columns, sample_str, feature_selection_algos, training_data_num, training_data_num_copy, \
        training_clean_data, training_clean_data_copy, predict_clean_data, predict_clean_data_copy, \
        predict_data_num, predict_data_num_copy, perform_pca, pca_features, pca_algo, \
        perform_mca, mca_features, mca_algo
    scaler = {}
    outlier_columns = {}
    sample_str = ""
    feature_selection_algos = []
    perform_pca = False
    pca_features = []
    pca_algo = 'PCA(n_components=3, random_state=42)'
    perform_mca = False
    mca_features = []
    mca_algo = 'MCA(n_components=3, random_state=42)'

    frontend_message = f'Reset Preprocess Options and Data\n'
    frontend_message += (f'Preprocessing options related to "Remove Outliers", "Scaling", "Feature Selection", '
                         f'"Sampling" have been reset\n')

    print_message = f'Rows in Training data before reset are: {training_data_num.shape[0]}\n'
    print(print_message)
    frontend_message += print_message

    training_clean_data = training_clean_data_copy.copy()
    training_data_num = training_data_num_copy.copy()

    print_message = f'Rows in Training data after reset are: {training_data_num.shape[0]}\n'
    print(print_message)
    frontend_message += print_message

    if not predict_data_num_copy.empty:
        print_message = f'Rows in Predict data before reset are: {predict_data_num.shape[0]}\n'
        print(print_message)
        frontend_message += print_message

        predict_clean_data = predict_clean_data_copy.copy()
        predict_data_num = predict_data_num_copy.copy()

        print_message = f'Rows in Predict data after reset are: {predict_data_num.shape[0]}\n'
        print(print_message)
        frontend_message += print_message

    alertFrontend(frontend_message)
    return


@app.route("/PreProcess", methods=["POST"])
def pre_process():
    """
    This function is called from front end screen during pre-processing. This function parses the input received for
    pre-processing and depending on the action to be performed - it executes - outlier removal, scaling, sampling and
    feature selection functionality
    """
    result = {"status": 200, 'has_table': False}

    problem_type = request.form["algorithm"]
    preprocess = json.loads(request.form["pre_process"])
    target = request.form["target"]
    feature_list = json.loads(request.form["features"])
    global sample_str, feature_selection_algos, perform_pca, pca_features, pca_algo, \
        perform_mca, mca_features, mca_algo
    selected_features = []
    for k, v in preprocess.items():
        if k == 'perform_pca' or k == 'perform_mca':
            for idx, func_details in enumerate(v):
                func_details['function'] = func_details['function_name']
                selected_features += func_details['feature_names']
            selected_features.sort()
            continue
        for idx, func_details in enumerate(v):
            func_details["function_name"] = (
                func_details["function_name"].replace("(", "").replace(")", "")
            )
            func_details["function"] = (
                    func_details["function_name"]
                    + "("
                    + ", ".join([f"{k}={v}" for k, v in func_details["parameters"].items()])
                    + ")"
            )

    for k, v in preprocess.items():
        if k == "outlier_removal":
            outlier_removal = {}
            for idx, func_details in enumerate(v):
                for col in func_details["feature_names"]:
                    outlier_removal[col] = func_details["function"]
            remove_outliers(problem_type=problem_type, outlier_removal=outlier_removal)
        if k == "scaling":
            scaling = {}
            for func_details in v:
                scaling[func_details["function"]] = func_details["feature_names"]
            data_scaling(problem_type=problem_type, scaling=scaling)
        if k == "sampling":
            for func_details in v:
                sample_str = func_details['function']
            frontend_message = 'Data Sampling\n'
            frontend_message += f"Sampling  of data shall be done before the following actions as appropriate:\n"
            frontend_message += f"1. Identification of optimal features using Feature Selection\n"
            frontend_message += f"2. Training of Algorithms to build Model\n"
            frontend_message += f"3. Prediction using selected Model\n"
            frontend_message += f"4. Hyper Parameter Tuning of Algorithms\n"
            alertFrontend(frontend_message)
        if k == 'feature_selection':
            # feature_selection_algos = v
            for func_details in v:
                feature_selection_algos.append([func_details['function_name'], func_details['parameters']])
            features_table = feature_selection(problem_type, target, feature_list)

            result["table"] = features_table.to_json(orient="table", index=False)
            result['has_table'] = True
            result['table_name'] = "Feature Selection"
        if k == 'perform_pca':
            perform_pca = True
            pca_features = selected_features
            pca_algo = v[0]['function_name']
            frontend_message = 'Perform PCA\n'
            frontend_message += f'PCA Algo selected is: {pca_algo}\n'
            frontend_message += f'Features selected for PCA are: {pca_features}\n'
            frontend_message += f"PCA  of data shall be done before the following actions as appropriate:\n"
            frontend_message += f"1. Training of Algorithms to build Model\n"
            frontend_message += f"2. Prediction using selected Model\n"
            frontend_message += f"3. Hyper Parameter Tuning of Algorithms\n"
            alertFrontend(frontend_message)
        if k == 'perform_mca':
            perform_mca = True
            mca_features = selected_features
            mca_algo = v[0]['function_name']
            frontend_message = 'Perform MCA\n'
            frontend_message += f'MCA Algo selected is: {mca_algo}\n'
            frontend_message += f'Features selected for MCA are: {mca_features}\n'
            frontend_message += f"MCA  of data shall be done before the following actions as appropriate:\n"
            frontend_message += f"1. Training of Algorithms to build Model\n"
            frontend_message += f"2. Prediction using selected Model\n"
            frontend_message += f"3. Hyper Parameter Tuning of Algorithms\n"
            alertFrontend(frontend_message)
        if k == 'reset_preprocess':
            reset_preprocess()

        toggleRefetchOutlierAnalysisData()
    return result


def perform_pca_mca(feature_list, X_train):
    """
    This function performs Principal Component Analysis and Multiple Correspondence Analysis
    """
    global pca_algo, mca_algo
    X_train_pca, X_train_mca = pd.DataFrame(), pd.DataFrame()
    remaining_features = feature_list
    print(f'Features before performing PCA / MCA are {X_train.shape[1]}')
    if perform_pca:
        X_train_pca = X_train[pca_features].copy()
        std_scaler = StandardScaler()
        X_train_pca = std_scaler.fit_transform(X_train_pca)
        remaining_features = list(set(remaining_features) - set(pca_features))
    if perform_mca:
        X_train_mca = X_train[mca_features].copy()
        for col in mca_features:
            X_train_mca[col] = X_train_mca[col].astype('category')
        remaining_features = list(set(remaining_features) - set(mca_features))
    if remaining_features:
        X_train = X_train[remaining_features].to_numpy()
    if perform_pca:
        pca = eval(pca_algo)  # PCA(n_components=3, random_state=42)
        X_train_pca = pca.fit_transform(X_train_pca)
        X_train = np.hstack((X_train_pca, X_train))
    if perform_mca:
        mca = eval(mca_algo)  # MCA(n_components=3, random_state=42)
        X_train_mca = mca.fit_transform(X_train_mca)
        X_train = np.hstack((X_train_mca, X_train))
    print(f'Features after performing PCA / MCA are {X_train.shape[1]}')
    return X_train
