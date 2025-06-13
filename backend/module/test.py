from module.CommonObjects import colsTest, featuresTest, aggregatorTest, read_data
from flask import Blueprint, json, request
# from icecream import ic

test = Blueprint('test', __name__)


@test.route('/', methods=['POST'])
def TesterRoute():
    try:
        train_file = request.files['trainingFile']
        predict_file = request.files['predictionFile']

        aggregator = json.loads(request.form.get('aggregator'))
        features = json.loads(request.form.get('features'))
        target = request.form.get('target')

        pred_df = read_data(predict_file)  # pd.read_csv('predict_sales.csv')
        train_df = read_data(train_file)  # pd.read_csv('train.csv')

        errors = []
        levels = [0]
        print(aggregator)
        print(features)
        print('Running colsTest')
        print(pred_df.columns,features.keys())
        print(set(pred_df.columns),set(features.keys()))
        if not colsTest(pred_df, train_df,target,features):
            print('Error detected - 001')
            errors.append('001')
            levels.append(2)
        print('Running featuresTest')
        if not featuresTest(features, pred_df):
            print('Error detected - 002')
            errors.append('002')
            levels.append(1)
        print('Running aggregatorTest')
        if not aggregatorTest(pred_df, aggregator):
            print('Error detected - 003')
            errors.append('003')
            levels.append(2)
        print(levels)
        return {
            'Error': bool(errors),
            'Errors': errors,
            'Level': max(levels)
        }
    except Exception as e:
        print(e)
        return {
            'Error': True,
            'Errors': [str(e)],
            'Level': 3
        }
