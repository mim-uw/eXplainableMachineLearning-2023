import pandas as pd
import shap
import dalex as dx
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from exp_utils import DataProcessor, Experiment

VERBOSE = False



def experiment_1(no_tests=5, save_path='./results/exp_big_XGBoost.parquet', model_metric='accuracy'):

    X = pd.read_csv('data/covtype.csv')
    data_processor = DataProcessor(X=X.iloc[:,:10], y=X.iloc[:,-1]-1,test_size=4 ** 9)
    

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': xgb.XGBClassifier,
        'model_params': {'max_depth': 3, 'subsample': 0.9 ** 3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9,
                         'colsample_bynode': 0.9, 'alpha': 0.1},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }
    
    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            model_metric=model_metric)

    return result


def experiment_2(no_tests=5, save_path='./results/exp_big_KNN.parquet', model_metric='accuracy'):
    X = pd.read_csv('./data/covtype.csv')
    data_processor = DataProcessor(X=X.iloc[:,:10], y=X.iloc[:,-1]-1,test_size=4 ** 9)

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': KNeighborsClassifier,
        'model_params': {},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            model_metric=model_metric)

    return result

def experiment_3(no_tests=5, save_path='./results/exp_big_shap_XGBoost.parquet', model_metric='accuracy'):

    X = pd.read_csv('data/covtype.csv')
    data_processor = DataProcessor(X=X.iloc[:,:10], y=X.iloc[:,-1]-1)
    

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': xgb.XGBClassifier,
        'model_params': {'max_depth': 3, 'subsample': 0.9 ** 3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9,
                         'colsample_bynode': 0.9, 'alpha': 0.1},
        'shap_class': shap.explainers.Tree,
        'shap_params': {'model_output': "raw"},
    }
    
    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            test_size=4 ** 9, model_metric=model_metric)

    return result

def experiment_4(no_tests=5, save_path='./results/exp_big_shap_knn.parquet', model_metric='accuracy'):

    X = pd.read_csv('data/covtype.csv')
    data_processor = DataProcessor(X=X.iloc[:,:10], y=X.iloc[:,-1]-1)
    

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': xgb.XGBClassifier,
        'model_params': {'max_depth': 3, 'subsample': 0.9 ** 3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9,
                         'colsample_bynode': 0.9, 'alpha': 0.1},
        'shap_class': shap.Explainer,
        'is_tree': False,
        'shap_params': {},
    }
    
    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            test_size=4 ** 9, model_metric=model_metric)

    return result



experiments = [
    #experiment_1,
    experiment_2
    

]

for exp in experiments:
    print(exp())
