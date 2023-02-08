import pandas as pd
import shap
import dalex as dx
import xgboost as xgb
from sklearn.datasets import load_diabetes, make_friedman1, load_breast_cancer,make_classification
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from exp_utils import DataProcessor, Experiment

VERBOSE = False


def experiment_1(no_tests=100, save_path='./results/synthetic_XGBoost2.parquet', model_metric='accuracy'):
    X, y = make_classification(n_samples=5000)
    data_processor = DataProcessor(X=X, y=y,test_size=4 ** 5)

    experiment_settings = {

        'data_processor': data_processor,
        'model_class': xgb.XGBClassifier,
        'model_params': {'max_depth': 4, 'subsample': 0.9 ** 3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9,
                         'colsample_bynode': 0.9, 'alpha': 0.1},
        'shap_class': shap.explainers.Tree,
        'shap_params': {'model_output': "raw"},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            test_size=4 ** 5, model_metric=model_metric)

    return result


def experiment_2(no_tests=100, save_path='./results/exp_synthetic_knn.parquet', model_metric='accuracy'):
    X, y = make_classification(n_samples=5000)
    data_processor = DataProcessor(X=X, y=y,test_size=4 ** 5)

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': KNeighborsClassifier,
        'model_params': {},
        'shap_class': shap.Explainer,
        'is_tree': False,
        'shap_params': {},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            test_size=4 ** 5, model_metric=model_metric)

    return result


experiments = [
    experiment_1,
    experiment_2
    

]

for exp in experiments:
    print('final:',exp())

