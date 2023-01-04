# ---------------------------------
# example: heart / SVM / age & sex
# > python heart-genetic.py
# ---------------------------------


import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--variable', default="age", type=str, help='variable: age / sex')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()
VARIABLE = args.variable

import code
import numpy as np
np.random.seed(args.seed)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/heart.csv")
rename_dict = {"thalach": "maximum heart rate",
               "trestbps": "resting blood pressure",
               "chol": "serum cholesterol"}
VARIABLES = {
    'age', 'sex', 'cp', rename_dict['trestbps'], rename_dict['chol'],
    'fbs', 'restecg', rename_dict['thalach'], 'exang', 'oldpeak',
    'slope',  'ca', 'thal'
}
df.rename(columns=rename_dict, inplace=True)

X, y = df.drop("target", axis=1), df.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=args.seed)


# ------------- age ---------------
if VARIABLE == "age":
    clf = Pipeline([("scaler", StandardScaler()),
                    ("model", SVC(C=3, probability=True, random_state=args.seed))])
    clf.fit(X_train, y_train)

    explainer = code.Explainer(clf, X_test)

    VARIABLES_TO_CHANGE = [rename_dict['thalach'], rename_dict['trestbps']]
    CONSTANT = VARIABLES.difference(VARIABLES_TO_CHANGE).difference([VARIABLE])

    alg = code.GeneticAlgorithm(
        explainer, 
        variable=VARIABLE,
        constant=CONSTANT
    )

    target_up = lambda x: 0.15 + (x-30)/50 / (3/2)
    alg.fool_aim(target=target_up, max_iter=250, random_state=args.seed)
    alg.plot_losses()
    alg.plot_explanation(target=False)
    alg.plot_data(height=2,constant=False)

    alg = code.GeneticAlgorithm(
        explainer, 
        variable=VARIABLE,
        constant=CONSTANT
    )
    
    target_down = lambda x: 1 - (0.15 + (x-30)/50 / (3/2))
    alg.fool_aim(target=target_down, max_iter=250, random_state=args.seed)
    alg.plot_losses()
    alg.plot_explanation(target=False)
    alg.plot_data(height=2, constant=False)


# ------------- sex ---------------
if VARIABLE == "sex":
    clf = Pipeline([("scaler", StandardScaler()),
                    ("model", SVC(C=5, probability=True, random_state=args.seed))])
    clf.fit(X_train, y_train)

    explainer = code.Explainer(clf, X_train)

    VARIABLES_TO_CHANGE = [rename_dict['thalach'], rename_dict['trestbps'], rename_dict['chol']]
    CONSTANT = VARIABLES.difference(VARIABLES_TO_CHANGE).difference([VARIABLE])

    alg = code.GeneticAlgorithm(
        explainer, 
        variable=VARIABLE,
        constant=CONSTANT,
        n_grid_points=2,
        std_ratio=1/12
    )

    target_suspected = np.array([1, 0.3])
    alg.fool_aim(target=target_suspected, max_iter=1500, random_state=args.seed)
    alg.plot_losses()
    alg.plot_explanation(categorical=True, target=False)
    alg.plot_data(height=2, constant=False)

    alg = code.GeneticAlgorithm(
        explainer, 
        variable=VARIABLE,
        constant=CONSTANT,
        n_grid_points=2,
        std_ratio=1/12
    )

    target_concealed = np.array([0.3, 0.6])
    alg.fool_aim(target=target_concealed, max_iter=1500, random_state=args.seed)
    alg.plot_losses()
    alg.plot_explanation(categorical=True, target=False)
    alg.plot_data(height=2, constant=False)