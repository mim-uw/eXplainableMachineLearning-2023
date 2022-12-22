# ---------------------------------
# example: heart / NN / variable
# > python heart-gradient.py
# ---------------------------------


import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior(
    prefer_float32=False
)
import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--variable', default="age", type=str, help='variable')
parser.add_argument('--strategy', default="target", type=str, help='strategy type')
parser.add_argument('--iter', default=50, type=int, help='max iterations')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--method', default="pd", type=str, help='method: pd/ale')
args = parser.parse_args()

VARIABLE = args.variable

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(args.seed)

import code
import numpy as np
np.random.seed(args.seed)
import pandas as pd

df = pd.read_csv("data/heart.csv")
X, y = df.drop("target", axis=1), df.target.values

normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(X)

model = tf.keras.Sequential()
model.add(normalizer)
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc', 'AUC'])
model.fit(X, y, batch_size=32, epochs=50, verbose=1)
explainer = code.Explainer(model, X)

VARIABLES = {
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal'
}
CONSTANT = [
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
]

alg = code.GradientAlgorithm(
    explainer, 
    variable=VARIABLE,
    constant=CONSTANT,
    learning_rate=0.1
)

if args.strategy == "target":
    alg.fool_aim(max_iter=args.iter, random_state=args.seed, method=args.method)
else:
    alg.fool(max_iter=args.iter, random_state=args.seed, method=args.method)

alg.plot_losses()
title = "Partial Dependence" if args.method == "pd" else "Accumulated Local Effects"
alg.plot_explanation(method=args.method, title=title)
alg.plot_data(constant=False)