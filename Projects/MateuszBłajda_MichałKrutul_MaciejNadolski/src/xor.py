# ---------------------------------
# test the algorithm implementation
# on a multidimensional XOR problem
# > python xor.py
# ---------------------------------


import tensorflow as tf
import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--algorithm', default="gradient", type=str, help='algorithm type')
parser.add_argument('--strategy', default="target", type=str, help='strategy type')
parser.add_argument('--center', default=None, type=str, help='center in strategy r')
parser.add_argument('--n', default=320, type=int, help='number of observations')
parser.add_argument('--size', default=16, type=int, help='number of neurons in layers')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

N = args.n
SIZE = args.size
tf.random.set_seed(args.seed)

import code
import numpy as np
import pandas as pd

np.random.seed(args.seed)
x1 = np.random.normal(size=N)
x2 = np.random.normal(size=N)
x3 = np.random.normal(size=N)
y = 1 * (x1 * x2 * x3 > 0)
X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(X)

model = tf.keras.Sequential()
model.add(normalizer)
model.add(tf.keras.layers.Dense(SIZE, activation="relu"))
model.add(tf.keras.layers.Dense(SIZE, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['acc', 'AUC'])
model.fit(X, y, batch_size=int(N/10), epochs=300, verbose=0)

explainer = code.Explainer(model, X)

if args.algorithm == "gradient":
    alg = code.GradientAlgorithm(explainer, variable="x1")
else:
    alg = code.GeneticAlgorithm(explainer, variable="x1", std_ratio=1/6)

if args.strategy == "target":
    alg.fool_aim(random_state=args.seed)
else:
    alg.fool(center=args.center, random_state=args.seed)

alg.plot_losses()
alg.plot_explanation()
alg.plot_data(constant=False)