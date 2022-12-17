import numpy as np


def check_early_stopping(df, epsilon, stop_iter=10):
    if len(df['loss']) >= stop_iter:
        relative_change = np.abs(df['loss'][-stop_iter] - df['loss'][-1] / np.mean(df['loss'][-stop_iter:]))
        return relative_change < epsilon
    else:
        return False
        

class AdamOptimizer:
    def __init__(self, m=0, v=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

    def calculate_step(self, gradient):
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        mh = self.m / (1 - self.beta1)
        vh = self.v / (1 - self.beta2)
        step = mh / (np.sqrt(vh) + self.epsilon)

        return step
