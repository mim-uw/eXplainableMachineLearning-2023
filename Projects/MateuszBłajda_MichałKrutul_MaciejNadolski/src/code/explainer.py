import numpy as np
import pandas as pd
import warnings

class Explainer:
    def __init__(self, model, data, predict_function=None):
        self.model = model

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, np.ndarray):
            warnings.warn("`data` is a numpy.ndarray -> coercing to pandas.DataFrame.")
            self.data = pd.DataFrame(data)
        else:
            raise TypeError(
                "`data` is a " + str(type(data)) +
                ", and it should be a pandas.DataFrame.")            

        if predict_function:
            self.predict_function = predict_function
        else:
            # scikit-learn extraction
            if hasattr(model, '_estimator_type'):
                if model._estimator_type == 'classifier':
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif model._estimator_type == 'regressor':
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError("Unknown estimator type: " + str(model._estimator_type) + ".")
            # tensorflow extraction
            elif str(type(model)).startswith("<class 'keras.engine."):
                if model.output_shape[1] == 1:
                    self.predict_function = lambda m, d: m.predict(np.array(d), verbose=False).reshape(-1, )
                elif model.output_shape[1] == 2:
                    self.predict_function = lambda m, d: m.predict(np.array(d), verbose=False)[:, 1]
                else:
                    warnings.warn("`model` predict output has shape greater than 2, predicting column 1.")   
            # default extraction
            else:
                if hasattr(model, 'predict_proba'):
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif hasattr(model, 'predict'):
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError(
                        "`predict_function` can't be extracted from the model. \n" + 
                        "Pass `predict_function` to the Explainer, e.g. " + 
                        "lambda m, d: m.predict(d), which returns a (1d) numpy.ndarray."
                    )
            
        try:
            pred = self.predict(data.values)
        except:
            raise ValueError("`predict_function(model, data)` returns an error.")
        if not isinstance(pred, np.ndarray):
            raise TypeError(
                "`predict_function(model, data)` returns an object of type " +
                str(type(pred)) +
                ", and it must return a (1d) numpy.ndarray."
            )
        if len(pred.shape) != 1:
            raise ValueError(
                "`predict_function(model, data` returns an object of shape " +
                str(pred.shape) +
                ", and it must return a (1d) numpy.ndarray."
            )

    def predict(self, data):
        return self.predict_function(self.model, data)

    # ************* pd *************** #

    def pd(self, X, idv, grid):
        """
        numpy implementation of pd calculation for 1 variable 
        
        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile
        
        returns:
        y - np.ndarray (1d), vector of pd profile values
        """
        
        grid_points = len(grid)
        # take grid_points of each observation in X
        X_long = np.repeat(X, grid_points, axis=0)
        # take grid for each observation
        grid_long = np.tile(grid.reshape((-1, 1)), (X.shape[0], 1))
        # merge X and grid in long format
        X_long[:, [idv]] = grid_long
        # calculate ceteris paribus
        y_long = self.predict(X_long)
        # calculate partial dependence
        y = y_long.reshape(X.shape[0], grid_points).mean(axis=0)

        return y

    def pd_pop(self, X_pop, idv, grid):
        """
        vectorized (whole population) pd calculation for 1 variable
        """
        grid_points = len(grid)
        # take grid_points of each observation in X
        X_pop_long = np.repeat(X_pop, grid_points, axis=1)
        # take grid for each observation
        grid_pop_long = np.tile(grid.reshape((-1, 1)), (X_pop.shape[0], X_pop.shape[1], 1))
        # merge X and grid in long format
        X_pop_long[:, :, [idv]] = grid_pop_long
        # calculate ceteris paribus
        y_pop_long = self.predict(
            X_pop_long.reshape(X_pop_long.shape[0] * X_pop_long.shape[1], X_pop_long.shape[2])
        ).reshape((X_pop_long.shape[0], X_pop.shape[1], grid_points))
        # calculate partial dependence
        y = y_pop_long.mean(axis=1)

        return y
    
    def ale(self, X, idv, grid):
        """
        numpy implementation of ale calculation for 1 variable 
        
        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile
        
        returns:
        y - np.ndarray (1d), vector of pd profile values
        """
        X_sorted = X[X[:, idv].argsort()]
        
        grid_points = len(grid)

        z_idx = np.searchsorted(X_sorted[:, idv], grid)
        N = z_idx[1:] - z_idx[:-1]
        
        g = 0
        y = np.zeros_like(grid)
        for k in range(1, grid_points):
            segment = X_sorted[z_idx[k-1]:z_idx[k], :]
            X_zk = segment.copy()
            X_zk[:, idv] = grid[k]
            X_zkm1 = segment.copy()
            X_zkm1[:, idv] = grid[k-1]
            sum = np.sum(self.model(X_zk) - self.model(X_zkm1))
            g = g if N[k-1] == 0 else g + sum / N[k-1]
            y[k] = g
        
        return y

  