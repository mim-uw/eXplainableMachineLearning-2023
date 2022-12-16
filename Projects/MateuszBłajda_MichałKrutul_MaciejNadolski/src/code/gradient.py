import numpy as np
import pandas as pd
import tqdm

from . import algorithm     
from . import loss
from . import utils

try:
    import tensorflow as tf
except:
    import warnings
    warnings.warn("`import tensorflow as tf` returns an error: gradient.py won't work.")

class GradientAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        **kwargs
    ):
        super().__init__(
            explainer=explainer,
            variable=variable,
            constant=constant,
            n_grid_points=n_grid_points
        )

        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            learning_rate=1e-2,
            optimizer=utils.AdamOptimizer()
        )
        
        for k, v in kwargs.items():
            params[k] = v

        self.params = params


    def fool(
        self,
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        aim=False,
        center=None,
        method="pd",
    ):
        self._aim = aim
        self._center = not aim if center is None else center
        if aim is False:
            super().fool(grid=grid, random_state=random_state, method=method)

        # init algorithm
        self._initialize()
        if method == "pd":
            self.result_explanation['changed'] = self.explainer.pd(
                self._X_changed, 
                self._idv, 
                self.result_explanation['grid']
            )
        elif method == "ale":
            self.result_explanation['changed'] = self.explainer.ale(
                self._X_changed, 
                self._idv, 
                self.result_explanation['grid']
            )
        self.append_losses(i=0)
        if save_iter:
            self.append_explanations(i=0)

        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for i in pbar:
            if method == "pd":
                self.result_explanation['changed'] = self.explainer.pd(
                    self._X_changed, 
                    self._idv, 
                    self.result_explanation['grid']
                )
            elif method == "ale":
                self.result_explanation['changed'] = self.explainer.ale(
                    self._X_changed, 
                    self._idv, 
                    self.result_explanation['grid']
                )

            gradient = self.calculate_gradient(self._X_changed, method=method)
            step = self.params['optimizer'].calculate_step(gradient)
            self._X_changed -= self.params['learning_rate'] * step

            self.append_losses(i=i)
            if save_iter:
                self.append_explanations(i=i)
            pbar.set_description("Iter: %s || Loss: %s" % (i, self.iter_losses['loss'][-1]))
            if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                break
        
        if method == "pd":
            self.result_explanation['changed'] = self.explainer.pd(
                X=self._X_changed,
                idv=self._idv,
                grid=self.result_explanation['grid']
            )
        elif method == "ale":
            self.result_explanation['changed'] = self.explainer.ale(
                X=self._X_changed,
                idv=self._idv,
                grid=self.result_explanation['grid']
            )
        print("RES: ", self.result_explanation['changed'])
        _data_changed = pd.DataFrame(self._X_changed, columns=self.explainer.data.columns)
        self.result_data = pd.concat((self.explainer.data, _data_changed))\
            .reset_index(drop=True)\
            .rename(index={'0': 'original', '1': 'changed'})\
            .assign(dataset=pd.Series(['original', 'changed'])\
                            .repeat(self._n).reset_index(drop=True))


    def fool_aim(
        self,
        target="auto",
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        method="pd",
    ):
        super().fool_aim(
            target=target,
            grid=grid,
            random_state=random_state,
            method=method,
        )
        self.fool(
            grid=None,
            max_iter=max_iter, 
            random_state=random_state, 
            save_iter=save_iter, 
            verbose=verbose, 
            aim=True,
            method=method,
        )


    #:# inside

    def calculate_pdp(self, data_tensor):
        data_long = tf.repeat(data_tensor, self._n_grid_points, axis=0)
        grid_long = tf.tile(tf.convert_to_tensor(self.result_explanation['grid']), tf.convert_to_tensor([self._n]))
        data_long = GradientAlgorithm.assign(data_long, (slice(None, None), self._idv), grid_long.reshape(-1, 1))
        return tf.reshape(self.explainer.model(data_long), (self._n, self._n_grid_points)).mean(axis=0)

    def calculate_ale(self, data_tensor):
        data_sorted_ids = tf.argsort(data_tensor[:, self._idv])
        data_sorted = tf.gather(data_tensor, data_sorted_ids, axis=0)

        z_idx = tf.searchsorted(data_sorted[:, self._idv], self.result_explanation['grid'])
        N = z_idx[1:] - z_idx[:-1]

        grid_points = len(self.result_explanation['grid'])

        y = tf.zeros(grid_points)

        for k in range(1, grid_points):

            if N[k-1] == 0:
                continue

            X_zk = tf.identity(data_sorted[z_idx[k-1] : z_idx[k], :])
            X_zkm1 = tf.identity(data_sorted[z_idx[k-1] : z_idx[k], :])

            X_zk = GradientAlgorithm.assign(
                X_zk,
                (slice(None, None), self._idv),
                self.result_explanation['grid'][k]
            )
            X_zkm1 = GradientAlgorithm.assign(
                X_zkm1,
                (slice(None, None), self._idv),
                self.result_explanation['grid'][k - 1]
            )
            scaling_factor = 1/N[k-1]

            partial_res = tf.math.reduce_sum((self.explainer.model(X_zk) - self.explainer.model(X_zkm1))) * scaling_factor
            y = GradientAlgorithm.assign(y, (k), partial_res)

        y = tf.math.cumsum(y)
        return y
       
        
    def calculate_loss(self, result):
        if self._aim:
            return tf.keras.losses.mean_squared_error(self.result_explanation['target'], result)
        else:
            assert False, "Not implemented"
    
    def calculate_gradient(self, data, method="pd"):
        input = tf.convert_to_tensor(data)
        with tf.GradientTape() as t:
            t.watch(input)
            if method == "pd":
                explanation = self.calculate_pdp(input)
            elif method == "ale":
                explanation = self.calculate_ale(input)
            loss = self.calculate_loss(explanation)
            gradient = t.gradient(loss, input)
            if isinstance(gradient, tf.IndexedSlices):
                gradient = tf.convert_to_tensor(gradient)
        
        return gradient.numpy()

    def assign(tensor, slc, values):
        '''
            Tensorflow can't do
                tensor[slc] = values
            this is a workaround
        '''

        mask = np.zeros_like(tensor.numpy())
        mask[slc] = 1
        mask = tf.convert_to_tensor(mask)
        return (1 - mask) * tensor + mask * values

    def assign2(tensor, slc, values):
        '''
            Tensorflow can't do
                tensor[slc] = values
            this is a workaround
        '''
        var = tf.Variable(tensor, trainable=True)
        with tf.control_dependencies([var[slc].assign(values)]):
                new_tensor = tf.identity(var)
        return new_tensor

    def _initialize(self):
        _X_std = self._X.std(axis=0) * 1/9
        _X_std[self._idv] = 0
        if self._idc is not None:
            for c in self._idc:
                _X_std[c] = 0        
        _theta = np.random.normal(loc=0, scale=_X_std, size=self._X.shape)
        self._X_changed = self._X + _theta

    #:# helper
              
    def append_losses(self, i=0):
        _loss = loss.loss(
            original=self.result_explanation['target'] if self._aim else self.result_explanation['original'],
            changed=self.result_explanation['changed'],
            aim=self._aim,
            center=self._center
        )
        self.iter_losses['iter'].append(i)
        self.iter_losses['loss'].append(_loss)

    def append_explanations(self, i=0):
        self.iter_explanations[i] = self.result_explanation['changed']