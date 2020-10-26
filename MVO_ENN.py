from mealpy.physics_based.MVO import BaseMVO
import numpy as np
from Tools2 import *
from keras.losses import MeanSquaredError


lb = [-100]
ub = [100]


class MVO(object):
    def __init__(self, model, dimensions, X_train, y_train, file_name=None, init_weights=None,
    n_iteration=20, n_particles=20):
        self.dimensions = dimensions
        self.init_weights = init_weights
        self.n_particles = n_particles
        self.n_iteration = n_iteration
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.file_name = file_name

    def MVO(self):
        mvo = BaseMVO(obj_func=self.f, lb=lb, ub=ub, problem_size=self.dimensions, epoch=30)
        best_pos, best_fit, list_loss = mvo.train()

        return best_fit, best_pos

    def f(self, Wb_flatten):
        # Assigning Weights
        self.model = set_elman_weights(self.model, Wb_flatten)

        # Computing MSE
        y_pred = self.model.sim(self.X_train)
        mse = MeanSquaredError()
        loss = mse(self.y_train, y_pred).numpy()

        return loss
