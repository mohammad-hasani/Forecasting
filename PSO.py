import pyswarms as ps
from Tools2 import *
from keras.losses import MeanSquaredError


class PSO(object):
    def __init__(self, model, dimensions, X_train, y_train, file_name, init_weights=None,
    n_iteration=20, n_particles=20):
        self.dimensions = dimensions
        self.init_weights = init_weights
        self.n_particles = n_particles
        self.n_iteration = n_iteration
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.file_name = file_name

    def PSO(self):
        init_pos = None
        if self.init_weights is not None:
            init_pos = list()
            init_pos.append(self.init_weights)
            for i in range(self.n_particles - 1):
                init_pos.append(self.init_weights * np.random.random())

            init_pos = np.array(init_pos)

        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # Call instance of PSO
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles,
         dimensions=self.dimensions, options=options, init_pos=init_pos)
        # Perform optimization
        cost, pos = optimizer.optimize(self.f, iters=self.n_iteration)

        return cost, pos

    def loss_function(self, Wb_flatten):
        # Assigning Weights
        Wb = get_weights_and_biases(self.model)
        Wb = unflatten_weights_and_biases(Wb, Wb_flatten)
        self.model = put_weights_and_biases(self.model, Wb)

        # Computing MSE
        y_pred = self.model.predict(self.X_train)
        mse = MeanSquaredError()
        loss = mse(self.y_train, y_pred).numpy()

        return loss


    def f(self, x):
        n_particles = x.shape[0]
        j = [self.loss_function(x[i]) for i in range(n_particles)]

        # with open(f'./Results/{self.file_name} list', 'a') as f:
        #     f.write(str(j))
        #     f.write('\n')

        return np.array(j)
