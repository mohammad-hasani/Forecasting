import numpy as np
from Tools import *
from Model import Model

from keras.losses import MeanSquaredError


class GA(object):
    def __init__(self, model, X_train, y_train, n_population=200, n_iteration=20):
        self.n_population = n_population
        self.population = list()
        self.mates = None
        self.mutation_rate = .1
        self.n_iteration = n_iteration
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def __call__(self):
        for i in range(self.n_iteration):
            self.parent_selection()
            self.crossover()
            self.mutation()
            self.fitness()
            print(i)
        return self.population[0][1], self.population[0][0]


    def set_population(self, data):
        # init_pos = list()
        # init_pos.append(data)
        # for i in range(self.n_population - 1):
        #     init_pos.append(self.init_weights * np.random.random())
        # init_pos = np.array(init_pos)

        data = np.random.random((self.n_population, len(data)))

        for i in range(len(data)):
            self.population.append([data[i], 0])

    def parent_selection(self):
        np.random.shuffle(self.population)
        self.mates = np.reshape(self.population, (int(self.n_population / 2) ,2, -1))

    def crossover(self):
        alpha = .3
        for mate in self.mates:
            parent_1 = mate[0][0]
            parent_2 = mate[1][0]

            child_1 = [alpha * parent_1 + (1 - alpha) * parent_2, 0]
            child_2 = [alpha * parent_2 + (1 - alpha) * parent_1, 0]

            flips = np.random.random(len(parent_1))
            flips = flips > .5

            child_3 = [np.zeros(len(parent_1)), 0]
            child_4 = [np.zeros(len(parent_1)), 0]
            for index, state in enumerate(flips):
                if state:
                    child_3[0][index] = parent_1[index]
                    child_4[0][index] = parent_2[index]
                else:
                    child_3[0][index] = parent_2[index]
                    child_4[0][index] = parent_1[index]

            self.population.extend([child_1, child_2, child_3, child_4])

    def mutation(self):
        for i in range(self.n_population):
            if np.random.random() < self.mutation_rate:
                index = np.random.randint(0, self.population[0][0].shape[0])
                self.population[i][0][index] = np.random.random()

    def fitness(self):
        for index, Wb_flatten in enumerate(self.population):
            # Assigning Weights
            Wb = get_weights_and_biases(self.model)
            Wb = unflatten_weights_and_biases(Wb, Wb_flatten[0])
            self.model = put_weights_and_biases(self.model, Wb)

            # Computing MSE
            y_pred = self.model.predict(self.X_train)
            mse = MeanSquaredError()
            loss = mse(self.y_train, y_pred).numpy()

            self.population[index][1] = loss
        self.population.sort(key=lambda x: x[1])
        self.population = self.population[:self.n_population]
