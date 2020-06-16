import os
import random
import time
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def act_fun(x, ActvFun='sigmoid'):
    # we are using np.exp because in feature we need to calculate sigmoid of 2D matrix
    # the sigmoid have -x
    if ActvFun.lower() == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif ActvFun.lower() == 'tanh':
        return np.tanh(x)
    elif ActvFun.lower() == 'relu':
        return np.maximum(0, x) * 1


def del_act_fun(a, ActvFun='sigmoid'):
    if ActvFun.lower() == 'sigmoid':
        return np.multiply(a, 1 - a)

    elif ActvFun.lower() == 'tanh':
        return 1 - np.multiply(a, a)

    elif ActvFun.lower() == 'relu':
        a[a <= 0] = 0
        a[a > 0] = 1
        return a


def draw_errors(error, epoch):
    figure = matplotlib.pyplot.figure()
    plot = figure.add_subplot(111)
    x = range(0, len(error), 1)
    plot.plot(error)
    name = './plots/epoch_NO' + str(epoch) + '.png'
    figure.savefig(name)
    plt.close()


class ANN:
    def __init__(self, size, Name=''):
        try:
            os.mkdir('./networks/')
        except:
            pass
        try:
            os.mkdir('./plots/')
        except:
            pass

        self.inp = np.matrix(np.zeros((size[0]))).transpose()
        self.lr = .01
        self.size = size
        self.activFun = ['relu'] * (len(size) - 1)
        self.layers = []
        self.name = Name
        if self.name != '':
            try:
                self.load()
            except:
                pass

    def create(self):
        self.layers = [LayerInput(self.size[0])]
        for i in range(1, len(self.size)):
            self.layers.append(
                LayerConnected(prev_layer=self.layers[-1],
                               No_of_n=self.size[i],
                               learningRate=self.lr,
                               activationFunction=self.activFun[i - 1]
                               ))
        self.error_o = np.zeros((self.size[-1]))
        self.summary()

    def save(self):
        # w = np.asarray(self.weights)
        # b = np.asarray(self.bias)
        # np.save('./networks/' + self.name + "ws", w)
        # np.save('./networks/' + self.name + "bs", b)
        pass

    def load(self, name=''):
        if name == '':
            name = self.name
        else:
            self.name = name
        # self.weights = np.load('./networks/' + name + "ws.npy", allow_pickle=True)
        # self.bias = np.load('./networks/' + name + "bs.npy", allow_pickle=True)
        pass

    def feedforward(self, inp):

        self.layers[0].activation = np.matrix(inp)
        self.inp = np.array(inp)
        for l in self.layers[1:]:
            l.forwardProp()
            # w = l.weights
            # b = l.bias
            # output = act_fun(w * output + b, l.activFun)
            # l.activation = output

        return self.layers[-1].activation

    def train(self, inputs, targets, n_epoch=20, batchSize=10, shuffle=True):
        error_allEPOCH = []
        for epoch in range(n_epoch):
            errors = []
            start = time.time()
            TRMP = inputs
            errors = []
            if shuffle:
                data = []
                for i in range(len(inputs)):
                    data.append([inputs[i], targets[i]])
                random.shuffle(data)
                inputs = []
                targets = []
                for d in data:
                    inputs.append(d[0])
                    targets.append(d[1])

            n = len(inputs) // batchSize
            for i in range(n):
                # Ann_full.resetpera()
                input_ = inputs[i * batchSize:(i + 1) * batchSize]
                target_ = targets[i * batchSize:(i + 1) * batchSize]
                self.backprop(input_, target_)
                errors.append(np.sum(np.square(self.error_o)))
                # for data, target in zip(input_, target_):
                #     self.backprop(data, target)
                #     sum_e = 0
                #     for e in self.error_o[0]:
                #         sum_e += np.square(e)
                #     errors.append(sum_e)
                self.update_WB()
            print('>epoch=%d, avg_error=%.3f' % (epoch, sum(errors) / len(errors)), ' time: ',
                  timedelta(seconds=time.time() - start))
            draw_errors(np.asarray(errors), epoch)
            for e in errors:
                error_allEPOCH.append(np.asarray(e))
        draw_errors(error_allEPOCH, 'full')
        return error_allEPOCH

    def backprop(self, inp, tar):

        pred = self.feedforward(inp)
        tar = np.array(tar)
        self.error_o = tar - pred
        self.layers[-1].error = self.error_o
        for l in reversed(self.layers[1:]):
            l.calculate_delta()
        '''
        Now you need to calculate the change in error with respect to w_y and w_x
        in other words del(error)/del w_y and del(error) / del w_x 
        in this del is partial differentiation

        applying chain rule
        del(error) / del W  = (activation)' * <coefficient of W>  ##  " ' " -> derivative 
        '''
        # errors = self.error_o
        #
        # for i in reversed(range(len(self.layers))):
        #     l = self.layers[i]
        #     w = l.weights.transpose()
        #     if i > 0:
        #         a = self.layers[i - 1].activation
        #     else:
        #         a = np.asarray(inp)
        #     l.error = errors
        #     errors = w * errors
        #     errors = np.multiply(errors, del_act_fun(a, l.activFun))
        # activ = inp
        # for l in self.layers:
        #     l.calculate_delta(activ)
        #     activ = l.activation.transpose()

    def update_WB(self):
        for l in self.layers[1:]:
            l.update_WB()

    def update_lr(self, lr):
        for i in range(1, len(self.layers)):
            self.layers[i].lr = lr
    def resetpera(self):
        for l in self.layers[1:]:
            l.restPera()
    def summary(self):
        totalParameter = 0
        totalTrainableParameter = 0
        totalNonTrainableParameter = 0
        l = self.layers[0]
        print('\t\tNo of Inputs :', l.No_of_n, end='\n')
        for l in self.layers[1:]:
            print("input: ", l.No_of_inp, '\t\tPerceptrons:', l.No_of_n, '\t\tActivalton Fun: ', l.activFun,
                  '\t\tIs Trainable', l.training, end='\n')
            Parameter = l.weights.size + l.bias.size
            totalParameter += Parameter
            if l.training:
                totalTrainableParameter += Parameter
            else:
                totalNonTrainableParameter += Parameter
        print('Total Parameter: ', totalParameter)
        print('Total Trainable Parameter: ', totalTrainableParameter)
        print('Total Non Trainable Parameter: ', totalNonTrainableParameter)


class LayerConnected:
    def __init__(self, prev_layer, No_of_n, learningRate=.000001, activationFunction='relu'):
        self.No_of_n = No_of_n
        self.No_of_inp = prev_layer.No_of_n
        self.prev_layer = prev_layer

        self.weights = np.matrix(np.random.random((self.No_of_inp, self.No_of_n)) * 2 - 1)
        self.bias = np.matrix(np.zeros(self.No_of_n))
        self.activation = np.zeros(self.No_of_n)
        self.error = np.zeros_like(self.activation)
        self.del_weights = np.zeros_like(self.weights)
        self.del_bias = np.zeros_like(self.bias)

        self.lr = learningRate
        self.activFun = activationFunction
        self.training = True

        self.h_w = self.del_weights * 0
        self.h_b = self.del_bias * 0

        self.G = np.zeros_like(self.weights)
        # self.G_b = np.zeros_like(self.bias)

        self.alpha = .9

    def forwardProp(self):
        # X = self.prev_layer.activation
        # W = self.weights
        # self.activation = X.dot(W) + B
        self.activation = self.prev_layer.activation.dot(self.weights) + self.bias

    def backProp(self, del_Z):
        # dX = dZ.dot(W.T)
        # dW = X.T.dot(dZ)
        dX = del_Z.dot(self.weights.T)
        dW = self.prev_layer.activation.T.dot(del_Z)

    def calculate_delta(self):
        X = self.prev_layer.activation
        self.error = np.multiply(self.error, del_act_fun(self.activation, self.activFun))
        if self.training:
            self.del_weights += X.T.dot(self.error)
            self.del_bias += sum(self.error)
        self.prev_layer.error = self.error.dot(self.weights.T)

    def restPera(self):
        self.h_w = self.del_weights * 0
        self.h_b = self.del_bias * 0

        self.G = np.zeros_like(self.weights)

    def update_WB(self):
        self.alpha = .001
        nu = np.zeros_like(self.weights)

        # self.G = self.G * self.alpha
        # self.G = self.G * self.alpha + (1 - self.alpha) * np.square(self.del_weights)
        # eta = self.lr / (np.square(self.G )+ 1e-8)  # lr with RMSprop
        # momentum
        # self.h_w = self.alpha * self.h_w + np.multiply(eta, self.del_weights)
        self.h_w = self.alpha * self.h_w + np.multiply(self.lr, self.del_weights)
        self.h_b = self.alpha * self.h_b + self.lr * self.del_bias

        self.weights += self.h_w
        self.bias += self.h_b

        self.del_bias = np.zeros_like(self.del_bias)
        self.del_weights = np.zeros_like(self.del_weights)


class LayerInput:
    def __init__(self, No_of_inp):
        self.No_of_n = No_of_inp
        self.activation = np.zeros(self.No_of_n)
        self.error = None


if __name__ == '__main__':
    # autoencoder

    input_size = 10
    Ann_full = ANN((input_size, input_size), Name='NewTry_1')
    for i in range(len(Ann_full.activFun)):
        Ann_full.activFun[i] = 'sigmoid'
    Ann_full.create()
    # Ann_full.load()
    data = []
    for i in range(1000):
        inp = np.random.randint(0, 2, size=input_size)
        data.append(inp)
    print('data created')
    error = Ann_full.train(data, data, 25, 1)
    draw_errors(error, 500)

    Ann_full.save()
    data = []
    for i in range(10000):
        inp = np.random.randint(0, 2, size=(input_size,))
        data.append(inp)
    print('data created')
    error = Ann_full.train(data, data, 15, 25)
    draw_errors(error, 501)

    Ann_full.save()
