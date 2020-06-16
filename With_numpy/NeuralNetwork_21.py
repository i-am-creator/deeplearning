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
        self.activFun = ['relu']*(len(size)-1)
        self.layers = []
        self.name = Name
        if self.name != '':
            try:
                self.load()
            except:
                pass

    def create(self):
        for i in range(1, len(self.size)):
            self.layers.append(Layer(self.size[i], self.size[i - 1], learningRate=self.lr, activationFunction=self.activFun[i-1]))
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

        output = np.matrix(inp).transpose()
        self.inp = output
        for l in self.layers:
            w = l.weights
            b = l.bias
            output = act_fun(w * output + b, l.activFun)
            l.activation = output

        return output

    def train(self, inputs, targets, n_epoch=20, batchSize=10, shuffle = True):
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
                input_ = inputs[i * batchSize:(i + 1) * batchSize]
                target_ = targets[i * batchSize:(i + 1) * batchSize]
                for data, target in zip(input_, target_):
                    self.backprop(data, target)
                    sum_e = 0
                    for e in self.error_o:
                        sum_e += e ** 2
                    errors.append(sum_e)
                self.update_WB()
            print('>epoch=%d, avg_error=%.3f' % (epoch, sum(errors) / len(errors)), ' time: ',
                  timedelta(seconds=time.time() - start))
            draw_errors(np.asarray(errors).T[0][0], epoch)
            for e in errors:
                error_allEPOCH.append(np.asarray(e)[0][0])
        draw_errors(error_allEPOCH, 'full')
        return error_allEPOCH

    def backprop(self, inp, tar):

        pred = self.feedforward(inp)
        tar = np.matrix(tar).transpose()
        self.error_o = tar - pred
        '''
        Now you need to calculate the change in error with respect to w_y and w_x
        in other words del(error)/del w_y and del(error) / del w_x 
        in this del is partial differentiation

        applying chain rule
        del(error) / del W  = (activation)' * <coefficient of W>  ##  " ' " -> derivative 
        '''
        errors = self.error_o

        for i in reversed(range(len(self.layers))):
            l = self.layers[i]
            w = l.weights.transpose()
            if i >0:
                a = self.layers[i - 1].activation
            else:
                a = np.asarray(inp)
            l.error = errors
            errors = w * errors
            errors = np.multiply(errors, del_act_fun(a, l.activFun))
        activ = inp
        for l in self.layers:
            l.calculate_delta(activ)
            activ = l.activation.transpose()

    def update_WB(self):
        for l in self.layers:
            l.update_WB()
    def update_lr(self, lr):
        for i in range(len(self.layers)):
            self.layers[i].lr = lr

    def summary(self):
        totalParameter = 0
        totalTrainableParameter = 0
        totalNonTrainableParameter = 0

        for l in self.layers:
            print("input: ", l.No_of_inp, '\t\tPerceptrons:', l.No_of_n,'\t\tActivalton Fun: ',  l.activFun, '\t\tIs Trainable', l.training,end= '\n')
            Parameter = l.weights.size + l.bias.size
            totalParameter += Parameter
            if l.training:
                totalTrainableParameter += Parameter
            else:
                totalNonTrainableParameter += Parameter
        print('Total Parameter: ', totalParameter)
        print('Total Trainable Parameter: ', totalTrainableParameter)
        print('Total Non Trainable Parameter: ', totalNonTrainableParameter)

class Layer:
    def __init__(self, No_of_n, No_of_inp, learningRate=.001, activationFunction='relu'):
        self.No_of_n = No_of_n
        self.No_of_inp = No_of_inp
        self.weights = np.matrix(np.random.random((No_of_n, No_of_inp)) * 2 - 1)
        self.bias = np.matrix(np.zeros(No_of_n)).transpose()
        self.activation = np.zeros(No_of_n)
        self.activFun = activationFunction
        self.error = np.zeros(No_of_n)
        self.del_weights = np.zeros((No_of_n, No_of_inp))
        self.del_bias = np.matrix(np.zeros(No_of_n)).transpose()
        self.lr = learningRate
        self.training = True

        self.h_w = self.del_weights * 0
        self.h_b = self.del_bias * 0

        self.G = self.del_weights + 0.000001

    def calculate_delta(self, input):
        if self.training:
            self.del_weights += self.error * np.matrix(input) * self.lr
            self.del_bias += self.error * self.lr

    def update_WB(self):
        alpha = .9
        self.G = self.G * alpha
        self.G += (1-alpha)*np.multiply(self.del_weights, self.del_weights)

        self.h_w = alpha * self.h_w + np.multiply(np.divide(1, np.sqrt(self.G)), self.del_weights) * self.lr
        self.h_b = alpha * self.h_b + self.del_bias * self.lr

        self.weights += self.h_w
        self.bias += self.h_b


        self.del_bias = self.del_bias * 0
        self.del_weights = self.del_weights * 0
if __name__ == '__main__':
    # autoencoder

    input_size = 100
    Ann_full = ANN((input_size, 120, 120, input_size), Name='NewTry_1')
    Ann_full.create()
    # Ann_full.load()
    data = []
    for i in range(10000):
        inp = np.random.randint(0, 2, size=input_size)
        data.append(inp)
    print('data created')
    error = Ann_full.train(data, data, 25, 1)
    draw_errors(error, 500)

    Ann_full.save()
    data = []
    for i in range(10000):~~~
        inp = np.random.randint(0, 2, size=(input_size,))
        data.append(inp)
    print('data creayed')
    error = Ann_full.train(data, data, 15, 25)
    draw_errors(error, 501)

    Ann_full.save()
