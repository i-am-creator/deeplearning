import time
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


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
        # we are creating a toy ANN to classification of points
        # we need 2 inputs X, Y of points
        if not os.path.isdir('plots'):
            path = os.path.join('plots/')
            os.mkdir(path)
        if not os.path.isdir('networks'):
            path = os.path.join('networks/')
            os.mkdir(path)
        self.inp = np.matrix(np.zeros((size[0]))).transpose()
        # We have one perceptron
        self.activation = np.matrix(np.zeros(
            (size[-1]))).transpose()  # we are using only one perceptron so the activation of that also act as output
        self.weights = []
        self.bias = []
        self.errors = []
        for i in range(1, len(size)):
            self.weights.append(np.matrix(np.random.uniform(-1, 1, (size[i], size[i - 1]))))
            # self.errors.append(np.matrix(np.zeros((size[i]))).transpose())
            self.bias.append(np.matrix(np.zeros((size[i]))).transpose())

        self.error_o = np.zeros((size[-1]))
        # learning rate
        self.lr = .00003
        self.activFun = 'relu'
        self.name = Name
        if self.name != '':
            try:
                self.load()
            except:
                pass

    def save(self):
        w = np.asarray(self.weights)
        b = np.asarray(self.bias)
        np.save('./networks/' + self.name + "ws", w)
        np.save('./networks/' + self.name + "bs", b)

    def load(self, name=''):
        if name == '':
            name = self.name
        else:
            self.name = name
        self.weights = np.load('./networks/' + name + "ws.npy", allow_pickle=True)
        self.bias = np.load('./networks/' + name + "bs.npy", allow_pickle=True)

    def feedforward(self, inp):

        output = np.matrix(inp).transpose()
        self.inp = output
        for w, b in zip(self.weights, self.bias):
            output = act_fun(w * output + b, self.activFun)
        self.activation = output
        return output

    def train(self, inputs, targets, n_epoch=20):
        error_allEPOCH = []
        for epoch in range(n_epoch):
            errors = []
            start = time.time()
            for data, target in zip(inputs, targets):
                self.backprop(data, target)
                sum_e = 0
                for e in self.error_o:
                    sum_e += e ** 2
                errors.append(sum_e)
            print('>epoch=%d, error=%.3f' % (epoch, sum(errors)/len(errors)), ' time: ', timedelta(seconds=time.time() - start))
            draw_errors(np.asarray(errors).T[0][0], epoch)
            for e in errors:
                error_allEPOCH.append(np.asarray(e)[0][0])
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
        activation = self.inp
        activations = [activation]
        # print(np.size(self.errors))
        for w, b in zip(self.weights, self.bias):
            activation = act_fun(w * activation + b, self.activFun)
            activations.append(activation)
        temp_w = self.weights[::-1]
        temp_b = self.bias[::-1]
        temp_a = activations[::-1]
        errors = self.error_o
        for w, b, a in zip(temp_w[:], temp_b[:], temp_a[1:]):
            w = w.transpose()
            self.errors.append(errors)
            errors = w * errors
            errors = np.multiply(errors, del_act_fun(a, self.activFun))
        self.errors = self.errors[::-1]

        for i in range(1, len(activations)):
            self.weights[i - 1] += self.errors[i - 1] * activations[i - 1].transpose() * self.lr
            self.bias[i - 1] += np.multiply(self.errors[i - 1], self.lr)
        # print(activations)
        # print(np.size(self.errors))

        # del_w_x = del_act_fun(pred) * inp[0] * self.error
        # del_w_y = del_act_fun(pred) * inp[1] * self.error
        '''
        updating weights and bias
        '''
        # self.w_x += del_w_x * self.lr
        # self.w_y += del_w_y * self.lr
        # self.b += self.error * self.lr

if __name__ == '__main__':
    # autoencoder

    input_size = 100
    Ann_full = ANN((input_size, 120, 120, input_size), Name='NewTry_1')
    # Ann_full.load()
    data = []
    for i in range(10001):
        inp = np.random.randint(0, 2, size=(input_size,))
        data.append(inp)
    print('data created')
    error = Ann_full.train(data, data, 25)
    draw_errors(error, 500)

    Ann_full.save()
    data = []
    for i in range(10000):
        inp = np.random.randint(0, 2, size=(input_size,))
        data.append(inp)
    print('data creayed')
    error = Ann_full.train(data, data, 10)
    draw_errors(error, 501)

    Ann_full.save()
