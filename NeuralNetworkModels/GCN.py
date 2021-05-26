#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pdb
import numpy as np
import sys
import codecs
import random
import os
from os import path
import argparse
import pdb
import copy
import re
import scipy as sp
from haversine import haversine
import theano
from lasagne.utils import floatX
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
from sklearn.preprocessing import normalize
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict, Counter
from _collections import defaultdict
import networkx as nx
from data import DataLoader, dump_obj, load_obj
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#####################################################################
########                                                     ########
########                       GCN Model                     ########
########                                                     ########
#####################################################################


'''
These sparse classes are copied from https://github.com/Lasagne/Lasagne/pull/596/commits
'''
class SparseInputDenseLayer(DenseLayer):
    '''
    An input layer for sparse input and dense output data.
    '''
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        #activation = S.dot(input, self.W)
        activation = S.structured_dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class SparseInputDropoutLayer(DropoutLayer):
    '''
    A dropout layer for sparse input data, note that this layer
    can not be applied to the output of SparseInputDenseLayer
    because the output of SparseInputDenseLayer is dense.
    '''
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)

class SparseConvolutionDenseLayer(DenseLayer):
    '''
    A graph convolutional layer where input is sparse and output is dense
    '''
    def __init__(self, incoming, A=None, **kwargs):
        super(SparseConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.A = A


    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.structured_dot(input, self.W)
        #do the convolution
        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class ConvolutionDenseLayer(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    '''

    def __init__(self, incoming, A=None, **kwargs):
        super(ConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.A = A

    def get_output_for(self, input, **kwargs):
        target_indices = kwargs.get('target_indices')
        activation = T.dot(input, self.W)
        #do the convolution
        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class ConvolutionDenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(ConvolutionDenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices

    def get_output_for(self, input, A=None, target_indices=None, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution

        if A:
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class ConvolutionDenseLayer3(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, **kwargs):
        super(ConvolutionDenseLayer3, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, A=None, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution

        if A:
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class ConvolutionDenseLayer_zero(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, A=None, **kwargs):
        super(ConvolutionDenseLayer_zero, self).__init__(incoming, **kwargs)
        self.A = A

    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution

        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)

class ConvolutionLayer(lasagne.layers.Layer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, A=None, nonlinearity=lasagne.nonlinearities.linear, **kwargs):
        super(ConvolutionLayer, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices
        self.A = A
        self.nonlinearity = nonlinearity

    def get_output_for(self, input, target_indices=None, **kwargs):
        #do the convolution
        activation = S.structured_dot(self.A, input)


        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class DenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(DenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices

    def get_output_for(self, input, target_indices=None, **kwargs):
        activation = T.dot(input, self.W)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)


class SparseConvolutionDenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input is sparse and output is dense
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''
    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(SparseConvolutionDenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices


    def get_output_for(self, input, A=None, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")


        activation = S.structured_dot(input, self.W)
        if A:
            #do the convolution
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


class MultiplicativeGatingLayer(lasagne.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1.0 - inputs[0]) * inputs[2]

def highway_dense(incoming, gconv=False,
                  #Wh=lasagne.init.Orthogonal(),
                  Wh=lasagne.init.GlorotUniform(),
                  bh=lasagne.init.Constant(0.0),
                  #Wt=lasagne.init.Orthogonal(),
                  Wt=lasagne.init.GlorotUniform(),
                  bt=lasagne.init.Constant(-4.0),
                  nonlinearity=lasagne.nonlinearities.sigmoid, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    #bt should be set to -2 according to http://people.idsia.ch/~rupesh/very_deep_learning/ and kim et al 2015
    # regular layer
    #l_h = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    if gconv:
        l_h = ConvolutionDenseLayer2(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    else:
        l_h = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    # gate layer
    l_t = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                                   nonlinearity=T.nnet.sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming), l_t

def residual_dense(incoming, nonlinearity=lasagne.nonlinearities.selu):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    convX = ConvolutionDenseLayer2(incoming, num_units=num_inputs, nonlinearity=None)
    convX_plus_X = lasagne.layers.ElemwiseSumLayer([convX, incoming], coeffs=1, cropping=None)
    return lasagne.layers.NonlinearityLayer(convX_plus_X, nonlinearity=nonlinearity)



def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class GraphConv():
    '''
    A general theano-based graph convolutional neural network model based in Kipf (2016).
    Note that the input is assumed to be sparse (as in BoW model of text).
    '''
    def __init__(self, input_size, output_size, hid_size_list, regul_coef, drop_out, dtype='float32', batchnorm=False, highway=True):

        self.input_size = input_size
        self.output_size = output_size
        self.hid_size_list = hid_size_list
        self.regul_coef = regul_coef
        self.drop_out = drop_out
        self.dtype=dtype
        self.dtypeint = 'int64' if self.dtype == 'float64' else 'int32'
        self.fitted = False
        self.batchnorm = batchnorm
        self.highway = highway
        logging.info('highway is {}'.format(self.highway))

    def build_model(self, A, use_text=True, use_labels=True, seed=77):
        np.random.seed(seed)
        logging.info('Graphconv model input size {}, output size {} and hidden layers {} regul {} dropout {}.'.format(self.input_size, self.output_size, str(self.hid_size_list), self.regul_coef, self.drop_out))
        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.train_indices_sym = T.lvector()
        self.dev_indices_sym = T.lvector()
        self.test_indices_sym = T.lvector()
        self.A_sym = S.csr_matrix(name='NormalizedAdj', dtype=self.dtype)
        self.train_y_sym = T.lvector()
        self.dev_y_sym = T.lvector()
        #nonlinearity = lasagne.nonlinearities.rectify
        #Wh = lasagne.init.GlorotUniform(gain='relu')
        nonlinearity = lasagne.nonlinearities.tanh
        Wh = lasagne.init.GlorotUniform(gain=1)

        #input layer
        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
        l_hid = SparseInputDenseLayer(l_in, num_units=self.hid_size_list[0], nonlinearity=nonlinearity)

        #add hidden layers

        l_hid = lasagne.layers.dropout(l_hid, p=self.drop_out)
        num_inputs_txt = int(np.prod(l_hid.output_shape[1:]))
        Wt_txt = lasagne.init.Orthogonal()
        self.gate_layers = []
        logging.info('{} gconv layers'.format(len(self.hid_size_list)))
        if len(self.hid_size_list) > 1:
            for i, hid_size in enumerate(self.hid_size_list):
                if i == 0:
                    #we have already added the first hidden layer which is nonconvolutional
                     continue
                else:
                    if self.highway:
                        l_hid, l_t_hid = highway_dense(l_hid, gconv=True, nonlinearity=nonlinearity, Wt=Wt_txt, Wh=Wh)
                        self.gate_layers.append(l_t_hid)
                    else:
                        l_hid = ConvolutionDenseLayer2(l_hid, num_units=hid_size, nonlinearity=nonlinearity)

        self.l_out = ConvolutionDenseLayer3(l_hid, num_units=self.output_size, nonlinearity=lasagne.nonlinearities.softmax)
        self.output = lasagne.layers.get_output(self.l_out, {l_in:self.X_sym}, A=self.A_sym, deterministic=False)
        self.train_output = self.output[self.train_indices_sym, :]
        self.train_pred = self.train_output.argmax(-1)
        self.dev_output = self.output[self.dev_indices_sym, :]
        self.dev_pred = self.dev_output.argmax(-1)
        self.train_acc = T.mean(T.eq(self.train_pred, self.train_y_sym))
        self.dev_acc = T.mean(T.eq(self.dev_pred, self.dev_y_sym))
        self.train_loss = lasagne.objectives.categorical_crossentropy(self.train_output, self.train_y_sym).mean()
        if self.regul_coef > 0:
            #add l1 regularization
            self.train_loss += lasagne.regularization.regularize_network_params(self.l_out, penalty=lasagne.regularization.l1) * self.regul_coef
            #add l2 regularization
            self.train_loss += lasagne.regularization.regularize_network_params(self.l_out, penalty=lasagne.regularization.l2) * self.regul_coef

        self.dev_loss = lasagne.objectives.categorical_crossentropy(self.dev_output, self.dev_y_sym).mean()

        #deterministic output
        self.determ_output = lasagne.layers.get_output(self.l_out, {l_in:self.X_sym}, A=self.A_sym, deterministic=True)
        self.test_output = self.determ_output[self.test_indices_sym, :]
        self.test_pred = self.test_output.argmax(-1)

        self.gate_outputs = []
        self.f_gates = []
        for i, l in enumerate(self.gate_layers):

            self.gate_outputs.append(lasagne.layers.get_output(l, {l_in:self.X_sym}, A=self.A_sym, deterministic=True))
            self.f_gates.append(theano.function([self.X_sym, self.A_sym], self.gate_outputs[i], on_unused_input='warn'))




        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(self.train_loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)

        self.f_train = theano.function([self.X_sym, self.train_y_sym, self.dev_y_sym, self.A_sym, self.train_indices_sym, self.dev_indices_sym],
                                       [self.train_loss, self.train_acc, self.dev_loss, self.dev_acc, self.output], updates=updates, on_unused_input='warn')#, mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
        self.f_val = theano.function([self.X_sym, self.A_sym, self.test_indices_sym], [self.test_pred, self.test_output], on_unused_input='warn')


        self.init_params = lasagne.layers.get_all_param_values(self.l_out)

        return self.l_out

    def fit(self, X, H, Y, train_indices, val_indices, n_epochs=10000, batch_size=1000, max_down=10, pseudolikelihood_thresh=0.2, verbose=True, seed=77):
        np.random.seed(seed)
        logging.info('training for {} epochs with batch size {}'.format(n_epochs, batch_size))
        best_params = None
        best_val_loss = sys.maxsize
        best_val_acc = 0.0
        n_validation_down = 0
        report_k_epoch = 1

        X_train, y_train = X, Y[train_indices]
        y_dev = Y[val_indices]
        for n in range(n_epochs):
            l_train, acc_train, l_val, acc_val, all_probs = self.f_train(X_train, y_train, y_dev, H, train_indices, val_indices)
            l_train, acc_train = l_train.item(), acc_train.item()
            l_val, acc_val = l_val.item(), acc_val.item()

            if  l_val < best_val_loss:
                best_val_loss = l_val
                best_val_acc = acc_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            if verbose:
                if n % report_k_epoch == 0:
                    logging.info('epoch {} train loss {:.2f} train acc {:.2f} val loss {:.2f} val acc {:.2f} best val acc {:.2f} maxdown {}'.format(n, l_train, acc_train, l_val, acc_val, best_val_acc, n_validation_down))
            if n_validation_down > max_down and n > 2 * report_k_epoch * max_down:
                logging.info('validation results went down. early stopping ...')
                break
        self.best_params = best_params
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        self.fitted = True

    def predict(self, X, A, test_indices):
        preds_test, prob_test = self.f_val(X, A, test_indices)
        return preds_test, prob_test

    def reset(self):
        lasagne.layers.set_all_param_values(self.l_out, self.init_params)

    def save(self, dumper, filename='./model.pkl'):
        if self.fitted:
            logging.info('dumping model params in {}'.format(filename))
            dumper(self.best_params, filename)
        else:
            logging.warn('The model is not trained yet!')

    def load(self, loader, filename):
         logging.info('loading the model from {}'.format(filename))
         self.best_params = loader(filename)
         lasagne.layers.set_all_param_values(self.l_out, self.best_params)
         self.fitted = True

    def get_gates(self, X, A):
        gates = []
        for fn in self.f_gates:
            gate = fn(X, A)
            gates.append(gate)
        return gates




#####################################################################
########                                                     ########
########                       GCN Main                      ########
########                                                     ########
#####################################################################




logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info('In order to work for big datasets fix https://github.com/Theano/Theano/pull/5721 should be applied to theano.')
np.random.seed(77)
model_args = None


def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))

    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
    dump_file = os.path.join(data_home, 'dump.pkl')
    if os.path.exists(dump_file) and not model_args.builddata:
        logging.info('loading data from dumped file...')
        data = load_obj(dump_file)
        logging.info('loading data finished!')
        return data

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    vocab = dl.vectorizer.vocabulary_
    logging.info('saving vocab in {}'.format(vocab_file))
    dump_obj(vocab, vocab_file)
    logging.info('vocab dumped successfully!')
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()

    dl.get_graph()
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')

    adj.setdiag(0)
    #selfloop_value = np.asarray(adj.sum(axis=1)).reshape(-1,)
    selfloop_value = 1
    adj.setdiag(selfloop_value)
    n,m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0/sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    A = D_pow_neghalf * adj * D_pow_neghalf
    A = A.astype(dtype)
    logging.info('adjacency matrix created.')

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}



    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    data = (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data


def main(data, args, **kwargs):
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', [100])
    dropout = kwargs.get('dropout', 0.0)
    regul = kwargs.get('regularization', 1e-6)
    dtype = 'float32'
    dtypeint = 'int32'
    check_percentiles = kwargs.get('percent', False)
    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    logging.info('stacking training, dev and test features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))
    Y = Y.astype(dtypeint)
    X = X.astype(dtype)
    A = A.astype(dtype)
    if args.vis:
        from deepcca import draw_representations
        draw_representations(A.dot(X), Y, filename='gconv1.pdf')
        draw_representations(A.dot(A.dot(X)), Y, filename='gconv2.pdf')
    input_size = X.shape[1]
    output_size = np.max(Y) + 1
    verbose = not args.silent
    fractions = args.lblfraction
    stratified = False
    all_train_indices = np.asarray(range(0, X_train.shape[0])).astype(dtypeint)
    logging.info('running mlp with graph conv...')
    clf = GraphConv(input_size=input_size, output_size=output_size, hid_size_list=hidden_size, regul_coef=regul, drop_out=dropout, batchnorm=args.batchnorm, highway=model_args.highway)
    clf.build_model(A, use_text=args.notxt, use_labels=args.lp, seed=model_args.seed)

    for percentile in fractions:
        logging.info('***********percentile %f ******************' %percentile)
        model_file = './data/model-{}-{}.pkl'.format(A.shape[0], percentile)
        if stratified:
            all_chosen = []
            for lbl in range(0, np.max(Y_train) + 1):
                lbl_indices = all_train_indices[Y_train == lbl]
                selection_size =  int(percentile * len(lbl_indices)) + 1
                lbl_chosen = np.random.choice(lbl_indices, size=selection_size, replace=False).astype(dtypeint)
                all_chosen.append(lbl_chosen)
            train_indices = np.hstack(all_chosen)
        else:
            selection_size = min(int(percentile * X.shape[0]), all_train_indices.shape[0])
            train_indices = np.random.choice(all_train_indices, size=selection_size, replace=False).astype(dtypeint)
        num_training_samples = train_indices.shape[0]
        logging.info('{} training samples'.format(num_training_samples))
        #train_indices = np.asarray(range(0, int(percentile * X_train.shape[0]))).astype(dtypeint)
        dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype(dtypeint)
        test_indices = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype(dtypeint)
        # do not train, load
        if args.load:
            #### IF LOAD NEED TO LOAD MODEL AND PREDICT    ####
            ####     COORDINATES FOR SAVING IN JSON        ####
            report_results = False
            model_file1 = './data/model-9475-0.6.pkl'
            clf.load(load_obj, model_file1)
            logging.info('test results:')
            y_pred, _ = clf.predict(X, A, test_indices)
            coordinates_count = []
            for key in classLatMedian.keys():
                current_count = np.count_nonzero(y_pred == int(key))
                if (current_count > 0):
                    coordinates_count.append({"lat": classLatMedian[key], "long": classLonMedian[key], "count": current_count})

            # mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
            d = {"locations": coordinates_count}
            with open("coordinates_count.json", "w", encoding="utf-8") as json_file:
                json.dump(d, json_file)
        else:
            #reset the network parameters if already fitted with another data
            if clf.fitted:
                clf.reset()
            clf.fit(X, A, Y, train_indices=train_indices, val_indices=dev_indices, n_epochs=10000, batch_size=batch_size, max_down=args.maxdown, verbose=verbose, seed=model_args.seed)
            if args.save:
                clf.save(dump_obj, model_file)

            logging.info('dev results:')
            y_pred, _ = clf.predict(X, A, dev_indices)
            mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
            with open('gcn_{}_percent_pred_{}.pkl'.format(percentile, output_size), 'wb') as fout:
                pickle.dump((distances, latlon_true, latlon_pred), fout)
            logging.info('test results:')
            y_pred, _ = clf.predict(X, A, test_indices)
            geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)

    if args.feature_report:
        vocab_file = os.path.join(args.dir, 'vocab.pkl')
        if not os.path.exists(vocab_file):
            logging.error('vocab file {} not found'.format(vocab_file))
            return
        else:
            vocab = load_obj(vocab_file)
        logging.info('{} vocab loaded from file'.format(len(vocab)))
        train_vocab = set([term for term, count in Counter(np.nonzero(X[train_indices])[1]).iteritems() if count >= 10])
        dev_vocab = set(np.nonzero(X[dev_indices].sum(axis=0))[1])
        X_onehot = sp.sparse.diags([1] * len(vocab), dtype=dtype)
        A_onehot = X_onehot
        feature_report(clf, vocab, X_onehot, A_onehot, classLatMedian, classLonMedian, train_vocab, dev_vocab, topk=200, dtypeint=dtypeint)


def feature_report(model, vocab, X, A, classLatMedian, classLonMedian, train_vocab=set(), dev_vocab=set(), topk=20, dtypeint='int32', filename='important_features.txt'):
    eval_indices = np.asarray(range(X.shape[0])).astype(dtypeint)
    preds, probs = model.predict(X, A, eval_indices)
    id2v = {v: k for k, v in vocab.iteritems()}
    logging.info('{} train vocab are being excluded!'.format(len(train_vocab)))
    #select top k most important features for each class
    feature_importance = np.argsort(-probs, axis=0)
    with codecs.open(filename, 'w', encoding='utf-8') as fout:
        for lbl in range(probs.shape[1]):
            important_vocab = ' '.join([id2v[idx] for idx in feature_importance[:, lbl].reshape(-1).tolist() if idx not in train_vocab][0:topk])
            lat, lon = classLatMedian[str(lbl)], classLonMedian[str(lbl)]
            fout.write(u'location: {},{} \nimportant features: {} \n\n'.format(lat, lon, important_vocab))
    logging.info('important features are written to {}'.format(filename))


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '-i','--dataset', metavar='str', help='dataset for dialectology', type=str, default='na')
    parser.add_argument( '-bucket','--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument( '-batch','--batch', metavar='int', help='SGD batch size', type=int, default=500)
    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument( '-mindf','--mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float', help='regularization coefficient)', type=float, default=1e-6)
    parser.add_argument( '-cel','--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument( '-conv', '--convolution', action='store_true', help='if true do convolution')
    parser.add_argument( '-tune', '--tune', action='store_true', help='if true tune the hyper-parameters')
    parser.add_argument( '-tf', '--tensorflow', action='store_true', help='if exists run with tensorflow')
    parser.add_argument( '-batchnorm', action='store_true', help='if exists do batch normalization')
    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)
    parser.add_argument( '-percent', action='store_true', help='if exists loop over different train/dev proportions')
    parser.add_argument( '-vis', metavar='str', help='visualise representations', type=str, default=None)
    parser.add_argument('-builddata', action='store_true', help='if exists do not reload dumped data, build it from scratch')
    parser.add_argument('-lp', action='store_true', help='if exists use label information')
    parser.add_argument('-notxt', action='store_false', help='if exists do not use text information')
    parser.add_argument( '-maxdown', help='max iter for early stopping', type=int, default=10)
    parser.add_argument('-silent', action='store_true', help='if exists be silent during training')
    parser.add_argument('-highway', action='store_true', help='if exists use highway connections else do not')
    parser.add_argument( '-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-save', action='store_true', help='if exists save the model after training')
    parser.add_argument('-load', action='store_true', help='if exists load pretrained model from file')
    parser.add_argument('-feature_report', action='store_true', help='if exists report the important features of each location')
    parser.add_argument('-lblfraction', nargs='+', type=float, help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args

    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
    main(data, args, batch=args.batch, hidden=args.hid, regularization=args.regularization, dropout=args.dropout, percent=args.percent)
