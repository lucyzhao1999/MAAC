import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import random
from collections import deque


class GetSAencodings:
    def __init__(self, layersWidths, stateDim, actionDim):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layersWidths = layersWidths

    def __call__(self, obs_, action_, scope):
        saEncoding_ = tf.concat(obs_ + action_, axis=1)
        with tf.variable_scope(scope):
            for layerWidth in self.layersWidths:
                saEncoding_ = layers.fully_connected(saEncoding_, num_outputs= layerWidth)

        return saEncoding_


class GetSencodings:
    def __init__(self, layersWidths, stateDim):
        self.stateDim = stateDim
        self.layersWidths = layersWidths

    def __call__(self, obs_, scope):
        sEncoding_ = obs_
        with tf.variable_scope(scope):
            for layerWidth in self.layersWidths:
                sEncoding_ = layers.fully_connected(sEncoding_, num_outputs=layerWidth)

        return sEncoding_


class BuildValueExtractor:
    def __init__(self, layersWidths, stateDim, actionDim, outputDim, activFunc = tf.nn.leaky_relu):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.outputDim = outputDim
        self.layersWidths = layersWidths
        self.activFunc = activFunc

    def __call__(self, saEncoding_, scope):
        activation_ = layers.fully_connected(activation_, num_outputs= self.outputDim, activation_fn= self.activFunc)

        return activation_

class BuildSelectorExtractor:
    def __init__(self, layersWidths, stateDim, actionDim, outputDim, activFunc = tf.nn.leaky_relu):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.outputDim = outputDim
        self.layersWidths = layersWidths
        self.activFunc = activFunc

    def __call__(self, obs_, action_, scope):
        activation_ = tf.concat(obs_ + action_, axis=1)

        with tf.variable_scope(scope):
            for layerWidth in self.layersWidths:
                activation_ = layers.fully_connected(activation_, num_outputs= layerWidth)

            activation_ = layers.fully_connected(activation_, num_outputs= self.outputDim, activation_fn= self.activFunc)

        return activation_
