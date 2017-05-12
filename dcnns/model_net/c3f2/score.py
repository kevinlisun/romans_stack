# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-11-04 16:14:29
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2016-11-06 05:18:42

from __future__ import division
import caffe
import numpy as np
import os
from datetime import datetime
from PIL import Image

from scipy import io


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)


def compute_abs_error(a, b):
    return np.mean(np.abs(a-b))

def get_object_prediction(predictions, method):

    if method is 'max':
        confidences = predictions.max(axis=1)
        key_frame = confidences.argmax()
        result = predictions[key_frame,:].argmax()
    elif method is 'ave':
        result = predictions.sum(axis=0).argmax()
    else:
        print "ERROR given a incorrect pooling method."

    return int(result)

def compute_classification_error(net, save_dir, dataset, layer, gt):

    # import shutil
    # shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss = 0
    accuracy_max = 0
    accuracy_ave = 0
    confusion_mat = np.zeros([40, 40])

    
    for idx in dataset:
        net.forward()

        label = int(net.blobs[gt].data[0])
        predictions = net.blobs[layer].data

        prediction = get_object_prediction(predictions, 'max')

        if prediction == label:
            accuracy_max += 1;

        prediction = get_object_prediction(predictions, 'ave')

        if prediction == label:
            accuracy_ave += 1;


        loss += net.blobs['loss'].data.flat[0]

        confusion_mat[label,prediction] += 1


    return accuracy_max/len(dataset), accuracy_ave/len(dataset), loss/len(dataset), confusion_mat/(np.tile(confusion_mat.sum(1),(40,1))).T

def do_tests(net, iter, save_dir, test_indices, layer='fc8', gt='label'):

    accuracy_max, accuracy_ave, loss, confusion_mat = compute_classification_error(net, save_dir, test_indices, layer, gt)
    # mean pixel-wise absolute error
    print '>>>', datetime.now(), 'Iteration', iter, 'mean classification accuracy (max) ', accuracy_max

    print '>>>', datetime.now(), 'Iteration', iter, 'mean classification accuracy (ave)', accuracy_ave
    # mean pixel-wise relative absolute error
    print '>>>', datetime.now(), 'Iteration', iter, 'mean testing loss', loss

    print '>>>', datetime.now(), 'Iteration', iter, 'mean confusion matrix'
    print np.diag(confusion_mat)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    io.savemat('{}/result_iter_{}.mat'.format(save_dir, iter), {'confusion_mat':np.array(confusion_mat)})


def model_classification_test(solver, save_dir, test_indices, layer='fc8', gt='label'):
    global data_mean
    global data_std

    print '>>>', datetime.now(), 'Begin model classification tests'
    solver.test_nets[0].share_with(solver.net)

    test_indices = test_indices[:,0]

    do_tests(solver.test_nets[0], solver.iter, save_dir, test_indices, layer, gt)
