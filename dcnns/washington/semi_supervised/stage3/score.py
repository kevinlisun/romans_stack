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

    loss_rgb = 0
    loss_depth = 0
    loss_rgbd = 0
    accuracy_rgb = 0
    accuracy_depth = 0
    accuracy_rgbd = 0
    confusion_mat = np.zeros([51, 51])

    
    for idx in dataset:
        net.forward()

        label = int(net.blobs[gt].data[0])
        rgb_prediction = (net.blobs['rgb_'+layer].data).argmax()
	depth_prediction = (net.blobs['depth_'+layer].data).argmax()
	rgbd_prediction = (net.blobs['rgbd_'+layer].data).argmax()

        if rgb_prediction == label:
            accuracy_rgb += 1;

        if depth_prediction == label:
            accuracy_depth += 1;

	if rgbd_prediction == label:
            accuracy_rgbd += 1;


        loss_rgb += net.blobs['rgb_loss'].data.flat[0]
	loss_depth += net.blobs['depth_loss'].data.flat[0]
	loss_rgbd += net.blobs['rgbd_loss'].data.flat[0]

        confusion_mat[label,rgbd_prediction] += 1


    return accuracy_rgb/len(dataset), accuracy_depth/len(dataset), accuracy_rgbd/len(dataset), loss_rgb/len(dataset), loss_depth/len(dataset), loss_rgbd/len(dataset), confusion_mat/(np.tile(confusion_mat.sum(1),(51,1))).T

def do_tests(net, iter, save_dir, test_indices, layer='fc8', gt='label'):

    accuracy_rgb, accuracy_depth, accuracy_rgbd, loss_rgb, loss_depth, loss_rgbd, confusion_mat = compute_classification_error(net, save_dir, test_indices, layer, gt)
    # mean pixel-wise absolute error
    print '>>>', datetime.now(), 'Iteration', iter, 'mean classification accuracy (rgb) ', accuracy_rgb
    print '>>>', datetime.now(), 'Iteration', iter, 'mean classification accuracy (depth)', accuracy_depth
    print '>>>', datetime.now(), 'Iteration', iter, 'mean classification accuracy (rgbd) ', accuracy_rgbd
    # mean pixel-wise relative absolute error
    print '>>>', datetime.now(), 'Iteration', iter, 'mean testing loss (rgb)', loss_rgb
    print '>>>', datetime.now(), 'Iteration', iter, 'mean testing loss (depth)', loss_depth
    print '>>>', datetime.now(), 'Iteration', iter, 'mean testing loss (rgbd)', loss_rgbd

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
