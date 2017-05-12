#!/usr/bin/python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-11-03 20:24:43
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2016-11-06 23:49:03


import caffe
from caffe import layers as L, params as P


def conv_relu(name, bottom, nout, ks=3, stride=1, pad=1, group=1, lr1=1, lr2=2):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, group=group, weight_filler=dict(type='xavier'),
        param=[dict(name=name+"_w", lr_mult=lr1, decay_mult=1), dict(name=name+"_b", lr_mult=lr2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def fc(bottom, nout, lr1=1, lr2=2, filler_type='xavier'):
    ip = L.InnerProduct(bottom, num_output=nout, weight_filler=dict(type=filler_type),
      param=[dict(lr_mult=lr1, decay_mult=1), dict(lr_mult=lr2, decay_mult=0)])
    return ip

def fc_relu(bottom, nout, lr1=1, lr2=2, filler_type='xavier'):
    ip = L.InnerProduct(bottom, num_output=nout, weight_filler=dict(type=filler_type),
      param=[dict(lr_mult=lr1, decay_mult=1), dict(lr_mult=lr2, decay_mult=0)])
    return ip, L.ReLU(ip, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def ave_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)

def cnn(split):
    n = caffe.NetSpec()
    pydata_params = dict(dataset_dir='/home/kevin/dataset/normal_feature', variable='normal_map', split=split, mean=(0,0,0),
            seed=1337, batch_size=256, img_size=(250,250))
    if split == 'deploy':
        n.img = L.Input(name='input', ntop=2, shape=[dict(dim=1),dict(dim=3),dict(dim=130),dict(dim=130)])
    else:
        if split is 'train':
            pydata_params['dtype'] = 'frame'
            pylayer = 'ModelNetDataLayer'
        else:
            pydata_params['dtype'] = 'object'
            pylayer = 'ModelNetDataLayer'
    
        n.img, n.label = L.Python(module='data_layers.model_net_layer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1, n.relu1 = conv_relu("conv1", n.img, 96, ks=11, stride=4, pad=0)
    n.pool1 = max_pool(n.relu1, ks=3)
    n.norm1 = L.LRN(n.pool1, lrn_param=dict(local_size=5, alpha=0.0005, beta=0.75, k=2))
    # n.bn1 = L.BatchNorm(n.pool1, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], batch_norm_param=dict(use_global_stats=True))

    n.conv2, n.relu2 = conv_relu("conv2", n.norm1, 256, ks=5, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, ks=3)
    n.norm2 = L.LRN(n.pool2, lrn_param=dict(local_size=5, alpha=0.0005, beta=0.75, k=2))
    # n.bn2 = L.BatchNorm(n.pool2, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], batch_norm_param=dict(use_global_stats=True))


    n.conv3, n.relu3 = conv_relu("conv3", n.norm2, 384, ks=3, pad=1, group=2)

    n.conv4, n.relu4 = conv_relu("conv4", n.relu3, 256, ks=3, pad=1, group=2)
    
    n.pool5 = max_pool(n.relu4, ks=3)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, lr1=1, lr2=2)   
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6 , 4096, lr1=1, lr2=2)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.fc8 = fc(n.drop7, 40, lr1=1, lr2=2)

    if split != 'deploy':
        #n.accuracyt = L.Accuracy(n.predictT, n.labelT)
        #n.losst = L.SoftmaxWithLoss(n.predictT, n.labelT)

        n.accuracy = L.Accuracy(n.fc8, n.label)
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

    # n.display = L.Scale(n.corr, param=[dict(lr_mult=0)], filler=dict(type='constant',value=1.0))
    # n.fc9_bn = L.BatchNorm(n.relu9, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], batch_norm_param=dict(use_global_stats=True))

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(cnn('train')))

    with open('test.prototxt', 'w') as f:
        f.write(str(cnn('test')))

    #with open('deploy.prototxt', 'w') as f:
    #    f.write(str(cnn('deploy')))

if __name__ == '__main__':
    make_net()
