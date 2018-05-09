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
    
    if split == 'train':
        pydata_params = dict(dataset_dir='/home/kevin/dataset/ws_exp/gp_labelled', split=split, mean=(104.00698793, 116.66876762, 122.67891434),
            seed=1337, img_size=(224,224), crop_size=(224,224,224,224))
        pylayer = 'WashingtonDataLayerHHA'
	pydata_params['randomize'] = True
	pydata_params['batch_size'] = 32
    elif split == 'test':
        pydata_params = dict(dataset_dir='/home/kevin/dataset/washington_rgbd_dataset', split=split, mean=(104.00698793, 116.66876762, 122.67891434),
            seed=1337, img_size=(224,224), crop_size=(224,224,224,224))
	pylayer = 'WashingtonDataLayerHHAtest'
	pydata_params['randomize'] = False
	pydata_params['batch_size'] = 1
    else:
        n.img = L.Input(name='input', ntop=2, shape=[dict(dim=1),dict(dim=1),dict(dim=224),dict(dim=224)])

        

    #---------------------------------Data Layer---------------------------------------#
    n.rgb, n.depth, n.label = L.Python(name="data", module='data_layers.washington_data_layer', layer=pylayer,
            ntop=3, param_str=str(pydata_params))

    #n.rgb_crop = L.Python(n.rgb, name="crop_rgb", module='data_layers.random_crop_layer', layer='RandomCropLayer', ntop=1, param_str=str(dict(h=224,w=224)))
    #n.depth_crop = L.Python(n.depth, name="crop_depth", module='data_layers.random_crop_layer', layer='RandomCropLayer', ntop=1, param_str=str(dict(h=227,w=227)))

    #---------------------------------RGB-Net---------------------------------------#

    # the vgg 16 base net
    n.conv1_1, n.relu1_1 = conv_relu("conv1_1", n.rgb, 64, pad=1, lr1=1, lr2=2)
    n.conv1_2, n.relu1_2 = conv_relu("conv1_2", n.relu1_1, 64, lr1=1, lr2=2)
    n.rgb_pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu("conv2_1", n.rgb_pool1, 128, lr1=1, lr2=2)
    n.conv2_2, n.relu2_2 = conv_relu("conv2_2", n.relu2_1, 128, lr1=1, lr2=2)
    n.rgb_pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu("conv3_1", n.rgb_pool2, 256, lr1=1, lr2=2)
    n.conv3_2, n.relu3_2 = conv_relu("conv3_2", n.relu3_1, 256, lr1=1, lr2=2)
    n.conv3_3, n.relu3_3 = conv_relu("conv3_3", n.relu3_2, 256, lr1=1, lr2=2)
    n.rgb_pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu("conv4_1", n.rgb_pool3, 512, lr1=1, lr2=2)
    n.conv4_2, n.relu4_2 = conv_relu("conv4_2", n.relu4_1, 512, lr1=1, lr2=2)
    n.conv4_3, n.relu4_3 = conv_relu("conv4_3", n.relu4_2, 512, lr1=1, lr2=2)
    n.rgb_pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu("conv5_1", n.rgb_pool4, 512, lr1=1, lr2=2)
    n.conv5_2, n.relu5_2 = conv_relu("conv5_2", n.relu5_1, 512, lr1=1, lr2=2)
    n.conv5_3, n.relu5_3 = conv_relu("conv5_3", n.relu5_2, 512, lr1=1, lr2=2)
    n.rgb_pool5 = max_pool(n.relu5_3)

    # fully conv
    n.rgb_fc6, n.rgb_relu6 = fc_relu(n.rgb_pool5, 4096, lr1=1, lr2=2)
    n.rgb_drop6 = L.Dropout(n.rgb_relu6, dropout_ratio=0.5, in_place=True)
    n.rgb_fc7, n.rgb_relu7 = fc_relu(n.rgb_drop6, 4096, lr1=1, lr2=2)
    n.rgb_drop7 = L.Dropout(n.rgb_relu7, dropout_ratio=0.5, in_place=True)

    n.rgb_fc8 = fc(n.rgb_drop7, 51, lr1=0, lr2=0)

    #---------------------------------Depth-Net---------------------------------------#

    # the base net
    # the vgg 16 base net
    n.conv1_1d, n.relu1_1d = conv_relu("conv1_1d", n.depth, 64, pad=1, lr1=1, lr2=2)
    n.conv1_2d, n.relu1_2d = conv_relu("conv1_2d", n.relu1_1d, 64, lr1=1, lr2=2)
    n.depth_pool1 = max_pool(n.relu1_2d)

    n.conv2_1d, n.relu2_1d = conv_relu("conv2_1d", n.depth_pool1, 128, lr1=1, lr2=2)
    n.conv2_2d, n.relu2_2d = conv_relu("conv2_2d", n.relu2_1d, 128, lr1=1, lr2=2)
    n.depth_pool2 = max_pool(n.relu2_2d)

    n.conv3_1d, n.relu3_1d = conv_relu("conv3_1d", n.depth_pool2, 256, lr1=1, lr2=2)
    n.conv3_2d, n.relu3_2d = conv_relu("conv3_2d", n.relu3_1d, 256, lr1=1, lr2=2)
    n.conv3_3d, n.relu3_3d = conv_relu("conv3_3d", n.relu3_2d, 256, lr1=1, lr2=2)
    n.depth_pool3 = max_pool(n.relu3_3d)

    n.conv4_1d, n.relu4_1d = conv_relu("conv4_1d", n.depth_pool3, 512, lr1=1, lr2=2)
    n.conv4_2d, n.relu4_2d = conv_relu("conv4_2d", n.relu4_1d, 512, lr1=1, lr2=2)
    n.conv4_3d, n.relu4_3d = conv_relu("conv4_3d", n.relu4_2d, 512, lr1=1, lr2=2)
    n.depth_pool4 = max_pool(n.relu4_3d)

    n.conv5_1d, n.relu5_1d = conv_relu("conv5_1d", n.depth_pool4, 512, lr1=1, lr2=2)
    n.conv5_2d, n.relu5_2d = conv_relu("conv5_2d", n.relu5_1d, 512, lr1=1, lr2=2)
    n.conv5_3d, n.relu5_3d = conv_relu("conv5_3d", n.relu5_2d, 512, lr1=1, lr2=2)
    n.depth_pool5 = max_pool(n.relu5_3d)

    # fully conv
    n.depth_fc6, n.depth_relu6 = fc_relu(n.depth_pool5, 4096, lr1=1, lr2=2)
    n.depth_drop6 = L.Dropout(n.depth_relu6, dropout_ratio=0.5, in_place=True)
    n.depth_fc7, n.depth_relu7 = fc_relu(n.depth_drop6, 4096, lr1=1, lr2=2)
    n.depth_drop7 = L.Dropout(n.depth_relu7, dropout_ratio=0.5, in_place=True)

    n.depth_fc8 = fc(n.depth_drop7, 51, lr1=1, lr2=2)

    #-----------------------------------final output---------------------------------#
    # Concatenation
    #n.concat = L.Concat(n.rgb_drop7, n.depth_drop7, axis=1)
    #n.rgbd_fc8 = fc(n.concat, 6, lr1=1, lr2=2)

    if split != 'deploy':
	n.rgb_accuracy = L.Accuracy(n.rgb_fc8, n.label)
        n.rgb_loss = L.SoftmaxWithLoss(n.rgb_fc8, n.label)
	n.depth_accuracy = L.Accuracy(n.depth_fc8, n.label)
        n.depth_loss = L.SoftmaxWithLoss(n.depth_fc8, n.label)
        #n.accuracy = L.Accuracy(n.rgbd_fc8, n.label)
        #n.loss = L.SoftmaxWithLoss(n.rgbd_fc8, n.label)

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
