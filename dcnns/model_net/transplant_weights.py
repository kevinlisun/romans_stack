import numpy as np
import matplotlib.pyplot as plt
import caffe

caffe.set_mode_gpu()

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('train_val.prototxt', 
                'VGG_fine_tuning.caffemodel', 
                caffe.TEST)
params = ['fc6', 'fc7', 'fc8_minc']
#fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

print "====================================================================================="
print "finish loading the source model"
print "====================================================================================="

for fc in params:
	print 'fc: {} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
	

net_full_conv = caffe.Net('train_val_conv.prototxt', 
                'VGG_fine_tuning.caffemodel', 
                caffe.TEST)
params_full_conv = ['fc6_conv', 'fc7_conv', 'fc8_conv']
#conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

print "====================================================================================="
print "finish loading the target model"
print "====================================================================================="

for conv in params_full_conv:
    print 'fc_conv: {} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)


# transplant the weights
for pr, pr_conv in zip(params, params_full_conv):
    print "-------------------------"
    print "-------------------------"
    print "-------------------------"
    print conv_params[pr_conv][0].shape
    print fc_params[pr][0].shape
    print "-------------------------"
    print "-------------------------"
    print "-------------------------"
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

print "save conv_caffemodel"
net_full_conv.save('./VGG_fine_tuning_full_conv.caffemodel')

print "====================================================================================="
