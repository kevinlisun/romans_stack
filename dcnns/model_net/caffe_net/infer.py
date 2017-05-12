import numpy as np
from scipy import io
import cv2

import caffe


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


def load_frames_from_model( file ):

	dataset_dir = '/home/kevin/dataset/processed_data'
	frame_num = 24
	variable = 'depth_map'
	img_size = [ 250, 250 ]
	mean = -10

	for i in range(0,frame_num):

		mat = io.loadmat('{}/{}_{}.mat'.format(dataset_dir, file, i+1))
		datai = mat[variable].astype(np.double)

		if not img_size is None:
			datai = cv2.resize(datai, (img_size[0],img_size[0]))

		datai -= mean
		if len(datai.shape) < 3:
			datai = datai[np.newaxis, ...]
		else:
			datai = datai.transpose((2,0,1))

		if i is 0:
			data = np.zeros([frame_num, datai.shape[0],datai.shape[1],datai.shape[2]], dtype=np.double)

			data[i,:,:,:] = datai

	return data


def main():       
	# init
	caffe.set_device(0)
	caffe.set_mode_gpu()

	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	in_ = load_frames_from_model( 'airplane/test/airplane_33' )

	# load net
	net = caffe.Net('deploy.prototxt', '/home/kevin/snapshot/model-net.caffemodel', caffe.TEST)

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['img'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	out = net.blobs['fc8'].data[0]

	result = get_object_prediction(out, 'ave')

	print net.blobs['fc8'].data.shape
	#out = net.blobs['upscore'].data[0]
	#out = net.blobs['fc8_conv'].data[0]

	print result


if __name__ == '__main__':main()

