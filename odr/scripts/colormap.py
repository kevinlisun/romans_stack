# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-12-09 15:10:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-02-20 20:23:06
import numpy as np

def getColor( label ):

	return {
		0: [100, 100, 100],
		1: [0, 0, 255],
		2: [0, 255, 0],
		3: [255, 0, 0],
		4: [255, 255, 0],
		5: [128, 0, 128],
		6: [0, 0, 0],
		7: [255, 255, 255],
		8: [0, 64, 0],
		9: [0, 192, 192],
		10: [255, 128, 0],
	}[label]

def getColoredSemanticmap( smap ):

	height = smap.shape[0]
	width = smap.shape[1]

	cmap = np.zeros( [height, width, 3], dtype=np.int8 )
	for i in xrange(height):
		for j in xrange(width):
			color = getColor(smap[i,j])
			cmap[i,j,:] = np.asarray(color, dtype=np.int8)

	return cmap

