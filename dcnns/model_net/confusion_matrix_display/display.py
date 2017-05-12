#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys

from plot_confusion_matrix import plot_confusion_matrix



def main():
    print ('__name__:', __name__)
    #confusion_matrix_minc_mat = sio.loadmat(sys.argv[1])
    #confusion_matrix_minc = confusion_matrix_minc_mat[confusion_matrix_minc_mat.keys()[1]]
  
    confusion_matrix_minc_probability_mat = sio.loadmat(sys.argv[1])
    confusion_matrix_minc_probability = np.around(np.array(confusion_matrix_minc_probability_mat['confusion_mat'])*100, decimals=2)
   
    #print confusion_matrix_minc_probability

    class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', \
        'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',  'flower_pot',\
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', \
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', \
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', \
        'wardrobe', 'xbox']


    #class_names = ['brick', 'carpet', 'ceramic','fabric','foliage','food','glass','hair','leather','metal','mirror',\
    #              'other','painted','paper','plastic','polishedstone','skin','sky','stone','tile','wallpaper','water','wood']

    print class_names

    plt.figure()
    plot_confusion_matrix(confusion_matrix_minc_probability, classes=class_names, normalize=False, title='Confusion matrix of MINC')
    plt.show()

if __name__ == '__main__':main()
