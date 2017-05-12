# Prerequsites #

(1) install caffe and pycaffe: http://caffe.berkeleyvision.org/install_apt.html

(2) install GPy (http://sheffieldml.github.io/GPy/):
-> sudo pip install --upgrade pip
-> sudo pip install GPy

<<<<<<< HEAD
(3) install SVM and Python interface
	cd ~
	git clone https://github.com/cjlin1/libsvm.git
	cd libsvm & make & cd python & make
	# add libsvm to the python path
	export PYTHONPATH=/home/kevin/libsvm/python:$PYTHONPATH

=======
(3) install libSVM and Python interface: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
>>>>>>> 62916adfeabb796099f3771f07a708e61fb7f57b


<<<<<<< HEAD
install dependencies:

GNU Scientific Library (GSL): sudo apt-get install libgsl0-dev

setup
=======
# Setup #
>>>>>>> 62916adfeabb796099f3771f07a708e61fb7f57b

create a catkin workspace

-> mkdir ~/catkin_ws/src

-> cd ~/catkin_ws/src

create .rosinstall file and copy the followings:

- git: {local-name: romans_stack, uri: 'git@bitbucket.org:kevinlisun/romans_stack.git', version: master}
- git: {local-name: iai_kinect2, uri: 'https://github.com/code-iai/iai_kinect2', version: master}
- git: {local-name: rtabmap_ros, uri: 'https://github.com/introlab/rtabmap_ros.git', version: master}
- git: {local-name: iiwa_stack, uri: 'https://github.com/SalvoVirga/iiwa_stack.git', version: master}

-> wstool update
-> rosdep install --from-paths src --ignore-src -r -y
-> cd ..
-> catkin_make -DCMakeType=RELEASE


# Programming Style #

We are following 
ROS C++ stype: http://wiki.ros.org/CppStyleGuide
Python REP8 style: http://www.ros.org/reps/rep-0008.html
