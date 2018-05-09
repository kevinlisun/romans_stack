## A Brief Description
- **odr**: **o**bject **d**etection and **r**ecognition framework.
- **camera**: camera simulator and configurations.
- **dcnn**: Deep Convolutional Neural Network architectures and training scripts.
- **matlab_toobox**: matlab scripts for data pre-processing.


## Prerequsites

- If you just want to run the **demo**.

  Install caffe and pycaffe: http://caffe.berkeleyvision.org/install_apt.html

- If you also want to run our **live demo**.

  In this implementation, MS kinect v2 is used unde Ubuntu through iai kinect2 package, so install iai_kinect2 following their instructions : https://github.com/code-iai/iai_kinect2 .

- If you also want to **train the models from the scrach**.

  Install GPy (http://sheffieldml.github.io/GPy/): 
  ```
  $ sudo pip install --upgrade pip
  $ sudo pip install GPy
  ```

- If you also want to do **comprision experiments** using R-CNN.

  Install SVM and Python interface: https://www.csie.ntu.edu.tw/~cjlin/libsvm/ .
  ```
  $ cd ~
  $ git clone https://github.com/cjlin1/libsvm.git
  $ cd libsvm & make & cd python & make
  ```
  Add libsvm to the python path:
  ```
  $ export PYTHONPATH=/home/kevin/libsvm/python:$PYTHONPATH
  ```

## Dependencies:

- **ROS**, install [ROS Indigo] (http://wiki.ros.org/indigo/Installation/Ubuntu).

- GNU Scientific Library (**GSL**).
  ```
  sudo apt-get install libgsl0-dev
  ```

## Install romans_stack
1. Create a catkin workspace:
  ```
  $ mkdir ~/catkin_ws/src
  $ cd ~/catkin_ws/src
  ```

2. Create .rosinstall file and copy the followings:
  ```
  - git: {local-name: romans_stack, uri: 'https://github.com/sunliamm/romans_stack', version: master}
  - git: {local-name: iai_kinect2, uri: 'https://github.com/code-iai/iai_kinect2', version: master}
  ```
3. Clone the repositories:
  ```
  $ wstool update
  $ rosdep install --from-paths src --ignore-src -r -y
  $ cd ..
  ```
4. Compile:
  ```
  $ catkin_make -DCMakeType=RELEASE
  ```
5. Add ROS workspace to the environment.

  add `source ~/catkin_ws/devel/setup.bash` to ~/.bashrc

## Run the Demo
Download the demo data (demo.rosbag file) and the trained caffe model (deploy.proto, romans_model_fast.caffemodel) from: https://drive.google.com/open?id=0B0jMesRKPfw9MGM4ekxiV2M1RWs
This demo assumes you download the 'romans' folder and put it under home directory (cd ~), change the directory depending your situation.

1. Get RGBD stream from rosbag .
  ```
  $ roslaunch camera kinect2_simulator.launch
  $ cd ~/romans/data & rosbag play --clock demo.bag
  ```

  **Or** get RGBD stream from kinect2
  ```
  $ roslaunch camera kinect2.launch
  ```

2. Run detection node .
  ```
  $ rosrun odr detection_server_kinect2
  ```

3. Run recognition node .
  ```
  $ rosrun odr inference_end2end.py /home/your_username/romans/models/fast
  ```

4. Run visualization node .
  ```
  $ rosrun odr visualization_server
  ```

5. Run the client .
  ```
  $ rosrun odr odr_test.py kinect2
  ```
## the semi-supervised demo in Washington RGBD dataset

1. Download the datset: http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/

2. create the experiment
   1) go to matlab_toolbox and 
   ```
   $ run script_create_experiment.m
   ```
   , and then split into labelled and unlabelled set
   ```
   $ slipt_labelled_unlabelled.m
   ```

3. label propagation, it takes several hours:
   ```
   $ cd ~/catkin_ws/src/romans_stack/odr/washington
   $ sh sh all_in_one.sh
   $ cd ~/catkin_ws/src/romans_stack/odr
   $ sh ./washington/all_in_one2.sh
   ```

4. train the dcnn with automatic labelled examples:
   ```
   $ cd ~/catkin_ws/src/romans_stack/dcnns/washington/semi_supervised
   $ sh train.sh
   ```

## Programming Style

This implementation is following:

ROS C++ style: http://wiki.ros.org/CppStyleGuide

Python REP8 style: http://www.ros.org/reps/rep-0008.html

## Reference
Li Sun, Cheng Zhao, Rustam Stolkin. Weakly-supervised DCNN for RGB-D Object Recognition in Real-World Applications Which Lack Large-scale Annotated Training Data. [ArXiv](https://arxiv.org/abs/1703.06370)
