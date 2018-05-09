#!/bin/bash

for req in $(cat ./washington/cat_list2.txt)

do rosrun odr train_classifier_ws.py gpu rgbd $req
   sleep 20
   rosrun odr classify_meta_data_ws.py gpu rgbd $req 
done 
