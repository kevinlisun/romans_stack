#!/bin/bash

for req in $(cat cat_list.txt)

do rosrun odr train_classifier_ws.py gpu rgbd $req
   sleep 20
   rosrun odr classify_meta_data_ws.py gpu rgbd $req 
    
done 
