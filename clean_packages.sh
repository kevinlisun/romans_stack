#!/bin/bash

echo "Script for cleaning binaries from stack.";
echo "Author: Kevin Li Sun, Henry Cheng Zhao, Jan-2016"
echo

echo "Removing backup files."
find ./ -name '*~' | xargs rm

find ./ -name '*.caffemodel' | xargs rm
find ./ -name '*.solverstate' | xargs rm



