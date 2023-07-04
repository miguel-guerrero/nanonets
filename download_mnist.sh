#!/bin/bash

mkdir -p tests/mnist
curl -o tests/mnist/mnist_train.csv  https://pjreddie.com/media/files/mnist_train.csv
curl -o tests/mnist/mnist_test.csv   https://pjreddie.com/media/files/mnist_test.csv
