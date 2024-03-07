#!/bin/sh
cat run.sh
g++ main.cpp RNN.cpp GreenMatrix.cpp -o radialNetApprox
./radialNetApprox 8 2
