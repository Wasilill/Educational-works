#!/bin/sh
rm -rf results neuron
mkdir results
g++ main.cpp SigmoidalNeuron.cpp SigmoidalNeuron.h Sample.h -o neuron -std=gnu++11
./neuron
