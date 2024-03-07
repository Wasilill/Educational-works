//
// Created by dmitry on 10.01.2021.
//

#ifndef LAB4_RNEURON_H
#define LAB4_RNEURON_H

#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

class RNeuron {
private:
    int inputsNum;
    std::vector<double> x;
    std::vector<double> c;
    std::vector<double> s;
public:
    explicit RNeuron(int _inputsNum): inputsNum(_inputsNum) {};
    void setX(std::vector<double> _x) { x = std::move(_x); };
    void setC(std::vector<double> _c) { c = std::move(_c); };
    void setS(std::vector<double> _s) { s = std::move(_s); };
    std::vector<double> getX() { return x; };
    std::vector<double> getC() { return c; };
    std::vector<double> getS() { return s; };
    // determine how close the point with x coordinates is to the center of the neuron
    double RadialFunctionValue() {
        double U = 0.;
        for(int i = 0; i < inputsNum; i++)
            U += pow(x[i] - c[i], 2) / pow(s[i], 2);
        return exp(-0.5 * U);
    }
};


#endif //LAB4_RNEURON_H
