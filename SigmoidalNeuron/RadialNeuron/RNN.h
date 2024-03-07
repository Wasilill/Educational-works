#ifndef LAB4_RNN_H
#define LAB4_RNN_H


#include <iostream>
#include <vector>
#include <cmath>

#include "Sample.h"
#include "GreenMatrix.h"
#include "RNeuron.h"

#define NU_MIN 1e-2
#define EPS_MIN 1e-1
#define T_STEPS 25
#define GNUPLOT_ERROR - 1

class RNN {
private:
    int neuronsNum;
    int inputsNum;
    std::vector<RNeuron> neurons;
    std::vector<double> w;  // neuron's weight
public:
    RNN(Sample& sample, int _neuronsNum, int _inputsNum);

    double Y_approx();
    void setW(Sample& sample);
    // consider the objective function to be minimized
    double E(Sample& sample);
    double approxError(Sample& sample);
    // we consider how much to change the coefficients C and S
    double Egrad_dc(int l, int i, Sample& sample);
    double Egrad_ds(int l, int i, Sample& sample);

    void training(Sample& sample);

    int printResultsToFile(Sample& sample, const std::string& path);
    static int createPlot(Sample& sample, int iterNum);
};


#endif //LAB4_RNN_H
