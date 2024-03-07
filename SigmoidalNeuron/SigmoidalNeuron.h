//
// Created by dmitry on 17.12.2020.
//

#ifndef LAB3_SIGMOIDALNEURON_H
#define LAB3_SIGMOIDALNEURON_H

#include <utility>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include "Sample.h"

#define OPENING_ERROR -1
#define GNUPLOT_ERROR -2

class SigmoidalNeuron {
private:
    int inputsNum;
    std::vector<double> x_vector;
    std::vector<double> w_vector;
public:
    explicit SigmoidalNeuron(int _inputsNum): inputsNum(_inputsNum) { for(int i = 0; i < inputsNum; i++) w_vector.push_back(0);}
    void setRandomWeights();
    void setInputValues(const std::vector<double>& _x_vector);
    void setInputWeights(const std::vector<double>& _w_vector);
    void printWeights() { for(double i : w_vector) { std::cout << i << " "; } std::cout << std::endl; }
    std::vector<double> getX() { return x_vector; }
    std::vector<double> getW() { return w_vector; }

    double generateOutputValues();
    double generateOutputValues(const std::vector<double>& _x_vector);
    double calcOutputDiff(std::vector<std::vector<double>> x, std::vector<double> d, std::vector<double> wNew);

    int writeTrainingResultsToFile(const std::string& path);

    static int createPlot(int iterNum);

    int offlineTraining(Sample& sample, double nuMin);

    void calcTrainingQuality(Sample& sample, std::string when);
    double calcTrainingQualityTerminal(Sample& sample);
};


#endif //LAB3_SIGMOIDALNEURON_H
