/*
Вариант 2 - 5 (сигмоидальный нейрон, выборка №5 обучающих данных)
*/

#include <vector>
#include <iostream>
#include "Sample.h"
#include "SigmoidalNeuron.h"

#define NU_MIN 1e-5
#define OPENING_ERROR -1

int readSampleFromFile(const std::string& path, Sample& sample) {
    // open file with sampleTwoInputsNeuron
    std::ifstream sampleFile;
    sampleFile.open(path);
    if(!sampleFile.is_open()) {
        std::cerr << "ERROR: Opening file with sample." << std::endl;
        return OPENING_ERROR;
    }
    std::cout << "INFO: File with sample successfully opened." << std::endl;

    // read sampleTwoInputsNeuron from file
    std::cout << "INFO: Start reading sample from file." << std::endl;
    int inputsNum = 3;
    int sampleDim = 33;
    sample.setInputsNum(inputsNum);

    std::vector<std::vector<double>> x(sampleDim);
    std::vector<double> d;
    double coefficient = 1;
    double number = 0.;
    for(int i = 0; !sampleFile.eof(); i++) {
        x[i].push_back(1.);  // polarization input
        for(int j = 0; j < inputsNum - 1; j++) {
            sampleFile >> number;
            x[i].push_back(number / coefficient);
        }
        sampleFile >> number;
        d.push_back(number / coefficient);
    }

    std::cout << "INFO: Sample successfully read." << std::endl;
    sampleFile.close();

    // save sampleTwoInputsNeuron to Sample object
    sample.setX(x);
    sample.setD(d);
    return 0;
}

int main() {
    std::string path = "sample.dat";
    Sample trainingSample;
    if(readSampleFromFile(path, trainingSample))
        return OPENING_ERROR;
    //trainingSample.print();

    // create a sigmoidal three-input neuron
    SigmoidalNeuron neuron(trainingSample.getInputsNum());

    // set random weights to inputs
    neuron.setRandomWeights();
    //neuron.printWeights();

    std::cout << "Сomparison of the results of the neuron with the training set:" << std::endl;
    neuron.calcTrainingQuality(trainingSample, "before");

    if(neuron.offlineTraining(trainingSample, NU_MIN))
        return OPENING_ERROR;

    neuron.calcTrainingQuality(trainingSample, "after ");
}
