#include "Sample.cpp"
#include "RNN.h"

int main(int argc, char* argv[]) {
    // read arguments
    int neuronsNum = 0;
    int inputsNum = 0;
    if(argc == 3) {
        neuronsNum = std::stoi(argv[1]);
        inputsNum = std::stoi(argv[2]);
    }
    if (neuronsNum < 4) {
        std::cout << "Min neurons num = 4" << std::endl;
        neuronsNum = 4;
    }
    if (neuronsNum % 2 == 1) {
        std::cout << "Even number of neurons required" << std::endl;
        neuronsNum += 1;
    }
    // gen sample and save it to Sample object
    genSample();
    Sample sample;
    readSampleFromFile(sample, inputsNum);

    //radial neuron network init
    RNN rNet(sample, neuronsNum, inputsNum);

    // training and gnuplot animation
    rNet.training(sample);

    return 0;
}