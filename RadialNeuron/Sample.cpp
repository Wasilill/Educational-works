#include "Sample.h"

void genSample() {  // gen sample for function y=sin(x1)+cos(x2) on x interval [-5;5]
    std::ofstream file("sample.txt");
    double leftBound = -5;
    double rightBound = 5;
    double delta = (rightBound - leftBound) / (SAMPLE_POINTS_IN_ROW - 1);
    for(int i = 0; i < SAMPLE_POINTS_IN_ROW; i++) {
        for(int j = 0; j < SAMPLE_POINTS_IN_ROW; j++) {
            double x1 = leftBound + i * delta;
            double x2 = leftBound + j * delta;
            double y = sin(x1) + cos(x2);
            file << x1 << ' ' << x2 << ' ' << y << std::endl;
        }
    }
    file.close();
}

int readSampleFromFile(Sample& sample, int inputsNum) {
    std::ifstream sampleFile;
    sampleFile.open("sample.txt");
    if(!sampleFile.is_open()) {
        std::cerr << "ERROR: Opening file with sample." << std::endl;
        return OPENING_ERROR;
    }
    std::cout << "INFO: File with sample successfully opened." << std::endl;

    std::cout << "INFO: Start reading sample from file." << std::endl;
    sample.setInputsNum(inputsNum);
    std::vector<std::vector<double>> x(SAMPLE_DIM);
    std::vector<double> d;
    double number = 0.;
    for(int i = 0; i < SAMPLE_DIM; i++) {
        for(int j = 0; j < inputsNum; j++) {
            sampleFile >> number;
            x[i].push_back(number);
        }
        sampleFile >> number;
        d.push_back(number);
    }

    std::cout << "INFO: Sample successfully read." << std::endl;
    sampleFile.close();

    // save sample
    sample.setX(x);
    sample.setD(d);
    return 0;
}