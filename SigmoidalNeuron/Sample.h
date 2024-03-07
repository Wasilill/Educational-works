//
// Created by dmitry on 18.12.2020.
//

#ifndef LAB3_SAMPLE_H
#define LAB3_SAMPLE_H

#include <vector>
#include <iostream>

class Sample {
private:
    std::vector<std::vector<double>> x;
    std::vector<double> d;
    int inputsNum;
public:
    void setInputsNum(int _inputsNum) { inputsNum = _inputsNum; }
    int getInputsNum() { return inputsNum; }
    void setX(std::vector<std::vector<double>> _x) { x = _x; }
    void setD(std::vector<double> _d) { d = _d; }
    std::vector<std::vector<double>> getX() { return x; }
    std::vector<double> getD() { return d; }
    void print() {
        for(int i = 0; i < x.size(); i++) {
            for(int j = 0; j < x[0].size(); j++)
                std::cout << "x" << i << "_" << j << " = " << x[i][j] << ";" << ' ';
            std::cout << "d" << i << " = " << d[i] << std::endl;
        }
    }
    int size() { return d.size(); }
};

#endif //LAB3_SAMPLE_H
