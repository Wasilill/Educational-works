//
// Created by dmitry on 09.01.2021.
//

#ifndef LAB4_SAMPLE_H
#define LAB4_SAMPLE_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#define SAMPLE_POINTS_IN_ROW 20
#define SAMPLE_DIM 400
#define OPENING_ERROR -1

class Sample {
private:
    std::vector<std::vector<double>> x;
    std::vector<double> d;
    int inputsNum;
public:
    void setInputsNum(int _inputsNum) { inputsNum = _inputsNum; }
    int getInputsNum() const { return inputsNum; }
    void setX(std::vector<std::vector<double>> _x) { x = _x; }
    void setD(std::vector<double> _d) { d = _d; }
    std::vector<std::vector<double>> getX() { return x; }
    std::vector<double> getX(int i) { return x[i]; }
    std::vector<double> getD() { return d; }
    void print() {
        for(int i = 0; i < x.size(); i++) {
            for(int j = 0; j < x[0].size(); j++)
                std::cout << "x" << i << "_" << j << " = " << x[i][j] << ";" << ' ';
            std::cout << "d" << i << " = " << d[i] << std::endl;
        }
    }
    std::vector<double> xMin() {
        std::vector<double> xMin(inputsNum);
        for(int i = 0; i < inputsNum; i++)
            xMin[i] = x[0][i];
        return xMin;
    }
    std::vector<double> xMax() {
        std::vector<double> xMax(inputsNum);
        for(int i = 0; i < inputsNum; i++)
            xMax[i] = x[SAMPLE_DIM - 1][i];
        return xMax;
    }
    int size() { return d.size(); }
};

#endif //LAB4_SAMPLE_H
