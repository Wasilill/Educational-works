//
// Created by dmitry on 10.01.2021.
//

#ifndef LAB4_GREENMATRIX_H
#define LAB4_GREENMATRIX_H

#include <vector>
#include <iostream>
#include <cmath>

class GreenMatrix {
private:
    int rows;  // N
    int cols;  // M
    std::vector<std::vector<double>> p;
public:
    // init matrix: P - samples num, L - neurons num
    GreenMatrix(int P, int L);
    void changeRows(int row1, int row2);
    void setP(int i, int j, double value) { p[i][j] = value; }
    double getP(int i, int j) { return p[i][j]; }
    GreenMatrix invert();
    GreenMatrix transpose();
    GreenMatrix operator*(GreenMatrix matrix2);
    GreenMatrix pseudoinv();
    void print();
};


#endif //LAB4_GREENMATRIX_H
