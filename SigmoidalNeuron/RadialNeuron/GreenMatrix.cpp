#include "GreenMatrix.h"

#include <cmath>

GreenMatrix::GreenMatrix(int P, int L) {
    rows = P;
    cols = L;
    std::vector<std::vector<double>> _p(rows, std::vector<double>(cols, 0));
    p = _p;
}

void GreenMatrix::changeRows(int row1, int row2) {
    double tmp;
    for(int j = 0; j < cols; j++) {
        tmp = p[row1][j];
        p[row1][j] = p[row2][j];
        p[row2][j] = tmp;
    }
}

GreenMatrix GreenMatrix::invert() {
    int ind_max;
    float maxAi;
    GreenMatrix A(rows, 2*rows);
    GreenMatrix Temp(rows,rows);
    for(int i=0; i<rows; i++)
        for(int j=0; j<rows; j++)
            A.p[i][j] = p[i][j];
    for(int i=0; i<rows; i++){
        for(int j=rows; j<2*rows; j++){
            A.p[i][j] = 0;
        }
        A.p[i][i+rows] = 1;
    }
    if (rows == cols){
        for(int j=0; j<rows; j++){
            maxAi = A.p[j][j];
            ind_max = j;
            for(int i=j; i<rows; i++){
                if(fabs(A.p[i][j]) > std::fabs(maxAi)){
                    maxAi = A.p[i][j];
                    ind_max = i;
                }
            }
            A.changeRows(j,ind_max);
            for(int i=j; i<rows; i++){
                for(int k=j+1; k<2*rows; k++){
                    if(i == j){
                        A.p[i][k] /= A.p[i][i];
                    }
                    else A.p[i][k] -= (A.p[i][j] * A.p[j][k]);
                }
                if( i == j) A.p[i][j] = 1;
                else A.p[i][j] = 0;
            }
        }
        for(int j=rows-1; j>0; j--){
            for(int i=j-1; i>=0; i--){
                for(int k=rows; k<2*rows; k++){
                    A.p[i][k] -=(A.p[i][j] * A.p[j][k]);
                }
                A.p[i][j] = 0;
            }
        }
        for(int i=0; i<rows; i++)
            for(int j=0; j<rows; j++)
                Temp.p[i][j] = A.p[i][j+rows];
        return Temp;
    }
    else {
        std::cout <<"Number of Colomns doesn't equal number of Rows " << std::endl;
        return *this;
    }
}

GreenMatrix GreenMatrix::transpose() {
    GreenMatrix MatrixT(cols,rows);
    for(int i = 0; i < cols; i++)
        for(int j = 0; j < rows; j++)
            MatrixT.p[i][j] = p[j][i];
    return MatrixT;
}

GreenMatrix GreenMatrix::operator*(GreenMatrix matrix2){
    GreenMatrix tmpMatrix(rows, matrix2.cols);
    if(cols == matrix2.rows) {
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < matrix2.cols; j++) {
                for(int k = 0; k < cols; k++)
                    tmpMatrix.p[i][j] += p[i][k] * matrix2.p[k][j];
            }
        return tmpMatrix;
    } else {
        std::cout <<"Number of Colomns doesn't equal number of Rows " << std::endl;
        return *this;
    }
}

GreenMatrix GreenMatrix::pseudoinv() {
    GreenMatrix GTG(cols, cols);
    GTG = this->transpose() * (*this);      // GTG = G.transpose * G

    GreenMatrix tmpMatrix(rows, cols);
    tmpMatrix = GTG;                        // tmpmatrix = G.transpose * G
    GTG = GTG.invert();                     // GTG = (G.transpose * G) ^ (-1)

    //tmpMatrix = GTG * tmpMatrix;            // единичная матрица: E = (G.transpose * G) ^ (-1) * (G.transpose * G)

    tmpMatrix = GTG * (this->transpose());  // tmpmatrix = (G.transpose * G) ^ (-1) * G.transpose
    return tmpMatrix;
}

void GreenMatrix::print() {
    for(int i = 0; i < p.size(); i++) {
        for(int j = 0; j < p[0].size(); j++)
            std::cout << p[i][j] << ' ';
        std::cout << std::endl;
    }
    std::cout << "rows = " << rows << " cols = " << cols << std::endl;
}