//
// Created by dmitry on 10.01.2021.
//

#include "RNN.h"

RNN::RNN(Sample& sample, int _neuronsNum, int _inputsNum) {
    neuronsNum = _neuronsNum;
    inputsNum = _inputsNum;
    for(int i = 0; i < neuronsNum; i++)
        neurons.emplace_back(inputsNum);

    // set initial widths S (Sigma)
    // sigma equal for ech neuron (but each axis has own sigma)
    std::vector<double> _s(inputsNum);
    std::vector<double> distBetweenNeuronCenters(inputsNum);

    // I divide the definition area into zones for each neuron
    // x_axis1 - number of zones on x1, x-axis2 - number of zones on x2
    std::vector<int> x_axis(inputsNum);  // only for 2-inputs neuron
    for(int i = 2; i <= sqrt(neuronsNum); i++) {
        if (neuronsNum % i == 0) {
            x_axis[0] = i;
            x_axis[1] = neuronsNum / i;
        }
    }
    std::cout << "neuronsNum = " << neuronsNum << std::endl;
    //std::cout << "zones: " << x_axis[0] << ' ' << x_axis[1] << std::endl;
    for(int i = 0; i < inputsNum; i++) {
        distBetweenNeuronCenters[i] = (sample.xMax()[i] - sample.xMin()[i]) / (x_axis[i]);
        _s[i] = 0.5 * distBetweenNeuronCenters[i];  // sigma on current axis
    }
    //std::cout << "s: " << _s[0] << ' ' << _s[1] << std::endl;

    // set initial neuron centers C
    std::vector<double> _c(inputsNum);
    int neuronId = 0;
    for(int i = 0; i < x_axis[0]; i++) {
        for(int j = 0; j < x_axis[1]; j++) {
            _c[0] = sample.xMin()[0] + _s[0] + i * distBetweenNeuronCenters[0];
            _c[1] = sample.xMin()[1] + _s[1] + j * distBetweenNeuronCenters[1];
            neurons[neuronId].setS(_s);
            neurons[neuronId].setC(_c);
            //std::cout << neuronId << std::endl;
            //neurons[neuronId].print();
            neuronId++;
        }
    }
}

double RNN::Y_approx() {
    double sum = 0;
    for(int l = 0; l < neuronsNum; l++)
        sum += w[l] * neurons[l].RadialFunctionValue();
    return sum;
}

void RNN::setW(Sample& sample) {
    GreenMatrix G(SAMPLE_DIM, neuronsNum);

    for(int i = 0; i < SAMPLE_DIM; i++) {
        for(int l = 0; l < neuronsNum; l++)
            neurons[l].setX(sample.getX(i));  // the current sample line (x) is entered into x of each neuron
        for(int l = 0; l < neuronsNum; l++)
            G.setP(i, l, neurons[l].RadialFunctionValue());
    }

    GreenMatrix Gp(neuronsNum, SAMPLE_DIM);
    Gp = G.pseudoinv();

    std::vector<double> _w(neuronsNum, 0);
    for(int i = 0; i < neuronsNum; i++) {
        for(int j = 0; j < SAMPLE_DIM; j++)
            _w[i] += Gp.getP(i, j) * sample.getD()[j];
    }
    w = _w;
}

double RNN::E(Sample& sample) {
    double eps = 0;
    for(int i = 0; i < SAMPLE_DIM; i++) {
        for(int l = 0; l < neuronsNum; l++)
            neurons[l].setX(sample.getX(i));
        eps += pow(Y_approx() - sample.getD()[i], 2);
    }
    return 0.5 * eps;
}

double RNN::approxError(Sample& sample) {
    double eps = 0;
    for(int i = 0; i < SAMPLE_DIM; i++) {
        for(int l = 0; l < neuronsNum; l++)
            neurons[l].setX(sample.getX(i));
        eps += pow(Y_approx() - sample.getD()[i], 2);
    }
    return sqrt(eps / SAMPLE_DIM);
}

double RNN::Egrad_dc(int l, int i, Sample& sample) {
    double grad = 0;
    for(int k = 0; k < SAMPLE_DIM; k++) {
        for(int nId = 0; nId < neuronsNum; nId++)
            neurons[nId].setX(sample.getX(k));
        grad += ( (Y_approx() - sample.getD()[k]) * w[l] * neurons[l].RadialFunctionValue() *
                  (neurons[l].getX()[i] - neurons[l].getC()[i]) ) / pow(neurons[l].getS()[i], 2);
    }
    return grad;
}

double RNN::Egrad_ds(int l, int i, Sample& sample) {
    double grad = 0;
    for(int k = 0; k < SAMPLE_DIM; k++) {
        for(int nId = 0; nId < neuronsNum; nId++)
            neurons[nId].setX(sample.getX(k));
        grad += ( (Y_approx() - sample.getD()[k]) * w[l] * neurons[l].RadialFunctionValue() *
                  pow(neurons[l].getX()[i] - neurons[l].getC()[i], 2) ) / pow(neurons[l].getS()[i], 3);
    }
    return grad;
}

void RNN::training(Sample& sample) {
    double nu = NU_MIN;
    int nuGCounter = 0;
    double epsPrev = 0;
    double epsNew = 0;
    std::vector<std::vector<double>> cPrev(neuronsNum, std::vector<double>(inputsNum, 0));
    std::vector<std::vector<double>> cNew(neuronsNum, std::vector<double>(inputsNum, 0));
    std::vector<std::vector<double>> sPrev(neuronsNum, std::vector<double>(inputsNum, 0));
    std::vector<std::vector<double>> sNew(neuronsNum, std::vector<double>(inputsNum, 0));
    int iterNum = 0;

    std::cout << "1 calculation of the weights of the output layer is accompanied by\n"
                 "25 cycles of refinement of the parameters of the radial functions." << std::endl;
    do{
        std::cout << "Wait. Radial Network training ..." << std::endl;
        // part 1: correct neuron's weights
        setW(sample);

        // part 2: adjusting the centers and widths of the Gaussian function for each neuron
        for(int t = 0; t < T_STEPS; t++) {
            epsPrev = E(sample);  // calculated the deviation of the approximation from the real function

            for(int l = 0; l < neuronsNum; l++) {
                for(int i = 0; i < inputsNum; i++) {
                    sPrev[l][i] = neurons[l].getS()[i];  // take the previous sigma values
                    cPrev[l][i] = neurons[l].getC()[i];  // take the previous centers values
                    sNew[l][i] = neurons[l].getS()[i] - nu * Egrad_ds(l, i, sample);
                    cNew[l][i] = neurons[l].getC()[i] - nu * Egrad_dc(l, i, sample);
                }
            }
            for(int l = 0; l < neuronsNum; l++) {
                neurons[l].setC(cNew[l]);
                neurons[l].setS(sNew[l]);
            }

            epsNew = E(sample);

            std::string path = "results/trainingIter" + std::to_string(iterNum++) + ".dat";
            printResultsToFile(sample, path);

            //std::cout << approxError(sample) << std::endl;
            //std::cout << "Целевая функция E = " << epsNew << std::endl;
            //std::cout << "epsNew - epsPrev = " << fabs(epsNew - epsPrev) << std::endl;

            // training rate adaptation strategy
            if(epsNew < epsPrev) {
                nuGCounter++;
                if(nuGCounter > 2) {
                    nu *= 2;
                    nuGCounter = 0;
                }
            } else {
                nuGCounter = 0;
                nu /= 2;
                for(int l = 0; l < neuronsNum; l++) {
                    neurons[l].setC(cPrev[l]);
                    neurons[l].setS(sPrev[l]);
                }
            }
        }
    }while(fabs(epsNew - epsPrev) >= EPS_MIN || approxError(sample) > 0.75);

    std::cout << "approxError = " << approxError(sample) << std::endl;
    //std::cout << fabs(epsNew - epsPrev) << " < EPS-MIN = " << EPS_MIN << std::endl;

    setW(sample);
    createPlot(sample, iterNum);
}

int RNN::printResultsToFile(Sample& sample, const std::string& path) {
    std::ofstream resultsFile;
    resultsFile.open(path.c_str());
    if(!resultsFile.is_open()) {
        std::cerr << "ERROR: Opening results file." << std::endl;
        return OPENING_ERROR;
    }

    for(int i = 0; i < SAMPLE_DIM; i++){
        for(int nId = 0; nId < neuronsNum; nId++)
            neurons[nId].setX(sample.getX(i));

        for(int j = 0; j < inputsNum; j++)
            resultsFile << sample.getX()[i][j] << ' ';
        resultsFile << Y_approx() << std::endl;
    }

    resultsFile.close();
    return 0;
}

int RNN::createPlot(Sample& sample, int iterNum) {
    FILE *gp = popen("gnuplot -persist","w"); // gp - pipe descriptor
    if (gp == nullptr){
        printf("Error opening pipe to GNU plot.\n");
        return GNUPLOT_ERROR;
    }
    fprintf(gp, "set title 'Обучение радиальной нейронной сети' font 'Arial,14'\n");
    fprintf(gp, "set grid xtics ytics ztics mxtics mytics\n");
    fprintf(gp, "set xrange [-5:5]\n");
    fprintf(gp, "set xlabel 'x1'\n");
    fprintf(gp, "set yrange [-5:5]\n");
    fprintf(gp, "set ylabel 'x2'\n");
    fprintf(gp, "set zlabel 'f(x)'\n");
    fprintf(gp, "set isosamples 100,100\n");

    std::cout << "Wait. animation creates ..." << std::endl;
    for(int i = 0; i < iterNum; i++) {
        std::string path = "results/trainingIter" + std::to_string(i) + ".dat";
        std::string plot = "splot \'" + path + "\' u 1:2:3 w points pt 7 lt rgb 'green' title 'f(x) approximation " +
                std::to_string(i + 1) + "/" + std::to_string(iterNum) +
                "', [-5:5][-5:5] sin(x)+cos(y) title 'f(x) = sin(x1) + cos(x2)  ' lt rgb 'blue'\n";
        fprintf(gp, "%s", plot.c_str());
    }
    pclose(gp);
    return 0;
}