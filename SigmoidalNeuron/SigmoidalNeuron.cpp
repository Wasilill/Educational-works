//
// Created by dmitry on 17.12.2020.
//

#include "SigmoidalNeuron.h"


void SigmoidalNeuron::setInputValues(const std::vector<double>& _x_vector) {
    x_vector.clear();
    for(double i : _x_vector)
        x_vector.push_back(i);
}

void SigmoidalNeuron::setRandomWeights() {
    // random double numbers
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1.0, 3.0);
    for(int i = 0; i < inputsNum; i++)
        w_vector[i] = (double)dist(mt);
}

void SigmoidalNeuron::setInputWeights(const std::vector<double>& _w_vector) {
    w_vector.clear();
    for(double i : _w_vector)
        w_vector.push_back(i);
}

double SigmoidalNeuron::generateOutputValues() {
    // count the weighted amount - U
    double Ui = 0;
    for(int i = 0; i < x_vector.size(); i++)
        Ui += w_vector[i] * x_vector[i];
    // count the output value
    return 1.0 / (1.0 + exp(-Ui));
}

double SigmoidalNeuron::generateOutputValues(const std::vector<double>& _x_vector) {
    setInputValues(_x_vector);
    // count the weighted amount - U
    double Ui = 0;
    for(int i = 0; i < x_vector.size(); i++)
        Ui += w_vector[i] * x_vector[i];
    // count the output value
    return 1.0 / (1.0 + exp(-Ui));
}

double SigmoidalNeuron::calcOutputDiff(std::vector<std::vector<double>> x, std::vector<double> d, std::vector<double> wNew) {
    double resultCur = 0.;
    double resultNew = 0.;
    for (int k = 0; k < d.size(); k++) {   // k - sampleTwoInputsNeuron index
        resultCur += pow(generateOutputValues(x[k]) - d[k], 2);

        // count the weighted amount - U
        double U = 0;
        for(int j = 0; j < wNew.size(); j++)  // j - input index
            U += wNew[j] * x[k][j];
        resultNew += pow(1.0 / (1.0 + exp(-U)) - d[k], 2);
    }
    return (resultNew - resultCur) / 2;
}

int SigmoidalNeuron::writeTrainingResultsToFile(const std::string& path) {
    // open file with sampleTwoInputsNeuron
    std::ofstream resultsFile;
    resultsFile.open(path.c_str());
    if(!resultsFile.is_open()) {
        std::cerr << "ERROR: Opening results file." << std::endl;
        return OPENING_ERROR;
    }

    if(inputsNum == 3) {
        // generate input and output values for gnuplot
        for (double x1 = -2; x1 < 10; x1 += 0.1){
            for(double x2 = -2; x2 < 10; x2 += 0.1) {
                std::vector<double> point;
                point.push_back(1.0);
                point.push_back(x1);
                point.push_back(x2);
                double signal = generateOutputValues(point);
                resultsFile<< point[1] << '\t' << point[2] << '\t' << signal << '\n';
            }
            resultsFile << '\n';
        }
    }
    if(inputsNum == 2) {
        for (double x1 = 0.; x1 < 14.1; x1 += 0.5) {
            std::vector<double> point;
            point.push_back(1.0);
            point.push_back(x1);
            double signal = generateOutputValues(point);
            resultsFile << point[1] << '\t' << signal << '\n' << '\n';
        }
    }
    resultsFile.close();
    return 0;
}

int SigmoidalNeuron::createPlot(int iterNum) {
    FILE *gp = popen("gnuplot -persist","w"); // gp - дескриптор канала
    if (gp == nullptr){
        printf("Error opening pipe to GNU plot.\n");
        return GNUPLOT_ERROR;
    }
    //fprintf(gp, "set terminal  png size 1000,650 font 'Verdana, 10' \n");
    //fprintf(gp, "set output '../trainingResult.png'\n");
    //fprintf(gp, "set terminal gif animate delay 35\n");
    //fprintf(gp, "set output '../trainingResult.gif'\n");
    fprintf(gp, "set view 0,0\n");
    fprintf(gp, "set xlabel 'x1'\n");
    fprintf(gp, "set xrange [-2:10]\n");
    fprintf(gp, "set ylabel 'x2'\n");
    fprintf(gp, "set zlabel 'E(w)'\n");
    //fprintf(gp, "set colorbox vertical default\n");
    fprintf(gp, "set palette model RGB defined (0 \"white\", 1 \"white\", 1 \"red\", 2 \"white\", 3 \"white\" )\n");
    fprintf(gp, "set pm3d map\n");
    for(int i = 0; i < iterNum + 1; i++) {
        std::string path = "results/trainingIter" + std::to_string(i) + ".dat";
        std::string plot = "splot \'" + path + "\' u 1:2:3 title \"\" ," + " 'sample.dat' u 1:2:3 with points pt 7 ps 0.7 lt rgb 'black' title \"\"\n";
        fprintf(gp, "%s", plot.c_str());
    }
    pclose(gp);
    return 0;
}

int SigmoidalNeuron::offlineTraining(Sample& sample, double nuMin) {
    double nu = 0.4;
    int countChangeNuOk = 0;
    int iterNum = 0;

    std::vector<std::vector<double>> x = sample.getX();
    std::vector<double> d = sample.getD();
    std::vector<double> wCorrection(inputsNum);
    int counter = 5;
    while(nu >= nuMin || calcTrainingQualityTerminal(sample) != 100) {

        // throw off the weights if the solution is not found in 5 iterations
        counter--;
        if (counter == 0 && calcTrainingQualityTerminal(sample) != 100) {
            setRandomWeights();
            nu = 0.4;
            counter = 5;
        }

        // calc wCorrect vector
        for(int k = 0; k < sample.size(); k++) {  // k - sample index
            double y_k = generateOutputValues(x[k]);
            double dy_k = y_k * (1 - y_k);
            for(int j = 0; j < sample.getInputsNum(); j++)  // j - input index
                wCorrection[j] += nu * dy_k * (y_k - d[k]) * x[k][j];
        }
        // calc wNew
        std::vector<double> w = getW();      // current weights
        std::vector<double> wNew(inputsNum);            // weights after correction
        for(int i = 0; i < w.size(); i++)
            wNew[i] = (w[i] - wCorrection[i]);

        // outputDiff = E(wNew(t+1)) - E(w(t))
        double outputDiff = calcOutputDiff(x, d, wNew);

        // nu adaptation strategy:
        // E(wNew(t+1)) - E(w(t)) < 0
        if(outputDiff < 0) {
            setInputWeights(wNew);

            std::string path = "results/trainingIter" + std::to_string(iterNum) + ".dat";
            if(writeTrainingResultsToFile(path))
                return OPENING_ERROR;

            countChangeNuOk++;
            if(countChangeNuOk > 2) {
                countChangeNuOk = 0;
                nu *= 2;
            }
            iterNum++;
        } else {
            nu /= 2;
            countChangeNuOk = 0;
        }
    }
    createPlot(iterNum - 1);
    return 0;
}

void SigmoidalNeuron::calcTrainingQuality(Sample& sample, std::string when) {
    std::vector<std::vector<double>> x = sample.getX();
    std::vector<double> d = sample.getD();
    double correctAnswers = 0;
    for(int k = 0; k < sample.size(); k++) {  // k - sampleTwoInputsNeuron index
        double y_k = generateOutputValues(x[k]);
        if ((int)round(y_k) == d[k])
            correctAnswers++;
    }
    std::cout << "Correct answers " + when + " training: " << correctAnswers / sample.size() * 100 << "%" << std::endl;
}

double SigmoidalNeuron::calcTrainingQualityTerminal(Sample& sample) {
    std::vector<std::vector<double>> x = sample.getX();
    std::vector<double> d = sample.getD();
    double correctAnswers = 0;
    for(int k = 0; k < sample.size(); k++) {  // k - sampleTwoInputsNeuron index
        double y_k = generateOutputValues(x[k]);
        if ((int)round(y_k) == d[k])
            correctAnswers++;
    }
    return correctAnswers / sample.size() * 100;
}
