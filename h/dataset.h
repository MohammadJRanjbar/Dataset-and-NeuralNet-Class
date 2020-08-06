#ifndef dataset_H
#define dataset_H
#include "Matrix.h"
#include <stdio.h>
#include <cmath>
#include <array>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>      
#include <random>
#include <optional>
class Dataset
{
    private:
        Matrix inputs;
        Matrix targets;
        Matrix train_inputs;
        Matrix train_targets;
        Matrix test_inputs;
        Matrix test_targets;
        double percentage{ 70 };
        size_t no_of_samples;
        size_t input_dim;
        size_t target_dim;
    public:
    void show();
    Dataset() = default;
    Dataset(Matrix inputs, Matrix targets, double percentage = 70);
    void Train_test_split();
    void shuffle();
    std::vector<Matrix> operator[](size_t j);
    size_t getNoOfSamples()const;
    size_t getNoOfTrainSamples();
    size_t getNoOfTestSamples();
    size_t getInputDim();
    size_t getTargetDim();
    Matrix getInputs()const;
    Matrix getTargets()const;
    Matrix getTrainInputs();
    Matrix getTrainTargets();
    Matrix getTestInputs();
    Matrix getTestTargets();
    Dataset operator+(const Dataset& dataset);
    friend std::ostream &operator<<(std::ostream &os, Dataset &ds);
};
#endif //APHW3_Dataset_H