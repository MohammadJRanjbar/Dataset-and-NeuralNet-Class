#ifndef neuralnet_H
#define neuralnet_H
#include "result.h"
#include "aphw3.h"
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
class NeuralNet
{
    private:
        Matrix w1; // Weights of layer 1
        Matrix w2; // Weights of layer 2
        Matrix b1; // Biases of layer 1
        Matrix b2; // Biases of layer 2
        Matrix a1; // Output of layer 1
        Matrix a2; // Output of layer 2
        Matrix n1; // n vector for layer 1
        Matrix n2; // n vector for layer 2
        Matrix s1; // s vector for layer 1
        Matrix s2; // s vector for layer 2 
        const char* af1{ "Sigmoid" };
        const char* af2{ "Sigmoid" };
        size_t hidden_layer_neurons{ 3 };
        double lr{ 0.01 };
        size_t max_iters{ 1000 };
        double min_loss{ 0.01 };
        Dataset dataset;
    public:
        NeuralNet(Dataset dataset, size_t hidden_layer_neurons, const char* f1 = "Sigmoid", const char* f2 = "Linear", double lr = 0.1, size_t max_iters = 10000, double min_loss = 0.01);
        double trainLoss();
        Result fit();
        void backPropagate(Matrix& input, Matrix& target);
        Matrix forwardPropagate(Matrix& input);
        void setW1(Matrix& w);
        void setW2(Matrix& w);
        void setB1(Matrix& b);
        void setB2(Matrix& b);
        Matrix getW1();
        Matrix getW2();
        Matrix getB1();
        Matrix getB2();
        double testLoss();
        void show();
        Matrix weight_initializer(size_t ,size_t);
        friend std::ostream &operator<<(std::ostream &os, NeuralNet &NN);
};
#endif //APHW3_NeuralNet_H