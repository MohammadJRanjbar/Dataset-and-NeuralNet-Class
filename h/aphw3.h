#ifndef APHW3_H
#define APHW3_H
#include "neuralnet.h"
#include <vector>
#include <iostream>
//function prototypes
Dataset loadFuncDataset(const char* filename);
std::vector<Result> testNeuralNets(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr, size_t max_iters, const char* af1 , const char* af2 );
Result findBestNeuralNet(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr , size_t max_iters, const char* af1 , const char* af2 );
void estimateFunction(const char* filename, size_t hidden_neurons_no);
Matrix Sigmoid(Matrix S);
Matrix Sigmoid_prime(Matrix S);
#endif //APHW3_H
