#ifndef result_H
#define result_H
#include "dataset.h"
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
#include <utility>
using namespace std::rel_ops;
class Result
{
    private:
        double train_loss;
        double test_loss;
        size_t no_of_hidden_neurons;
        double lr{};
        size_t iters{};
        const char* af1;
        const char* af2;
    public:
        Result(double train_loss, double test_loss, size_t no_of_hidden_neurons, double lr = 0.01, size_t iters = 10000, const char* af1 = "Sigmoid", const char* af2 = "Linear");
        Result(const Result &r) = default;
        Result(double test_loss);
        double getTestLoss();
        void show();
        friend std::ostream &operator<<(std::ostream &os, Result &R);
        bool operator<(const Result &R)const;
        bool operator==(const Result &R)const;
};
#endif //APHW3_Result_H