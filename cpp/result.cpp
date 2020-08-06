#include "result.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
Result::Result(double train_loss, double test_loss, size_t no_of_hidden_neurons, double lr , size_t iters, const char* af1 , const char* af2 )
{
    //initializing values
    this->train_loss = train_loss;
    this->test_loss = test_loss;
    this->no_of_hidden_neurons = no_of_hidden_neurons;
    this->lr = lr;
    this->iters = iters;
    this->af1 = af1;
    this->af2 = af2;
}
Result::Result(double test_loss) : Result(-1, test_loss, 0){};
double Result::getTestLoss()
{
    return test_loss;
}
void Result::show()
{
    std::cout << "Result:" << std::endl;
    std::cout << "   " << "Train loss:" << train_loss << std::endl;
    std::cout << "   " << "Test loss:" << test_loss << std::endl;
    std::cout << "   " << "No of hidden neurons:" << no_of_hidden_neurons << std::endl;
    std::cout << "   " << "Layer 1 activation function : " << af1 << std::endl;
    std::cout << "   " << "Layer 2 activation function : " << af2 << std::endl;
}
std::ostream &operator<<(std::ostream &os, Result &R)
{
     //this opreator is friend so it does have access to private variables
    os << "Result:" << std::endl;
    os << "   " << "Train loss:" << R.train_loss << std::endl;
    os << "   " << "Test loss:" << R.test_loss << std::endl;
    os << "   " << "No of hidden neurons:" << R.no_of_hidden_neurons << std::endl;
    os << "   " << "Layer 1 activation function : " << R.af1 << std::endl;
    os << "   " << "Layer 2 activation function : " << R.af2 << std::endl;
    return os ;
}
bool Result:: operator<(const Result &R)const
{
    //this opreator return the boolean result of comparison of the two test_loss of two "Result"
    return  test_loss<R.test_loss;
}
bool Result:: operator==(const Result &R)const
{
    //this opreator return the boolean result of comparison of the two test_loss of two "Result"
    return  test_loss==R.test_loss;
}
