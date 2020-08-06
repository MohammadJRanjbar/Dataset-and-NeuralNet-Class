#include "neuralnet.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
NeuralNet::NeuralNet(Dataset dataset, size_t hidden_layer_neurons, const char* f1, const char* f2 , double lr , size_t max_iters, double min_loss)
{
    //initializing values
    this->dataset = dataset;
    this->hidden_layer_neurons = hidden_layer_neurons;
    this->af1 = f1;
    this->af2 = f2;
    this->lr = lr;
    this->max_iters = max_iters;
    this->min_loss = min_loss;
    //Matrix g1{ hidden_layer_neurons,dataset.getInputDim(),1 };
    this->w1 = weight_initializer(hidden_layer_neurons,dataset.getInputDim());
    //Matrix g2{ dataset.getTargetDim(),hidden_layer_neurons,1 };
    this->w2 = weight_initializer(dataset.getTargetDim(),hidden_layer_neurons);
    Matrix e1{ hidden_layer_neurons,1,1 };
    this->b1 = e1;
    Matrix e2{ 1,dataset.getTargetDim(),1 };
    this->b2 = e2;
}
Matrix NeuralNet::weight_initializer(size_t r ,size_t c)
{
    //intilize a random number between 0-1 to weights matrix
    Matrix w{r,c,1};
    srand(static_cast <size_t> (time(0)));
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            w[i][j]=(rand() % 100 + 1) / static_cast <double>(100);
        }
        
    }
    return w;
}
double NeuralNet::trainLoss()
{
    double sum = 0;
    Matrix inputs{ dataset.getTrainInputs() };
    Matrix targets{ dataset.getTrainTargets() };
    for (size_t i = 0; i < targets.getSize()[1]; i++)
    {
        Matrix IN{inputs.col(i)};
        //we use forwardPropagate to get the output of that input
        a2 = forwardPropagate(IN);
        Matrix TN{targets.col(i)};
        for (size_t k = 0; k < TN.getSize()[0]; k++)
        {
            //and substract the real target of estimated target and sum all of them
            sum = sum + pow((a2 - TN)[k][0], 2);
        }
        
        
    }
    //normalize the value of sum
    sum = sum / (inputs.getSize()[1]* targets.getSize()[0]); 
    return sum;
}
Result NeuralNet::fit()
{
    Matrix inputst{ dataset.getTrainInputs() };
    Matrix targetst{ dataset.getTrainTargets() };
    size_t iters = 0;
    std::vector<size_t> temp;
    srand(static_cast <size_t> (time(0)));
    //we use this to train the network randomly
    for (size_t i = 0; i < inputst.getSize()[1]; i++)
    {
        temp.push_back(i);
    }
    //the training stop when the loss is less than min_loss or iterations are more than max_iters
    while(trainLoss()>min_loss && iters<max_iters)
    {
        for (size_t i = 0; i < inputst.getSize()[1]; i++)
        {
            // get a random index that have an element and train with it
            size_t index = rand() % temp.size();
            Matrix IN{inputst.col(temp[index])};
            //first we get the output of the given input
            this->a2 = forwardPropagate(IN);
             Matrix TN{targetst.col(temp[index])};
             //and with backPropagate we update the baises and weights for this sample
             backPropagate(IN,TN);
        }
        iters++;

    }
    Result Re{ trainLoss(),testLoss(), hidden_layer_neurons,lr,iters,af1,af2};
    return Re;
}
void NeuralNet::backPropagate(Matrix& input, Matrix& target)
{
    //with the function strcmp we find out what is the activation function 
        if (std::strcmp(af2,"Sigmoid")==0)
            this->s2 = Sigmoid_prime(n2)*(target - a2)*-2;
        else if (std::strcmp( af2 , "Linear")==0)
            this->s2 = (target - a2)*-2;
        this->w2 = w2 - s2*a1.T()*lr;
        this->b2 = b2 - s2*lr;
        if (std::strcmp(af1,"Sigmoid")==0)
            this->s1 = Sigmoid_prime(n1)*w2.T()*s2;
        else if (std::strcmp( af1 , "Linear")==0)
            this->s1 = (w2.T())*s2;
        this->w1 = w1 - s1*((input).T())*lr;
        this->b1 = b1 - s1*lr;
}
Matrix NeuralNet::forwardPropagate( Matrix& input) 
{
    
    this->n1 = w1*input + b1;
    if (std::strcmp(af1,"Sigmoid")==0)
        this->a1 = Sigmoid(n1);
    else if (std::strcmp( af1 , "Linear")==0)
        this->a1 = n1;
    n2 = w2*a1 + b2;
    if (std::strcmp(af2,"Sigmoid")==0)
        this->a2 = Sigmoid(n2);
    else if (std::strcmp( af2 , "Linear")==0)
        this->a2 = n2;
    return a2;

}
void NeuralNet::setW1(Matrix& w)
{
    this->w1 = w;
}
void NeuralNet::setW2(Matrix& w)
{
    this->w2 = w;
}
void NeuralNet::setB1(Matrix& b)
{
    this->b1 = b;
}
void NeuralNet::setB2(Matrix& b)
{
    this->b2 = b;
}
Matrix NeuralNet::getW1()
{
    return this->w1;
}
Matrix NeuralNet::getW2()
{
    return this->w2;
}
Matrix NeuralNet::getB1()
{
    return this->b1;
}
Matrix NeuralNet::getB2()
{
    return this->b2;
}
double NeuralNet::testLoss()
{
    double sum = 0;
    Matrix inputs{ dataset.getTestInputs() };
    Matrix targets{ dataset.getTestTargets() };
    for (size_t i = 0; i < targets.getSize()[1]; i++)
    {
        Matrix IN{inputs.col(i)};
        a2 = forwardPropagate(IN);
        for (size_t k = 0; k < targets.getSize()[0]; k++)
        {
           sum = sum + pow((a2 - targets.col(i))[k][0], 2);
        }

    }
    sum = sum / (inputs.getSize()[1]* targets.getSize()[0]); 
    return sum;
}
void NeuralNet::show()
{
    std::cout<<"Neural Network:"<<std::endl;
    std::cout << "   " << "No of hidden neurons: " << hidden_layer_neurons << std::endl;
    std::cout << "   " << "Input dimension:" << dataset.getInputDim() << std::endl;
    std::cout << "   " << "output dimension:" << dataset.getTargetDim() << std::endl;
    std::cout << "   " << "Layer 1 activation function : " << af1 << std::endl;
    std::cout << "   " << "Layer 2 activation function : " << af2 << std::endl;
}
std::ostream &operator<<(std::ostream &os, NeuralNet &nn)
{
    //this opreator is friend so it does have access to private variables
    std::cout<<"Neural Network:"<<std::endl;
    std::cout << "   " << "No of hidden neurons: " << nn.hidden_layer_neurons << std::endl;
    std::cout << "   " << "Input dimension:" << nn.dataset.getInputDim() << std::endl;
    std::cout << "   " << "output dimension:" << nn.dataset.getTargetDim() << std::endl;
    std::cout << "   " << "Layer 1 activation function : " << nn.af1 << std::endl;
    std::cout << "   " << "Layer 2 activation function : " << nn.af2 << std::endl;
    return os ;
}