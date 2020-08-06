#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
void Dataset::show()
{
    std::cout << "Dataset:" << std::endl;
    std::cout << "   " << "No of sampels: " << no_of_samples << std::endl;
    std::cout << "   " << "Train sampels " << train_inputs.getSize()[1] << std::endl;
    std::cout << "   " << "Test sampels: " << test_inputs.getSize()[1] << std::endl;
    std::cout << "   " << "Input dimensions: " << input_dim << std::endl;
    std::cout << "   " << "Target dimensions: " << target_dim << std::endl;
}
Dataset::Dataset(Matrix inputs, Matrix targets, double percentage)
{
    //initializing values
    this->inputs = inputs;
    this->targets = targets;
    this->percentage = percentage;
    this->no_of_samples = inputs.getSize()[1];
    this->input_dim = inputs.getSize()[0];
    this->target_dim = targets.getSize()[0];
    Train_test_split();
}
void Dataset::Train_test_split()
{
    //n is the number if train inputs 
    size_t n = static_cast <size_t> ((no_of_samples*percentage) / 100);
    Matrix traininputs{ input_dim ,n,0 };
    Matrix traintargets{ target_dim ,n,0 };
    //the remaining data are for test
    Matrix testinputs{ input_dim ,no_of_samples - n,0 };
    Matrix testtargets{ target_dim ,no_of_samples - n,0 };
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < input_dim; j++)
        {
            traininputs[j][i] = inputs[j][i];
            
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < target_dim; j++)
        {
            traintargets[j][i] = targets[j][i];
        }
    }
    size_t Te{ 0 };
    for (size_t i = n; i < no_of_samples; i++)
    {
        for (size_t j = 0; j < input_dim; j++)
        {
            testinputs[j][Te] = inputs[j][i];

        }
        Te++;
    }

    Te = 0;
    for (size_t i = n; i < no_of_samples; i++)
    {
        for (size_t j = 0; j < target_dim; j++)
        {
            testtargets[j][Te] = targets[j][i];

        }
        Te++;
    }
    this->train_inputs = traininputs;
    this->train_targets = traintargets;
    this->test_inputs = testinputs;
    this->test_targets = testtargets;
}
void Dataset::shuffle()
{
    //make a vector of number of sampels
    std::vector<size_t> temp;
    for (size_t i = 0; i < no_of_samples; i++)
    {
        temp.push_back(i);
    }
    std::vector<size_t> temp2;
    //intializing a random seed based on the computer time
    srand(static_cast <size_t> (time(0)));
    Matrix shuffled_inputs{ inputs };
    Matrix shuffled_targets{ targets };
    for (size_t i = 0; i < no_of_samples; i++)
    {
        for (size_t j = 0; j < input_dim; j++)
        {
            //get a random index of that vector
            size_t index = rand() % temp.size();
            //and give to element of that to the inputs and outputs
            shuffled_inputs[j][i] = inputs[j][temp[index]];
            temp2.push_back(temp[index]);
            //and erase that element 
            temp.erase(temp.begin() + index);
        }

    }

    for (size_t i = 0; i < no_of_samples; i++)
    {
        for (size_t j = 0; j < target_dim; j++)
        {
            shuffled_targets[j][i] = targets[j][temp2[i]];
        }

    }
    this->inputs = shuffled_inputs;
    this->targets = shuffled_targets;
    //to intialize train and test data
    Train_test_split();
}
std::vector<Matrix> Dataset::operator[](size_t j)
{
    std::vector<Matrix> temp{ inputs.col(j), targets.col(j) };
    return temp;
}
size_t Dataset::getNoOfSamples()const
{
    return no_of_samples;
} 
size_t Dataset::getNoOfTrainSamples()
{
    return train_inputs.getSize()[1];
}
size_t Dataset::getNoOfTestSamples()
{
    return test_inputs.getSize()[1];
}
size_t Dataset::getInputDim()
{
    return inputs.getSize()[0];
}
size_t Dataset::getTargetDim()
{
    return targets.getSize()[0];
}
Matrix Dataset::getInputs()const
{
    return inputs;
}
Matrix Dataset::getTargets()const
{
    return targets;
}
Matrix Dataset::getTrainInputs()
{
    return train_inputs;
}
Matrix Dataset::getTrainTargets()
{
    return train_targets;
}
Matrix Dataset::getTestInputs()
{
    return test_inputs;
}
Matrix Dataset::getTestTargets()
{
    return test_targets;
}
Dataset Dataset::operator+(const Dataset& dataset)	
{
    //new matrices with no sampels of two datasets
    Matrix new_input{ input_dim ,no_of_samples + dataset.getNoOfSamples(),0 };
    Matrix new_targets{ target_dim ,no_of_samples + dataset.getNoOfSamples(),0 };
    //and initialize the value of each matrices
    for (size_t i = 0; i < input_dim; i++)
    {
        for (size_t j = 0; j < no_of_samples; j++)
        {
            new_input[i][j] = inputs[i][j];
        }
    }

    for (size_t i = 0; i < input_dim; i++)
    {
        for (size_t j = no_of_samples; j < no_of_samples + dataset.getNoOfSamples(); j++)
        {
            new_input[i][j] = dataset.getInputs()[i][j];
        }
    }

    for (size_t i = 0; i < target_dim; i++)
    {
        for (size_t j = 0; j < no_of_samples; j++)
        {
            new_targets[i][j] = targets[i][j];
        }
    }

    for (size_t i = 0; i < target_dim; i++)
    {
        for (size_t j = no_of_samples; j < no_of_samples + dataset.getNoOfSamples(); j++)
        {
            new_targets[i][j] = dataset.getTargets()[i][j];
        }
    }
    //make new dataset with the new inputs and targets
    Dataset DT{new_input,new_targets,percentage};
    return DT;
}
std::ostream &operator<<(std::ostream &os, Dataset &ds)
{
    //this opreator is friend so it does have access to private variables
    os << "Dataset:" << std::endl;
    os << "   " << "No of sampels: " << ds.no_of_samples << std::endl;
    os << "   " << "Train sampels " << ds.train_inputs.getSize()[1] << std::endl;
    os << "   " << "Test sampels: " << ds.test_inputs.getSize()[1] << std::endl;
    os<< "   " << "Input dimensions: " << ds.input_dim << std::endl;
    os << "   " << "Target dimensions: " << ds.target_dim << std::endl;
    return os ;
}