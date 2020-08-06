#include "aphw3.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>      
#include <random>
Dataset loadFuncDataset(const char* filename)
{
	std::ifstream DS;
	//open the file 
	DS.open(filename);
	size_t rows{ 0 };
	size_t cols{ 0 };
	std::string line;
	//find out how may rows and coulmns it has 
	while (std::getline(DS, line))
		rows++;
	DS.clear();
	DS.seekg(0, std::ios_base::beg);
	while (getline(DS, line, ','))
		cols++;

	cols = (cols - 1) / rows + 1;
	//make a matrix
	std::vector<std::vector<double>> Temo;
	std::string A{};
	std::string B{};
	DS.close();
	DS.open(filename);
	//and convert string that is in file to double and save it in the matrix
	for (size_t i = 0; i < rows; i++)
	{
		std::vector<double>D;
		for (size_t j = 1; j < cols; j++)
		{
			getline(DS, A, ',');
			D.push_back(std::stod(A));
		}
		getline(DS, B, '\n');
		D.push_back(std::stod(B));
		Temo.push_back(D);

	}

	DS.close();
	Matrix M{ Temo };
	size_t NC = (M.T()).getSize()[1];
	Matrix Targets{ (M.T()).col(1) };
	Matrix Inputs{ (M.T()).delCol(NC-1) };
	Dataset DST{ Inputs.T(),Targets.T(),70 };
	return DST;
}
std::vector<Result> testNeuralNets(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr, size_t max_iters, const char* af1 , const char* af2 )
{
	//make a vector of Results
	std::vector<Result> RT;
	for (size_t i = 0; i < hidden_neurons.size(); i++)
	{	
		//make a NeuralNet with hidden_neurons[i] and push back it in a vector 
		NeuralNet nn{ dataset ,hidden_neurons[i] };
		//get the result related to that NeuralNet
		Result R{ nn.fit() };
		//and push back it the result vector
		RT.push_back(R);
	}
	return RT;
}
Result findBestNeuralNet(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr , size_t max_iters, const char* af1 , const char* af2 )
{	
	//get the result for every number of hidden_neurons
	std::vector<Result> RT= testNeuralNets(dataset,hidden_neurons,lr,max_iters,af1,af2);
	double TL=RT[0].getTestLoss();
    size_t index{0};
	//in this for we find the index of the result that has the less test_loss
	for (size_t i = 0; i < RT.size(); i++)
	{
		if(TL>RT[i].getTestLoss())
        {
            TL=RT[i].getTestLoss();
            index=i;
        }
	}
	return RT[index];
}
void estimateFunction(const char* filename, size_t hidden_neurons_no)
{
	Dataset D{ loadFuncDataset(filename) };
	//we make a neural network with the dataset
	NeuralNet nn{ D, hidden_neurons_no };
	nn.fit();
	//calculate the biases and weights to estimate
	Matrix Estimated_targets{ D.getTargets() };

	for (size_t i = 0; i < D.getInputs().getSize()[0]; i++)
	{
		for (size_t j = 0; j < D.getInputs().getSize()[1]; j++)
		{   
			//estimate the whole inputs
            Matrix IN{D.getInputs().col(j)};
			for (size_t k = 0; k < D.getTargets().getSize()[0]; k++)
			{
				Estimated_targets[k][j] = nn.forwardPropagate(IN)[k][0];
			}
			
			
		}
	}
	std::cout << std::setw(20) << "NO";
	std::cout << std::setw(20) << "Target";
	std::cout << std::setw(20) << "Estimated"<<std::endl;
	for (size_t i = 0; i < 100; i++)
	{
		std::cout << "*";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < D.getNoOfSamples(); i++)
	{
		for (size_t j = 0; j < D.getTargetDim(); j++)
		{
			std::cout << std::setw(20) << i+1;
			std::cout << std::setw(20) << D.getTargets()[j][i];
			std::cout << std::setw(20) << Estimated_targets[j][i]<<std::endl;
		}
	}
}
Matrix Sigmoid(Matrix S)
{
	for (size_t i = 0; i < S.getSize()[0]; i++)
	{
		for (size_t j = 0; j < S.getSize()[1]; j++)
		{
			S[i][j] = static_cast<double> (1 / (1 + exp(-S[i][j])));
		}
	}
	return S;
}
Matrix Sigmoid_prime(Matrix S)
{
	Matrix ones{ S.getSize()[0],S.getSize()[1],1 };
	Matrix F{Sigmoid(S)};
	Matrix B{ (ones-F) };

	std::vector<double> temp;
	for (size_t i = 0; i < F.getSize()[0]; i++)
	{
		for (size_t j = 0; j < F.getSize()[1]; j++)
		{
			temp.push_back(F[i][j]*B[i][j]);

		}
	}
	//to make a square matrix that to main dimater is sigmoid primes 
	size_t n = std::max (F.getSize()[0], F.getSize()[1]);
	Matrix out{ n,n,0 };
	for (size_t i = 0; i < n; i++)
	{
		out[i][i] = temp[i];
	}
	return out;
}