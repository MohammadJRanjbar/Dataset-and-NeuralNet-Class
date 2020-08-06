#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
Matrix::Matrix(std::vector<std::vector<double>> data)
{
    //initializing the matrix size and data values 
    this->size[0] = std::size(data);
    this->size[1] = data[0].size();
    this->data = data;
}

Matrix::Matrix(size_t m, size_t n, bool ones)
{

    std::vector<double > temp;
    //we made a tempoprary vector 
    //and push back the the one or zero based on the if in the temp vector and then push_back it to the data 
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            //push_back in the temp vector
            if (ones == true)
                temp.push_back(1);
            else
                temp.push_back(0);
        }
        //push back in data
        data.push_back(temp);
    }
    //initializing matrix size values 
    this->size[0] = m;
    this->size[1] = n;
}
std::vector<double> &Matrix::operator[](size_t i)
{
    //this [i] operator return a "i"th vector of a matrix and more
    return data[i];
}
Matrix Matrix::operator+(Matrix& a)const
{
    //with this operator we can add up two Matrices together 
    std::array<size_t, 2> A{ a.getSize() };
    std::vector<std::vector<double>> sum(A[0], std::vector<double>(A[1]));
    for (size_t i = 0; i < A[0]; i++)
    {
        for (size_t j = 0; j < A[1]; j++)
        {
            //add every element of the first matrix to the Corresponding element of the second matrix
            sum[i][j] = data[i][j] + a[i][j];
        }
    }
    Matrix SUM{ sum };
    return sum;
}

Matrix Matrix::operator-(Matrix a)const
{
    //with this operator we can calculate subtraction of two Matrices and more
    std::array<size_t, 2> A{ a.getSize() };
    std::vector<std::vector<double>> sum(A[0], std::vector<double>(A[1]));
    for (size_t i = 0; i < A[0]; i++)
    {
        for (size_t j = 0; j < A[1]; j++)
        {
            //substract every element of the first matrix with the Corresponding element of the second matrix
            sum[i][j] = data[i][j] - a[i][j];
        }
    }
    Matrix SUM{ sum };
    return sum;
}

Matrix Matrix::operator*(Matrix a)const
{
    //this calculate the dot prodcut of two Matrices

    size_t c1 = size[1];
    size_t r1 = size[0];
    std::array<size_t, 2> A{ a.getSize() };
    //for dot product this below equation  for Matrices size should be correct and this if is based on that
    // (m*k) * (k*n)=(m*n)
    //the dot product of a row in the first matrix and a column in the second matrix.
    if (c1 != A[0])
    {
        std::cout << "the dimensions are not correct" << std::endl;
        Matrix zeros{ 1 , 1 , 0 };
        return zeros;
    }

    std::vector<std::vector<double>> C(r1, std::vector<double>(A[1]));
    for (size_t i = 0; i < r1; i++)
    {
        for (size_t j = 0; j < A[1]; j++)
        {
            C[i][j] = 0;
            for (size_t k = 0; k < A[0]; k++)
            {
                // we calculate the dot product of a row in the first matrix and a column in the second matrix 
                C[i][j] += data[i][k] * a[k][j];
            }
        }
    }
    Matrix{ C };
    return C;

}
Matrix Matrix::operator*(double a)const
{
    //for calcuating double multiply to matrix
    std::vector<std::vector<double>> C(size[0], std::vector<double>(size[1]));
    for (size_t i = 0; i < size[0]; i++)
    {
        for (size_t j = 0; j < size[1]; j++)
        {
            C[i][j] = data[i][j] * a;
        }
    }
    Matrix M{ C };
    return M;
}
std::array<size_t, 2> Matrix::getSize()
{
    //initializing the matrix size
    //how many rows it has
    this->size[0] = std::size(data);
    //first row how many cols it has
    this->size[1] = data[0].size();
    return size;
}


Matrix Matrix::delCol(size_t i)
{
    //in this function we delete the "i"th col of the given matrix 
    //make a matrix with one cols less than orginal matrix
    std::vector<std::vector<double>> Mat(size[0], std::vector<double>(size[1] - 1));
    size_t col{ 0 };

    for (size_t k = 0; k < size[0]; k++)
    {
        for (size_t j = 0; j < size[1]; j++)
        {
            if (j != i)
            {
                // giving all the elements except those in "i"th col to the new matrix 
                Mat[k][col] = data[k][j];
                col++;
            }

        }
        col = 0;
    }
    //changing the orginal matrix to the new one
    this->size[1] = size[1] - 1;
    this->data = Mat;
    Matrix DeM{ Mat };
    return Mat;
}

Matrix Matrix::delRow(size_t i)
{
    //in this function we delete the "i"th row of the given matrix 
    //make a matrix with one cols less than orginal matrix
    std::vector<std::vector<double>> Mat(size[0] - 1, std::vector<double>(size[1]));
    size_t row{ 0 };

    for (size_t k = 0; k < size[1]; k++)
    {
        for (size_t j = 0; j < size[0]; j++)
        {
            if (j != i)
            {
                // giving all the elements except those in "i"th row to the new matrix 
                Mat[row][k] = data[j][k];
                row++;
            }

        }
        row = 0;
    }
    //changing the orginal matrix to the new one
    this->size[0] = size[0] - 1;
    this->data = Mat;
    Matrix DeM{ Mat };
    return Mat;
}

Matrix Matrix::col(size_t i)const
{
    //this function we return the "i"th column of the input matrix
    std::vector<std::vector<double>> Mat(size[0], std::vector<double>(1));
    for (size_t j = 0; j < size[0]; j++)
    {
        Mat[j][0] = data[j][i];
    }
    //changing the orginal matrix to the new one
    Matrix cols{ Mat };
    return cols;
}

double Matrix::det()const
{
    double sum{ 0 };
    //only square matrix has determinant
    if (size[0] != size[1])
    {
        std::cout << "The Matrix is not square" << std::endl;
        return 0;
    }
    //if matrix has one element the determinant is that element 
    if (size[0] == 1)
        return data[0][0];


    //for 2x2 matrix determinant is substract of multiply Main diagonal and other diagonal
    else if (size[0] == 2)
        return (data[0][0] * data[1][1] - data[0][1] * data[1][0]);

    /* for matrix bigger than 2x2
    There are various equivalent ways to define the determinant of a square matrix , i.e. one with the same number of rows and columns.
    Perhaps the simplest way to express the determinant is by considering the elements in the top row and the respective minors; starting at the left,
    multiply the element by the minor, then subtract the product of the next element and its minor,
    and alternate adding and subtracting such products until all elements in the top row have been exhausted
    */

    else
    {
        for (size_t i = 0; i < size[1]; i++)
        {

            Matrix temp{ data };
            temp = temp.delRow(0);
            temp = temp.delCol(i);
            sum = sum + data[0][i] * temp.det() * pow(-1, i);

        }


    }
    return sum;

}

Matrix Matrix::T()const
{
    //the number of column of new matrix is the number of rows of old matrix and the new matrix's row is old matrix column
    std::vector<std::vector<double>> transpose(size[1], std::vector<double>(size[0]));
    for (size_t i = 0; i < size[1]; i++)
    {
        for (size_t j = 0; j < size[0]; j++)
        {
            transpose[i][j] = data[j][i];
        }
    }
    Matrix Trans{ transpose };
    return Trans;
}
Matrix Matrix::inv()const
{
    //Inverse of a matrix exists only if the matrix is non-singular i.e., determinant should not be 0.
    std::vector<std::vector<double>> Inverse(size[0], std::vector<double>(size[1], 1));
    Matrix temp{ data };
    temp.getSize();
    double Deter = temp.det();
    //matrix should be square to has inverse
    if (size[0] != size[1])
    {
        std::cout << "The Matrix is not square" << std::endl;
        std::vector<std::vector<double>> M(size[0], std::vector<double>(size[1], 0));
        Matrix NotSquare{ M };
        return NotSquare;
    }
    //if matrix det is zero the matrix doesn't have inverse
    else if (temp.det() == 0)
    {
        std::cout << "The Matrix det is zero" << std::endl;
        std::vector<std::vector<double>> M(size[0], std::vector<double>(size[1], 0));
        Matrix zeros{ M };
        return zeros;
    }
    //if matrix only has one element the inverse is 1 
    else if (size[0] == 1)
    {
        std::vector<std::vector<double>> M(1, std::vector<double>(1, 1));
        Matrix ones{ M };
        return ones;
    }
    //otherwise we calculate the matrix inverse like this
    // A^-1 = adj(A)/det(A)
    else
    {
        for (size_t i = 0; i < size[0]; i++)
        {
            for (size_t j = 0; j < size[1]; j++)
            {
                Matrix temp{ data };
                temp = temp.delRow(i);
                temp = temp.delCol(j);
                Inverse[i][j] = temp.det()* pow(-1, i + j) / Deter;
            }
        }

        Matrix temp{ Inverse };
        temp = temp.T();
        return temp;
    }



}
void Matrix::show()const
{
    for (size_t i = 0; i < size[0]; i++)
    {
        for (size_t j = 0; j < size[1]; j++)
        {
            std::cout << std::left << std::setw(15) << data[i][j];
        }
        std::cout << std::endl;
    }
}

void Matrix::save(const char* Address)
{
    //we made or open a file named OutMatrix
    std::fstream OutMatrix;
    OutMatrix.open(Address, std::ofstream::out | std::ofstream::trunc);
    //and save every data in it
    for (size_t j = 0; j < size[0]; j++)
    {
        for (size_t i = 0; i < size[1]; i++)
        {
            OutMatrix << data[j][i] << ",";
        }
        OutMatrix << std::endl;
    }
}

void Matrix::load(const char* FileAddress)
{
    std::ifstream Grades;
    //open the file 
    Grades.open(FileAddress);
    size_t rows{ 0 };
    size_t cols{ 0 };
    std::string line;
    //find out how may rows and coulmns it has 
    while (std::getline(Grades, line))
        rows++;
    Grades.clear();
    Grades.seekg(0, std::ios_base::beg);
    while (getline(Grades, line, ','))
        cols++;
    cols = (cols - 1) / rows + 1;
    //make a matrix
    std::vector<std::vector<double>> Mat;

    std::string A{};
    std::string B{};
    Grades.close();
    Grades.open(FileAddress);
    //and convert string that is in file to double and save it in the matrix
    for (size_t i = 0; i < rows; i++)
    {
        std::vector<double>D;
        for (size_t j = 1; j < cols; j++)
        {
            getline(Grades, A, ',');
            D.push_back(std::stod(A));
        }
        getline(Grades, B, '\n');
        D.push_back(std::stod(B));
        Mat.push_back(D);
    }

    this->size[0] = rows;
    this->size[1] = cols;
    std::vector<std::vector<double>> data(size[1], std::vector<double>(size[0]));
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            data[i][j] = Mat[i][j];
        }

    }

    this->data = data;
    Grades.close();
}
Matrix::Matrix(std::unique_ptr<std::unique_ptr<double[]>[]> &data, size_t m, size_t n)
{
    this->data = std::vector<std::vector<double>>(m, std::vector<double>(n, 0));
    for (size_t i{}; i < m; i++)
    {
        for (size_t j{}; j < n; j++)
        {
            this->data[i][j] = data[i][j];
        }
    }
    this->size[0] = m;
    this->size[1] = n;
}
Matrix::Matrix() : Matrix(0, 0, 0) {};