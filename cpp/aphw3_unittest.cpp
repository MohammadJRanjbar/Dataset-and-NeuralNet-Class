#include <limits.h>
#include "aphw3.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include "gtest/gtest.h"
namespace
{
TEST(APHW3Test, DatasetTest)  // 8 points
{
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    size_t no{ds.getNoOfSamples()};
    size_t train_no{ds.getNoOfTrainSamples()};
    size_t test_no{ds.getNoOfTestSamples()};
    EXPECT_EQ(40, no)<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_EQ(28, train_no)<<std::setw(100) <<" ********minus 1 points\n";    
    EXPECT_EQ(12, test_no)<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_EQ(0., ds.getTargets()[0][0])<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_EQ(-2, ds.getInputs()[0][0])<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_TRUE(ds[10][1][0][0] > 0.29 && ds[10][1][0][0] < 0.3)<<std::setw(100) <<" ********minus 1 points\n";
    double a{ds.getTrainInputs()[0][0]};
    ds.shuffle();
    double b{ds.getTrainInputs()[0][0]};
    ds.shuffle();
    double c{ds.getTrainInputs()[0][0]};
    EXPECT_TRUE(a != b || a!= c)<<std::setw(100) <<" ********minus 1 points\n";
    Dataset ds1{ds + ds};
    EXPECT_EQ(80, ds1.getNoOfSamples())<<std::setw(100) <<" ********minus 1 points\n";
}
TEST(APHW3Test, DatasetTest2) //  10 Points
{
    Dataset ds{loadFuncDataset("APHW3Data2.csv")};
    size_t no{ds.getNoOfSamples()};
    size_t train_no{ds.getNoOfTrainSamples()};
    size_t test_no{ds.getNoOfTestSamples()};
    EXPECT_EQ(100, no)<<std::setw(100) <<" ********minus 2 points\n";
    EXPECT_EQ(70, train_no)<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_EQ(30, test_no)<<std::setw(100) <<" ********minus 1 points\n";
    EXPECT_TRUE((ds.getInputs()[0][0] > -5.1) && (ds.getInputs()[0][0] < -4.9))<<std::setw(100) <<" ********minus 1 points\n";
    double a{ds.getTrainInputs()[0][0]};
    ds.shuffle();
    double b{ds.getTrainInputs()[0][0]};
    ds.shuffle();
    double c{ds.getTrainInputs()[0][0]};
    double d{ds.getTrainInputs()[0][10]};
    ds.shuffle();
    double e{ds.getTrainInputs()[0][10]};
    ds.shuffle();
    double f{ds.getTrainInputs()[0][10]};
    EXPECT_TRUE(a != b || a!= c || d!=e || d!=f || e!=f)<<std::setw(100) <<" ********minus 1 points\n";
    Dataset ds1{ds + ds};
    EXPECT_EQ(200, ds1.getNoOfSamples())<<std::setw(100) <<" ********minus 2 points\n";
}
TEST(APHW3Test, DatasetCoutTest) // 10 Points
{
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    std::cout<<ds<<std::endl;
    std::cout<<std::setw(100) <<" ********minus 15 points if dataset is not printed well!********\n";   
}
TEST(APHW3Test, NeuralNetCoutTest)  // 10 Points
{
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    NeuralNet nn{ds, 2};
    nn.fit();
    std::cout<<nn<<std::endl;
    std::cout<<std::setw(100) <<" ********minus 15 points if dataset is not printed well!********\n";   
}
TEST(APHW3Test, DatasetIndexing) // 20 Points
{
    Dataset ds{loadFuncDataset("APHW3Data2.csv")};
    EXPECT_TRUE((ds[2][0][1][0] > .2) && (ds[2][0][1][0] < .21)) << std::setw(100) <<" ********minus 10 points********";
    EXPECT_TRUE((ds[2][1][0][0] > 1.19) && (ds[2][1][0][0] < 1.20)) << std::setw(100) <<" ********minus 10 points********";
}
TEST(APHW3Test, EstimationTest1)  //  15 Points
{
    estimateFunction("APHW3Data2.csv", 4);
    std::cout<<std::setw(100) <<" ********minus 15 points if results are not close!********\n";
}
TEST(APHW3Test, NeuralNetFPTest) // 5 points
{
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    ds.shuffle();
    NeuralNet nn{ds, 2};
    Matrix p{1,1};
    Matrix w1{2, 1}, w2{1, 2}, b1{2, 1}, b2{1, 1};
    b1[0][0] = -.48;
    b1[1][0] = -0.13;
    w1[0][0] = -.27;
    w1[1][0] = -0.41;
    b2[0][0] = 0.48;
    w2[0][0] = 0.09;
    w2[0][1] = -0.17;
    nn.setW1(w1);
    nn.setW2(w2);
    nn.setB1(b1);
    nn.setB2(b2);
    Matrix fp{nn.forwardPropagate(p)};
    EXPECT_TRUE(fp[0][0]>0.445 && fp[0][0]<0.447)<<std::setw(100) <<" ********minus 5 points\n";
}
TEST(APHW3Test, NeuralNetFitTest) // 5 points
{
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    ds.shuffle();
    NeuralNet nn{ds, 2};
    Matrix p{1,1};
    Matrix w1{2, 1}, w2{1, 2}, b1{2, 1}, b2{1, 1};
    b1[0][0] = -.48;
    b1[1][0] = -0.13;
    w1[0][0] = -.27;
    w1[1][0] = -0.41;
    b2[0][0] = 0.48;
    w2[0][0] = 0.09;
    w2[0][1] = -0.17;
    nn.setW1(w1);
    nn.setW2(w2);
    nn.setB1(b1);
    nn.setB2(b2);
    Matrix fp{nn.forwardPropagate(p)};
    Result r{nn.fit()};
    EXPECT_TRUE(r.getTestLoss()<1)<<std::setw(100) <<" ********minus 5 points\n";
}
TEST(APHW3Test, NeuralNetLossTest) //  15 Points
{
    Dataset ds{loadFuncDataset("APHW3Data2.csv")};
    ds.shuffle();
    NeuralNet nn{ds, 4};
    Result r{nn.fit()};
    EXPECT_TRUE(r.getTestLoss() == nn.testLoss())<<std::setw(100) <<" ********minus 10 points\n";
    EXPECT_TRUE(nn.trainLoss()<100)<<std::setw(100) <<" ********minus 5 points\n";
}
TEST(APHW3Test, ResultTest) //  2 Points
{
    Result r1(0.1, 0.4, 5);
    Result r2(0.2, 0.3, 5);
    EXPECT_TRUE(r2<r1)<<std::setw(100) <<" ********minus 2 points\n";
}
} 
