#include <iostream>
#include "aphw3.h"
#include "Matrix.h"
#include "gtest/gtest.h"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "RUNNING TESTS ..." << std::endl;
    int ret{RUN_ALL_TESTS()};
    std::cout << "Here!"<<std::endl;
    if (!ret)
        std::cout << "<<<SUCCESS>>>" << std::endl;
    else
        std::cout << "FAILED" << std::endl;
    Dataset ds{loadFuncDataset("APHW3Data1.csv")};
    std::cout<<ds<<std::endl;
    return 0;
}
