#include <utility>
#include <vector>
#include <iostream>

#include "r_tree.h"

using namespace std;

Training_data generate_testset(unsigned num_datapoints);
int main()
{


    Training_data train_data = generate_testset(100);
    Training_data test_data  = generate_testset(20);
    RegressionTree tree {};

    tree.train(train_data);

    std::cout << "Training finished, starting predictions: \n\n";
    for(auto t : test_data)
    {
        auto result = tree.predict(t.first[0], t.first[1]);
        std::cout << "What we expected was: " << t.second << " and the predicted value was: " << result << std::endl;
    }
        auto result = tree.predict(31, 50000);
        std::cout << "What we expected was: " << 31 << " and the predicted value was: " << result << std::endl;
    return 0;
}



Training_data generate_testset(unsigned num_datapoints)
{
    Training_data tr {};
    for(double i {}; i < num_datapoints; i++)
    {
        std::pair<std::vector<double>, double> t {};
        t.first.push_back(i); 
        t.first.push_back(i * 2); 
        t.second = i; 
        tr.push_back(t);
    }
    return tr;
}
