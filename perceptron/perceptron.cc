#include "perceptron.h"
#include <random>
#include <iostream>

 
/* Classify to 1 if the dot product is bigger than 0 between weight and
	input vector (i.e. the vectors angles is less than 90 degrees) */ 
inline int Perceptron::predict(std::vector<float> const& x)const
{
	return input(x) > 0 ? 1 : -1;
}

/* Returns the dotproduct of weight vector and input vector */ 
float Perceptron::input(std::vector<float> const& x)const
{
	float aggregated_sum {};
	for(unsigned int i {}; i < x.size(); i++) 
		aggregated_sum += x[i] * weights[i + 1];

	return bias + aggregated_sum; 
}

void Perceptron::fit(std::vector< std::vector<float> > const& train_data,
						std::vector<float> const& gold, int epochs)
{
	// Create a random number generator that samples between -1 and 1
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<double> distribution(-1, 1);
	 	
	// Make sure the number of weights is the same number as the number of inputs for each train_data
	for(auto it {begin(train_data[0])}; it != end(train_data[0]); it++)
		weights.push_back(distribution(rng));
	
	for(int i {}; i < epochs; i++)
	{
		for(unsigned int row {}; row < train_data.size(); row++)
		{
			int error = gold[row] - predict(train_data[row]) != 0;
			/* If prediction not correct, update weights */
			if(error != 0)
			{
				/* Loop through weights with an index following */
				// Maybe add some learning factor here to not overfit?
				for(auto p { std::make_pair(begin(weights) + 1, 0)}; p.first != end(weights);
					 ++p.first, ++p.second)
				{		
					(*p.first) -= train_data[row][p.second];
					(*p.first) += gold[row];  
				}
			}
		}
	}
} 
