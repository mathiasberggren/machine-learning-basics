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
	float bias {weights[0]};
	float prob {};
	for(unsigned int i {}; i < x.size(); i++) 
		prob += x[i] * weights[i + 1];

	return bias + prob; 
}

void Perceptron::fit(std::vector< std::vector<float> > const& train_data,
						std::vector<float> const& gold, int epochs)
{
	// Create a random number generator that samples between -1 and 1
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<double> distribution(-1, 1);
	 	
	// Make sure the number of weights is the same number as the number of columns in train_data
	for(auto it {begin(train_data[0])}; it != end(train_data[0]); it++)
		weights.push_back(distribution(rng));
	
	for(int i {}; i < epochs; i++)
	{
		for(unsigned int row {}; row < train_data.size(); row++)
		{
			/* 
				Calculate the weight change by multiplying the error with the learning rate
				update = learn_rate * (y - ŷ)  
				this gives a weight change of 0 if the prediction was correct
			*/
			float weight_update { learning_rate * (gold[row] - predict(train_data[row])) };
			/* Loop through weights with an index following */
			for(auto p { std::make_pair(begin(weights) + 1, 0)}; p.first != end(weights);
					 ++p.first, ++p.second)
				/* Update the weight with weight_update * the training sample */
				(*p.first) += weight_update * train_data[row][p.second];
			/* Update the bias to the update */ 
			weights[0] = weight_update;	
		}
	}
} 
