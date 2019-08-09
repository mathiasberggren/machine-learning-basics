/* Uses the Perceptron Learning Algorithm as proposed by Minsky and Papert (1969) */

#include <vector>

class Perceptron
{

public:
	Perceptron(float threshold = 0)
		: bias{threshold},  weights{}
	{}
	/* Returns a binary classification based on if the dotproduct > threshold */ 
	inline int predict(std::vector<float> const&)const;
	void fit(std::vector< std::vector<float> > const&, std::vector<float> const&, int epochs);
private:
	float bias; /* Threshold */ 
	std::vector<float> weights;	
	float input(std::vector<float> const&)const;
};
