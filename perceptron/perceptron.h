/* 
 Uses the Perceptron Learning Algorithm as proposed by Minsky and Papert (1969)
*/

#include <vector>

class Perceptron
{

public:
	Perceptron(float learn_rate = 0.1)
		: learning_rate{learn_rate},  weights{}
	{}
	float input(std::vector<float> const&)const;
	inline int predict(std::vector<float> const&)const;
	void fit(std::vector< std::vector<float> > const&, std::vector<float> const&, int epochs);
private:
	float learning_rate;
	std::vector<float> weights;	
};
