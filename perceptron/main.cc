#include <iostream>
#include <fstream>
#include <sstream>

#include "perceptron.h"

void getIrisY(std::vector<float> & container)
{
	std::ifstream inFile { "../datasets/iris/y.data"};
	std::string s {};
	while(inFile >> s)
	{
		container.push_back( s == "Iris-setosa" ? 1 : -1);
	}	
}

void getIrisX(std::vector< std::vector<float> > & container)
{
	std::ifstream af {"../datasets/iris/a.data"}, bf {"../datasets/iris/b.data"},\
					 cf {"../datasets/iris/c.data"}, df {"../datasets/iris/d.data"};

	std::string a, b, c, d;
	while(getline(af, a) && getline(bf, b) && getline(cf, c) && getline(df, d))
	{
		float fa {stof(a.substr(a.size() - 4))};
		float fb {stof(b.substr(b.size() - 4))};
		float fc {stof(c.substr(c.size() - 4))};
		float fd {stof(d.substr(d.size() - 4))};

		container.push_back(std::vector<float> {fa, fb, fc, fd});
		container.push_back(std::vector<float> {fa, fb, fc, fd});
	}
	af.close(); bf.close(); cf.close(); df.close();
}

int main()
{
	std::vector<float> gold_data {};
	std::vector< std::vector<float> > training_data {};
	getIrisY(gold_data);
	getIrisX(training_data);

	Perceptron p {};
		
	p.fit(training_data, gold_data, 15);


	return 0;
}
