/* Made by Mathias Berggren (mathiasberggren1@gmail.com)
 * Inspired by 
 * https://docs.opencv.org/2.4/modules/ml/doc/decision_trees.html
 */

#ifndef REGRESSION_TREE
#define REGRESSION_TREE

#include <utility>
#include <vector>


typedef std::vector< std::pair<std::vector<double>, double> > Training_data;

class RegressionTree
{
    /* Forward declaration */ 
    private:
        struct Node;
        struct Split;
    public:
        //RegressionTree();
        RegressionTree(unsigned depth = 10, unsigned leaf_k = 5);
        RegressionTree(RegressionTree const&) = default;
        ~RegressionTree() { delete root; }
        RegressionTree& operator=(RegressionTree const&) = default;
        double predict(double x1, double x2)const;
        void train(Training_data const&);
    private:
        void train(Node*& element, Training_data const&, Node* const& parent = nullptr);
        // double mean_y(Training_data const& train_data);
        unsigned max_depth; 
        unsigned leaf_threshold;
        Node* root; 

        struct Node
        {
            struct Split
            {
                Split(double threshold, int feature);
                inline bool predict(double x1, double x2)const;
                int var_idx;
                //bool inversed;
                /* Could be boolean function* instead */ 
                double threshold;
                // double quality;      
                // Split* next;
            };

            Node(); 
            Node(Node* const&); 
            Node(Node const&) = default; 
            Node& operator=(Node const&) = default;
            Node(Node* const&, Split* const&); 
            Node(Split* const&); 
            ~Node() ;
            double predict(double x1, double x2)const;
            double gini_impurity(Training_data const&)const;
            Node* parent;
            Node* right;
            Node* left;
            Split* split;
             
            unsigned depth;
            double value;
        };
};



#endif