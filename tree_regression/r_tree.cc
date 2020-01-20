#include <cassert>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iostream>

/* Self created library files */ 
// #include <type_printing>
/* ************************** */
#include "r_tree.h"

#define DEBUG 0
#define DEBUG_TRAIN_FUNCTION 0


RegressionTree::~RegressionTree() 
{
    /* Recursive call via ~Node, no need to iterate through tree */ 
    // Node* tmp {root};
    // while(tmp)
    // {
        // if(tmp->left)
            // tmp = tmp->left;
        // else if(tmp->right)
            // tmp = tmp->right;
        // else
        // {
            // Node* tmp_del = tmp;
// 
            // if(tmp_del->split)
            // {
                // delete tmp_del->split;
                // tmp_del->split = nullptr;
            // }
// 
            // tmp = tmp->parent;
            // if(tmp && tmp->left == tmp_del)
                // tmp->left = nullptr;
            // else if(tmp && tmp->right == tmp_del)
                // tmp->right = nullptr;
// 
            // delete tmp_del;
        // }
    // } 

    delete root;
}

template <class T = void>
struct sum_trainingdata_y
{
    constexpr double operator()(double const& lhs, T const& rhs)
    {
        return lhs + rhs.second;
    }
};

double mean_y(Training_data const& train_data)
{
    return std::accumulate(std::begin(train_data), std::end(train_data), 0.0, sum_trainingdata_y<decltype(*std::begin(train_data))>()) / train_data.size();
}


RegressionTree::RegressionTree(unsigned depth, unsigned leaf_k)
    : max_depth{depth}, leaf_threshold{leaf_k}, root{nullptr}
{}

RegressionTree::Node::Node()
    : parent {nullptr}, right {nullptr}, left {nullptr}, split{nullptr}, depth{0}, value {}
{}


RegressionTree::Node::Node(Node* const& p)
    : parent {p}, right {nullptr}, left {nullptr}, split{nullptr}, depth{p->depth + 1}, value {}
{}

RegressionTree::Node::Node(Node* const& p, \
        Split* const& s) : parent {p}, right {nullptr}, left {nullptr}, split{s}, depth {}, value {}
{

    if(p == nullptr)
        std::cout << "This was weird, parent is a nullptr" << std::endl;
    else 
        depth = p->depth + 1;
}

RegressionTree::Node::Node(Split* const& s)
    : parent {nullptr}, right {nullptr}, left {nullptr}, split{s}, depth {}, value {}
{}

RegressionTree::Node::~Node()
{ 
    if(split != nullptr)
    {
        delete split; 
        split == nullptr;
    }
    if(left != nullptr)
    {
        delete left; 
        left == nullptr;
    }
    if(right != nullptr)
    {
        delete right; 
        right == nullptr;
    }
}

RegressionTree::Node::Split::Split(double threshold, int feature)
    : var_idx{feature}, threshold{threshold}
{}

double RegressionTree::predict(double x1, double x2)const
{
    return root -> predict(x1, x2);  
}

double RegressionTree::Node::predict(double x1, double x2)const
{
    /* If there has been a split at this node,
     * check which way to descend for this datapoint to reach leaf node. 
     * If there is no split = leaf node is reached, return value of the node
     */

#if DEBUG == 1
    if(split != nullptr && (left == nullptr && right == nullptr))
    {
        std::cout << "Error: RegressionTree::Node::predict. Tree was built incorrect. "\
            << "split exists but both of the node's left/right ptr's is null" << std::endl;
    }
#endif
    if(split != nullptr && left != nullptr && right != nullptr)
    {
        if(split -> predict(x1, x2))
            return left  -> predict(x1,x2);
        else
            return right -> predict(x1,x2);
    }
    else
        return value; 
}

inline bool RegressionTree::Node::Split::predict(double x1, double x2)const
{
    /* Only covers two variables atm, might make better */ 
    //std::cout << "Going into node::split" << std::endl;
    bool return_bool {};
    if(var_idx == 0)
        return_bool = (x1 < threshold); 
    else if(var_idx == 1)
        return_bool = (x2 < threshold); 
#if DEBUG == 1
    else 
    {
        std::cout << "Error: RegressionTree::Node::Split::predict. \
            Index error, var_idx = " << var_idx << " which is not a valid index." << std::endl;
    }
#endif

    /* ^ = logical xor operation */ 
    //return return_bool ^ inversed;
    return return_bool;
}

void RegressionTree::train(Training_data const& train_data)
{
    assert(train_data.size() != 0 && train_data[0].first.size() != 0);

    /* If we have a fitted tree, remove it */ 
    if(root != nullptr)
    {
        delete root;
        root = nullptr;
    }
    /* Call recursive train function */ 
    train(root, train_data);
}


/* Regression tree pseudo code ~(ish, done by myself)
 * K = Hyperparameter for # of points required to create a separate node
 *
 * For all features
 *     For all datapoints (i)
 *         temp_node = mean value between point[i] and point[i+1]
 *         for all datapoints (j)
 *             calculate residual between temp_node and point j
 *     Select node with smallest residual
 * Select feature with smallest residual
 * Create child_nodes if both have more than K points to them
 */
void RegressionTree::train(Node*& element, Training_data const& train_data, Node* const& parent)
{

#if DEBUG == 1
    std::cout << "New iteration of train started!" << std::endl;
#endif
    /* If we reached max depth, dont create new node */ 
    if(parent && parent->depth + 2 > max_depth)
    {
#if DEBUG_TRAINING_FUNCTION == 1
        std::cout << "Max depth was hit, quitting training" << std::endl;
#endif
        element = nullptr;
        return;
    }

    if(train_data.size() < leaf_threshold * 2)
    {
#if DEBUG_TRAINING_FUNCTION == 1
        std::cout << "Not enough data left to train on for node, returning" << std::endl;
#endif
        element = nullptr;
        return;
    }
    size_t num_features {train_data[0].first.size()};

    //double gini_values[num_features] {};


    /* Keeps track of how the node with the best split looks
     * 0 = residual for all datapoints for that node, 
     * 1 = mean value for the node, 
     * 2 = index for training_data point (might not be needed)
     */ 
    std::vector< std::vector<double> > best_split {}; 
    /* Set all residuals to infinity */ 
    for(unsigned column {}; column < num_features; column++)
    {
        best_split.push_back(std::vector<double> {});
        best_split.back().push_back(std::numeric_limits<double>::infinity());
        best_split.back().push_back(std::numeric_limits<double>::infinity());
        best_split.back().push_back(column);
    }

    for(auto it {train_data.begin()}; it != train_data.end() - 1; it++) 
    { 
        double* mean = new double[num_features] {};
        /* Calculate mean for current datapoint in all features */ 
        for(unsigned i {}; i < num_features; i++) 
        {
            mean[i] = (it->first[i] + (it+1)->first[i]) / 2;
        }
#if DEBUG_TRAIN_FUNCTION == 1
        for(unsigned i {}; i < num_features; i++) 
        {
            std::cout << "The mean in train::best_split was: " << mean[i] << std::endl;
        }
#endif

        unsigned* k_count = new unsigned[num_features] {};
        double* residual = new double[num_features] {};

        /* Classify all points according to previous mean */ 
        for(auto train_sample : train_data)
        {
            for(unsigned i {}; i < num_features; i++)
            {
                residual[i] += std::pow(train_sample.second - mean[i], 2);
                /* Check how many points that will be classified 
                 * into the possibly new left/right leaf nodes */
                if(mean[i] < train_sample.first[i])
                    ++k_count[i];

            }
        }

#if DEBUG_TRAIN_FUNCTION == 1
        for(unsigned k {}; k < num_features; k++)
        {
            std::cout << "The residual for feature " << k << " was: " << residual[k] << std::endl;
            std::cout << "k-count: " << k_count[k] << " for feature " << k << std::endl;
        }
#endif

        /* Classify all data points to check if we have enough k_count to motivate new node */ 
        for(unsigned i {}; i < num_features; i++)
        {
            if(leaf_threshold < k_count[i] && ((train_data.size() - leaf_threshold) > k_count[i]))
            {
                if(residual[i] < best_split[i][0])
                {
                    best_split[i][0] = residual[i];
                    best_split[i][1] = mean[i];
                }
            }
        }
        delete[] mean;
        delete[] k_count;
        delete[] residual;
    }

#if DEBUG_TRAIN_FUNCTION == 1
    for(unsigned i {}; i < num_features; i++)
        std::cout << "best residual: " << best_split[i][0] << " for feature " << i << std::endl;
#endif
    /* Should hopefully have best possible node from each feature */

    auto max = std::max_element(best_split.begin(), best_split.end(), 
            [](auto const& el1, auto const& el2)
            {
            return el1[0] > el2[0]; 
            });

    /* Create a new node with new split */ 


    /* TO DO: 
     * Make the split have the mean value of all datapoints classified to it as value? 
     */
    Node::Split* split = new Node::Split((*max)[1], (*max)[2]);
#if DEBUG == 1
    std::cout << "The residual was: " << (*max)[0] << " the feature chosen for split was: " << (*max)[2] << \
        " and the threshold is: " << (*max)[1] << std::endl;
#endif
    element = new Node(parent, split);


    /* Split training data depending on split and then send it down recursively */ 
    Training_data t1 {};
    Training_data t2 {};
    for(auto train : train_data)
    {
        /* Change to array instead to allow more features */ 
        if(element->split->predict(train.first[0], train.first[1]))
            t1.push_back(train);
        else
            t2.push_back(train);
    }

    train(element->left, t1, element);
    train(element->right, t2, element);
#if DEBUG_TRAIN_FUNCTION == 1
    std::cout << "This is our left ptr: " << element->left << \
        " and this is our right ptr: " << element->right << std::endl;
#endif

    double mean_y_val = mean_y(train_data);
    if(element->left == nullptr && element->right == nullptr)
    {
        delete element->split;
        element->split = nullptr;
        element->value = mean_y(train_data);

#if DEBUG == 1
        std::cout << "This is the mean_y_val: " << mean_y_val << std::endl;
#endif

    }
    else if(element->left == nullptr && element->right != nullptr)
    {
        delete element->right;
        delete element->split;
        element->right = nullptr;
        element->split = nullptr;
        element->value = mean_y(train_data);
    }
    else if(element->left != nullptr && element->right == nullptr)
    {
        delete element->left;
        delete element->split;
        element->left = nullptr;
        element->split = nullptr;
        element->value = mean_y(train_data); 

#if DEBUG == 1
        std::cout << "This is the mean_y_val: " << mean_y_val << std::endl;
#endif

    }   
    // (*max)[0] //= residual;
    // (*max)[1] = mean value for node
    // (*max)[2] = feature 
}
