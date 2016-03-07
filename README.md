# Hybrid-CI-System

##Introduction:

Pattern recognition is an interesting CS problem that requires that objects be described in terms of a
set of measurable features. The selection and quality of the features representing each pattern have a
considerable bearing on the success of subsequent pattern classification. In this assignment we are
given MNIST data(binary images of numeric digits) and we will construct a Hybrid CI system that
performs PCA on the given dataset to reduce the input dimensionality of the data.
GA’s are parallel, iterative optimizers, and have been successfully applied to a broad spectrum of
optimization problems, including many pattern recognition and classification tasks. The problem of
dimensionality reduction is well suited to formulation as an optimization problem. Given a set of -
dimensional input patterns, the task of the GA is to find a transformed set of patterns in an -
dimensional space that maximizes a set of optimization criteria.
Hence, in assignment we will be using Genetic Algorithm (constructed in previous assignment of
this course) to perform dimensionality reduction. MLP is used a classifier and GA will rely on this
classifier to evaluate its performance. 

##Summary of Contruction

We initialize a random GA population. This is done by picking some random features and setting
them to 1(which will be later used for dimensionality reduction).

###Dimensionality Reduction:
 

Next we will minimize the number of features actually used for classification, since each feature used adds to the design
and manufacturing costs as well as the running time of a recognition system. This is done by GA which selects the best
subset of features to be used by the MLP. In our case features means the dimensions of image array.


###Evolution:

GA’s job is to provide us with a search procedure that can be used to explore the space of all subsets of the given feature
set. The problem(to be solved by GA) is to select a subset “d” of the available “m” features in such a way as to not
significantly degrading the performance of the classifier system. An initial set of

features that were somewhat randomly selected will be provided as input. Then in each iteration
CrossOver and Mutation is performed in the hope that it will yield fitter members with time.

Performance Evaluation by using MLP:

The performance of each of the selected feature subsets is measured by invoking a fitness evaluation
function. It is given the initial randomly generated input(population member). It will then perform
feature reduction by counting the number of 1’s and apply a filter to the input data array and then
pass the dimensionally reduced data binary array to the MLP. In the training set given to MLP, an
output set representing positive and negative examples will also be included for which classification
is to be performed.

###Detailed system description:

Since each feature used as part of a classification procedure can increase the cost and running time of
a recognition system we use try to design and implement systems with small feature sets. At the
same time there is a potentially opposing need to include a sufficient set of features to achieve high
recognition rates under difficult conditions.
Feature Selection Using GAs
Genetic algorithms (GAS) are best known for their ability to efficiently search large spaces about
which little is known a priori. This is accomplished by their ability to exploit accumulating
information about an initially unknown search space in order to bias subsequent search into
promising subspaces.
Genetic algorithms are a natural approach for performing subset selection. A feature subset is
represented as a binary Array, with the setting of each member of the array indicating whether the
corresponding feature is used or discarded.
A collection of such binary arrays is called a population. Initially, a random population is created,
which represents different points in the search space. An objective and fitness function is associated
with each member of the population that represents the degree of Fitness(goodness) of that particular
member. Based on the principle of survival of the fittest, a few of the strings are selected and each is
assigned a number of copies that go into the mating pool. Biologically inspired operators like
crossover and mutation are applied on these strings to yield a new generation of strings. the process
of selection, crossover and mutation continues for a fixed number of generations or till a termination
condition is satisfied.

For the feature selection problem, subset of selected features are represented as binary Array of
length L(total number of features in the problem at hand), where “1” in the i position indicates that
the feature is included in the subset, and “0” indicates that the feature is excluded. In order to
evaluate the fitness of an individual (chromosome), the corresponding binary array is fed into an
MLP classifier(figure 3, Raymer et. al., 2000). The size of the input layer is fixed to but the inputs
corresponding to nonselected features are set to 0. The fitness function includes a classifier accuracy

###Fitness Evaluation:

In order to evaluate the fitness of an individual, the selected feature subset is
fed into a neural network classifier of fixed architecture. The transformed patterns are evaluated based upon both their dimensionality, and the classification
accuracy.  The accuracy obtained is then returned to the GA as a measure of the quality of the transformation matrix used to obtain the set of transformed patterns. Using this
information, the GA searches for a transformation that minimizes the dimensionality of the transformed patterns
while maximizing classification accuracy.
