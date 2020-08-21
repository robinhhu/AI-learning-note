# AI learning note for Machine Learning course provided by Andrew-Ng in Coursera
## What is Machine learning?
Machine learning is the science of getting computers to learn without being explicitly programmed.
## Supervised learning and unsupervised learning:
### Supervised learning:
In every example in our data set, we are told what is the “correct answer”. The task of the algorithm is to give a more correct answer
#### Two common problems related to supervised learning:
* Regression: to predict a continuous-valued output
* Classification: to predict a discreet valued output
### Unsupervised learning: 
A learning setting where the algorithm is given a ton of data and just asked to find structure in the data for us.

## SOME NOTATION IN THE COURSE:
* m = Number of training examples
* x = “input” variable/features(e.g. size of the house)
* y = “output” variable/“target” variable(e.g. price of the house)
* (x,y): a single training example
* (x^(i),y^(i)): the i^th training example

## How supervised learning algorithm works:
Training Set  ➡️feed to➡️  Learning Algorithm  ➡️to output➡️  h(hypothesis function)(A function that maps from x’s to y’s)

For linear regression with one variable(x), or called univariate linear regression, h is represented as h_theta (x) = theta_0 + theta_1 x

## Cost function: 
Usage: help to figure out how to fit the best possible straight line to our data.

h_theta (x) = theta_0 + theta_1 x is the hypothesis function in linear regression, where theta_0 and theta_1 are parameters. Different parameters lead to different hypothesis function. So the point is how to choose them.
 
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/cost%20function.jpg)
 
## Gradient descent:
Usage: to minimize the cost function J.  

![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent1.jpg)

### How to apply gradient descent algorithm into our cost function(linear regression model)?  
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent2.jpg)

The type of gradient descent we used is called “Batch” Gradient descent. It means each step of gradient descent uses all the training examples. (i.e. m in sigma)

## Some Basic:
### Matrix
Matrix is a rectangular array of numbers. The dimension of Matrix is the number of rows * number of columns(e.g. 4 * 3 matrix)

Matrix Elements(entries of matrix): A = [1402, 191] A_ij = “i,j entry” in the i^th row, j^th column. So A_11 = 1402, A_12 = 191.

#### Identity matrix: 
Denoted I(or I_i*i). For any matrix A, A * I = I * A = A.

#### Matrix inverse: 
If A is an m * m matrix, and if it has an inverse, A * A^-1 = A^-1 * A = I. (Only square matrix has an inverse. Matrices don’t have an inverse are “singular” or “degenerate”

#### Matrix transpose: 
A is an m*n matrix, and let B = A^T. Then B is an n*m matrix and B_ij = A_ji.

### Vector
Vector is a matrix with one column(e.g. y = [460; 232; 315; 178] is a 4 * 1 matrix or a 4-dimensional vector). Y_i = i^th element. Can be 1-indexed or 0-indexed. By convention, people use uppercase notation to refer to matrices and lowercase notation to refer to vectors.

### Some algorithm:
* Matrix addition: can only add two matrices of the same size(Add together at every position)
* Scalar multiplication: multiply every position.
* Matrix * vector: to get y_i, multiply A(A is the matrix)’s i^th row with elements of vector x, and add them up.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/matrixvectormul.png)
* Matrix * matrix: the i^th column of the matrix C(result) is obtained by multiplying A(one of the matrices be multiplied with the i^th column of B(another matrix).
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/matrixmatrixmul.png)
#### properities:
1. A * B != B * A(not identity matrix)(not commutative)
2. A * (B * C) = (A * B) * C(associative)

## Multivariate linear regression:
Multiple features(variables) x_1, x_2...

### NOTATION:
n = number of features

Then x^(2) can be an n-dimensional matrix. That is the same as x^(2)_j, where7 j is greater than 1 and less than n.

Our hypothesis becomes h_theta(x) = theta_0 + theta_1*x_1 + ..... theta_n * x_n
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/multivariatelinearregression.png)

### Gradient descent for linear regression for multiple variables:
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent%20for%20mv.png)
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent%20for%20mv2.png)
#### How to make gradient descent work well:
##### Feature scaling: 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/featurescaling.png)
make sure features are on a similar scale(so that gradient descent can converge much more faster): Get every features into approximately in range between -1 and 1 by divide or multiply a number.

There are two ways to implement it. We can divide by the max possible value. For example, x in range 0 to 1000, then divide x by 1000. Or use mean normalization. That is, replace x with (x-u)/s to make features have approximately zero mean, where u is the average value and s is the range of the value.

##### “Debugging”: 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/plot.png)
make sure gradient descent is working correctly. The cost function should decrease after every iteration. Examine by plotting above or automatic convergence test(declare convergence if cost function decrease by less than a threshold) to declare convergence.

Learning rate(alpha): If is too small, then the convergence may be slow; if it is too large, the cost function may not decrease on every iteration or may not converge. Just try a range of value, from 1, 0.3, 0.1, 0.03, 0.01... And pick the largest possible value.

## Features: 
we can improve our features and the form of our hypothesis. E.g. can multiply two features together. Or can use polynomial regression(because our h function need not be linear) to make it quadratic, cubic or square root function.

E.g., create additional features based on x_1, to get new feature x_2 to be x_1 square. And if you choose your features this way then feature scaling becomes very important.

## Normal equation: 
method to solve for theta analytically. (Another way to get theta)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/normalequa.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/normalequapic1.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/normalequapic2.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/normalequa2.png)
### What if X^T*X is non-invertible(singular/degenerate)? 
* What happened: 1. Redundant features(linearly dependent) 2. Too many features(e.g. m <= n)
* Solution: 1. see if have redundant features, 2. Delete some features or use regularization

## Use vectorization to speed up:
Improve efficiency. Make numeric computation to vector computation(build-in matrix operation)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/vectorization.jpg)
A more concrete example:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/example.png)

## Classification problem:
variable y takes on two classes 0 and 1. 0 is the negative class and 1 is the positive class. As an convention, 0 represents the absent of sth while 1 represents the existence of sth.

Can threshold classifier output at 0.5 if apply linear regression. That is, if h_theta(x) >= 0.5, predict y=1, and else versa. So we need a classification algorithm called logistic regression so that h_theta(x) is greater than 0 while less than 1.

### Logistic regression: 
want 0 <= h_theta(x) <= 1 
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/logistic%20regression.jpg)
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/interpretation.jpg)

### Decision boundary:
Suppose predict y=1 if h_theta(x) >= 0.5 and y=0 if h_theta(x) < 0.5
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/decisionboundary.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/decisionboundary.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/non-linear%20db.jpg)

### Cost function:
We have a training set of m examples {(x^(1),y^(1)),...(x^(m),y^(m))}, x belongs to [x_0, x_1...x_n], x_0 = 1, y belongs to {0,1}, and the hypothesis function, how to choose parameters theta?

Using cost function of linear regression can not get a convex function, so gradient descent can not be used in logistic regression. Need to come up with a new cost function that is convex so that we can develop gradient descent upon it(guarantee to find the global minimum).  
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/cost%20function%20for%20logistic%20regression.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/cost%20function%20for%20logistic%20regression2.jpg)

### Advanced optimism for quicker find the theta:
Have J(theta), and partial derivative of j(theta), we can not only use gradient descent, but also conjugate gradient, BFGS and L-BFGS. These algorithms do not need to manually pick alpha, and often faster than gradient descent. But they are more complex also.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/advancedopt.png)
(Octave is index-one a based)
You only need to plug in additional code to compute jVal(J(theta)) and gradient(gradient(1)...gradient(n+1)), so that they can be used in advanced build-in functions.

### Multi-class classification problems(y = 1, y = 2, y = 3, y = 4)
One-vs-all: separate one class with all another classes one by one.

That is, h_theta^(i)(x) = P(y = i|x;theta)  for (i = 1,2,3).Train a logistic regression classifier h_theta^(i)(x) for each class i to predict the probability that y = i. On a new input x, to make a prediction, pick the class i that maximizes the classifiers.

## The problem of overfitting:
The term “undefit” and “High bias” means the hypothesis function does not fit data set well.

“Overfit” and “high variance” means we have too many features, the learned hypothesis may fit the training set very well (J(theta) almost equal to 0), but fail to generalize to new examples(predict prices on new examples)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/overfitting.png)

### What can we do to address it?
1. Reduce number of features(manually select features to keep or model selection algorithm)
2. Regularization(keep all features, But reduce magnitude/ values of parameters theta. Work well when we have a lot of features and each contribute a little to the result)

## Regularization:
Small values for parameters theta_0,..theta_n, corresponds to “simpler” hypothesis and less prone to overfitting.

### Cost function(modified):
We can reduce the weight that some of the terms in our function carry by increasing their cost. So we can modify our cost function:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/modifiedcf.jpg)
Regularized linear regression:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/regularizedlr.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/regularizedlr2.jpg)
Regularized logistic regression:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/regularizedlogisticre.jpg)

## Neural Network: Algorithm that try to mimic the brain.
### Neural model: logistic unit
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20model.jpg)
### Neural network : group of neons 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20network.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20network.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20network2.png)

#### Example of neural network:
AND
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/and.png)
OR
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/or.png)
#### Complex non-linear hypothesis:
XNOR
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/xnor.png)

### Multi-class classification:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/multiclassclassification.png)
How to represent the training set? 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/multiclassclassification2.png)
Where x^(i) is image of four classes, y^(i) is the classifier that image corresponds to.

## Fitting parameters in neural network:
### NOTATION: 
* L = total no. Of layers in network
* S_l = no. of units(not counting bias unit) in layer l
* K = number of output units/classes

![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20network%20cf1.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/neural%20network%20cf2.png)

### Algorithm to minimize cost function: back propogation
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/algorithm%20to%20minimize%20cf.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/bpa.png)

### Implementation:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/ao.png)
How to unroll D matrix into vectors so that they can fit into advance functions
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/implementationexample.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/implementationexample2.png)

### Gradient checking:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20checking.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gc%20note.png)

### Initial value of theta:
For gradient descent and advanced optimization, need initial value for theta. All zero do not work for neural network, while it does work for gradient descent.

How to deal with it? —random initialization 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/randomini.png)

### Summary of neural network learning algorithm:
1. Pick a network architecture(how many hidden layers(usually the more the better), how many units in each layer(usually the same in every hidden unit, decided by feature pattern))
2. Training a neural network(randomly initialize weights, implement forward propagation to get h_theta(X^(i)) for any X^(i), compute cost function, backdrop to compute partial derivatives, use gradient checking(then disable gradient checking code), finally use gradient descent or advanced optimization method to get theta)

## Evaluating a learning algorithm:
### Decide what to do next:
Suppose you have implemented regularized linear regression to predict housing prices. however, when you test your hypothesis on a new set of houses, you find that it makes unacceptably large errors in its prediction. What should you try next?
* get more training examples
* Try smaller sets of features
* Try getting additional features
* Try adding polynomial features(X_1^2, X_2^2, X_1*X_2, etc)
* Try decreasing lambda
* Try increasing lambda

Rather than try above methods randomly, we have simple way to decide in what way we spend our time pursuing.

Introduce how to evaluate learning algorithm and machine learning diagnostic(a test that can run to gain insight what is/isn’t working with a learning algorithm, and gain guidance as to how best to improve its performance)(diagnostics can take time to implement, but doing so can save a lot of time afterwards)

### Evaluate hypothesis:
#### How to tell if a hypothesis is overfitting? 
Split our dataset into two portions. One for training set(70%), another for test set(30%), where data should be randomly shuffled before training.
Training procedure:
1. Learn parameter theta from training data(minimizing training error J(theta))
2. Compute test set error. Can compute J_test(theta) as usual for linear regression, or use misclassification error(0/1 misclassification error) for logistic regression
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/testerror.jpg)

### Model selection(How to choose polynomial terms, lambda, etc):
Once parameters theta_0, theta_1.. theta_4 were to fit to some set of data(training set), the error of the parameters as measured on that data(J_theta) is likely to be lower than the actual generalization error.
So the J_test(theta) we got from our test set may just fit our test set and can not be generalized well.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/twosets.png)
So instead we are going to split our data set into 3 pieces. 60% for training set, 20% for cross validation set, 20% for test set.
Similarly, we compute training error, cross validation error and test error for each set.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/threesets.png)
We can then get the generalization error for our model.

### Diagnose problems(Bias vs Variance(under fitting vs overfitting)):
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/errorplots.png)
So how to figure out whether it is a bias or variance?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/diagnose.png)

### How regularization affect bias/variance?
Large lambda(lambda = 100000), all theta = 0, high bias(underfit)

Small lambda(lambda = 0), high variance(overfit)

Intermediate lambda, “just right”
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/trainlambda.png)
Define J without using regularization, then select lambda
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/selectlambda.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/ftolambda.png)
The procedure:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/procedure.png)

### Learning curve: figure out whether your hypothesis suffer from bias or variance or both
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/learning%20curve.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/lchb.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/lchv.png)

### What to do?
* get more training examples.         ——fix high variance problem
* Try smaller sets of features.          ——fix high variance problem
* Try getting additional features.        ——fix high bias problem
* Try adding polynomial features(X_1^2, X_2^2, X_1*X_2, etc).       ——fix high bias
* Try decreasing lambda.      ——fix high bias
* Try increasing lambda.       ——fix high variance


### What about problems in neural network?
Using single hidden layer is a reasonable default, but if want to choose the number of hidden layers, try training neural network with various hidden layers and see which performs best on the cross-validation sets

## System design:
First example:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/spamexample.png)
And we have a lot thing to do, here are some examples
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/spamoptions.png)
The question is how to choose from those options that can save your most of the time

### Recommended approach:
* start with a simple algorithm that you can implement quickly. Implement it and test  it on your cross-validation data.
* Plot learning curves to decide if more data, more features, etc. are likely to help
* ERROR ANALYSIS: Manually examine the examples(in cross validation set) that your algorithm made error on. See if you spot any systematic trend in what type of examples it is making errors on(Know current shortcomings quickly). For example, categorize misclassified emails based on what type of email it is(pay particular attention to those categories that misclassified frequency. Add features to them to help) and what cues you think would have helped the algorithm classify correctly(For example, after manually examination, most of the misclassified emails contain deliberate misspellings. So misspellings is something that you should try a lot of time to write algorithm to detect).

### Evaluate your algorithm using numerical evaluation:
Error analysis may not be helpful for deciding if this is likely to improve performance. Only solution is to try it and see if it works.

Need numerical evaluation(e.g., cross validation error) of algorithm’s performance with and without the solution.

### One exception: skewed classes
Where the ratio of positive to negative examples is very close to one of the two extremes(The number of positive example much much smaller than negative or vice versa). That is the case called skewed classes. We have a lot more examples from one class than the other class. So that a function always output 1 or 0 can actually do better than our algorithm.

And that’s the problem for using classification error or classification accuracy as our evaluation matrix. So we need another evaluation matrix: precision recall.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/precisionrecall.png)
In this case when we have algorithm y=0, the recall is 0 because the true positive is always 0.

### Trade-off between precision and recall:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/tradeoff.png)
By switching between threshold we can actually get different precision and recall.
So how do we decide which algorithm is the best?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/Fscore.png)
Try different threshold and pick the one which gives you the biggest f score on your cross-validation set.

### How many data?
Assume feature x belongs to R^n+1 has sufficient information to predict y accurately, and we use a learning algorithm with many parameters, then J_train^(theta) will be small(low bias). And if we use a very large training set(unlikely to overfit)(low variance), J_train(theta) will approximately equal to J_test(theta). So J_test(theta) will also be small.

## Support vector machine: Alternative view of logistic regression
The cost example above is the amount each example contribute the final cost(The cost for each set).
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/alternativeforlogisticregre.png)
We replace cost function at y = 1 and y = 0 with cost_1(z) and cost_0(z).
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/svmcostfun.png)
We get rid of constant 1/m, and replace lambda with C to get the cost function for SVM. So minimize the function at the bottom can get the parameters learned by SVM.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/whatweget.png)
We can get theta after implementing the cost function at top. And rather than output the probability like logistic regression, our hypothesis is to simply output 1 if theta^transpose*X is greater or equal to 0, and 0 otherwise.

### Large margin intuition:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/largemargin.png)
We want more than Theta^transpose*X greater than 0, we want it to be greater than 1 so that the output result is safe.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/lm.png)
SVM always gives us larger margin thus greater solutions. That is to separate the positive and negative example with as big a margin as possible. But it is also sensitive to any outliners(And that’s the case when C is very large(lambda very small). If C is set to value smaller, the algorithm can ignore few outliners).

Kernels:
How to define features?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel1.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel2.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel3.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel4.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel5.png)
how to choose landmark?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel6.png)
How to combine kernel with SVM?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel7.png)
Don’t need to worry about implementation details at the bottom.

How to choose C? The bias and variance trade-off:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernel8.png)

Using SVM:
Use software package(e.g. liblinear,libsvm...) to solve for parameters theta.

Need to specify:
* Choice of parameter C
* Choice of kernel(similarly function)(E.g. No kernel, “linear kernel”, predict “y=1” if theta^transpose*X >= 0(when n large, m small))Or Gaussian kernel, also need to choose sigma square(n small, m large).

There are something to do if choose to use Gaussian kernel
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kernelfun.png)
Can also choose other kernels. But not all are valid.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/choicesforkernel.png)

Multi-class classification:
Many SVM package already have build-in multi-class classification functionality. Otherwise, use one-vs-all methods(Train K SVMs, one to distinguish y = i from the rest, for i = 1,2, ..., K), get Theta^(1),... pick class i with largest (theta(i)^transpose)*X

Logistic regression vs. SVMs:
Logistic regression and SVM without kernel is similar. SVM is powerful with kernels.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/comparison.png)

# Unsupervised learning:
We are given data that have no labels associate. Training set {X^(1),X^(2)...}. we just ask the algorithm find some structure in the data for us.

## Clustering:
### K-means algorithm:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/kmeans.jpg)
An example for non-separated clusters:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/non-separated.png)

#### Optimization objective:
*	help to debug the learning algorithm and make sure k-means is running correctly
*	Help k-means find better costs and avoid the local Ultima

Just a reminder that
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/reminder.jpg)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/k-meansoptimization.png)

#### Random initialization:
1.	Should have K < m
2.	Randomly pick K training examples
3.	Set mu_1,mu_2,...mu_k equal to these K examples

Random initialization can also be dangerous
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/dangerouslocal.png)
Solution: try multiple random initialization
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/manytimes.png)
It works when k is not large. For the case where k >= 10, it is likely that you can get decent solution for the first time

#### Choose the number of clusters:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/elbow.png)
The picture on the left is ideal, but most of the time we get ambiguous picture on the right. So not a good choice. 
The later purpose:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/laterpurpose.png)

## Another type of unsupervised learning problem: dimensionality reduction
### Motivation:
1.	Data compression(save space, speed up learning algorithm)
Reduce data from 2D to 1D: if we have data that have two dimensions, one in cm and another in inches. Since we can easily change from inches to cm, the data is highly redundant and we can reduce the data into 1D. What we do is to draw a line and project data into the line as follows:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/datacompression.png)
Another example from 3D to 2D:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/datacompression2.png)
When all of data lies roughly on the plane like so, we project data into a plane.
2.	Data Visualization
given huge number of data, can we examine data in a better way?
———reduce data from 50D to 2D, and then we can plot dot on figure(summarize 50 features and plot the figure in 2D. But it doesn’t astride a physical meaning to these new features. So it is up to us to figure out, roughly what these features means)

### Algorithm for dimensionality reduction: Principle component analysis
fine the line or plane to project data on
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/pca.png)
PCA is not linear regression:
Linear regression is to find a line that fit the data. And we evaluate using vertical distance between data and output. Where in contrast, in PCA we try to minimize the shortest distance between data and our line. 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/pca2.png)

#### How to implement:
1. Data preprocessing(feature scaling/mean normalization
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/datapre.png)
2. What we want PCA to compute is u^(1), u^(2) and vector z
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/pcapre.png)
3. The procedure
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/pca3.png)
We first compute covariance matrix, stores it to sigma(n*n), and then use svd function to compute evigenvectors. The output U is also a n*n matrix, and to reduce to k dimensions, we only need to pick the first k vectors from U output.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/pca4.png)
What we want is z.

To summarize:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/summary.png)

#### How to get back?
From low dimensions to higher. For example, we get 1D value and now we want approximation to original 2D value.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/inverse.png)

#### Choosing k:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/choosingk.png)
Notice that when we compress from x(R^n) to z(R^k), where k < n, some data lost. So when we change from z(R^k) to x_approx(R^n), we only get back part of the data that based on the plane. It is slightly different with the previous x where we have some extra information above/below the plane.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/simplerway.png)
Don’t need to call on svd function again and again. Only call svd once and increase k again and again until the smallest value of k for which 99%(E.g.) of variance retained.

#### PCA application:
1.	Supervised learning speedup(choose k by % of variance retained）
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/speedup.png)
2.	Reduce memory/disk needed to store data(choose k by % of variance retained)
3.	visualization（k = 2 or k = 3)
4.	It is bad use of PCA to prevent overfitting.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/baduse.png)
Since when reduce dimensions, PCA may lose some valuable information while regularization process also keeps an eye on y.
5.	Another misuse
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/baduse2.png)
Only when implement the raw data and found too slower/memory requirement too high. Then implement PCA.

## Unsupervised learning: Anomaly Detection:
### Motivation: Examine aircraft engine.
We examine aircraft engine features like heat generated and vibration intensity based on a dataset. And then when we get a new engine X_test, we can decide whether it needs further examination.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/examineaircraft.png)
In detail, we have a serious of features given by normal aircrafts and we want to estimate whether a new engine is anomalous based on algorithm. That is, for example, given a unlabeled training set, we are going to build a model for p(x)(probability of x where x are features). Having build the model we can say that if p(X_test) < epsilon, then we flag this as anomaly.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/anomaly.png)
An application: detect unusual users, manufacturing use such as examine aircraft and monitoring computer in a data center.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/application.png)

### Gaussian distribution(normal distribution):
the standard deviation decides the width of the figure.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/normaldis.png)
Examples of Gaussian distribution:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/disexample.png)
The shaded area is always integrated to 1. So when the width of the distribution decrease, its height will increase.
When given dataset(x^(1),x^(2)...,x^(m)) x^(i) belongs to R, how to estimate parameters mu and sigma square?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/estimate.png)

Apply Normal distribution to anomaly detection algorithm:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/densityestimation.jpg)

### Anomaly detection algorithm:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/ada.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/examplepx.png)

### Evaluate the anomaly detection algorithm:
Based on evaluation to evaluate and decide whether to use some feature or the value of epsilon.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/evaluation.png)
Specific example:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/evaluationexample.png)
How to split labeled and unlabeled examples(6000 training set are all unlabeled). We use 60% 20% 20% for good engines, and put anomalous just in the cv set and test set.
How to implement?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/implementdetail.png)
We treat all training set as normal. And since the data is very skewed(anomaly is rare), we need evaluation metrics.

#### Anomaly detection vs. supervised learning
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/avss.png)
Key difference is that we have a small number of positive examples and it is hard for supervised algorithm to learn much from such limited examples.

### Choosing what features to use?
1. Use hist(x) to plot histogram(hist(x,50) for 50 bins).
And to make algorithm work better, we change some non-Gaussian features Gaussian(even thought they also work well when they are not Gaussian).
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/non-g.png)
xNew = x.^0.5 can be a new feature.
2. Error analysis
If we have an anomalous example buried among normal examples, we can manually examine that example thus come up with another feature x_2
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/errorana.png)
3. Choose features that might take on unusual large or small values in the event if an anomaly.
E.g. an anomaly can cased by high CPU load and network traffic(normally these are linearly related), we can create a feature CPU load/net work traffic that corresponds to unusual combination of values of features.

### Modified version:
Previous algorithm fails to detect that CPU and memory have sort of linear relation. The green data indicated there have neither too low CPU load nor high memory use even though it is actually an anomaly. So how can we fix the problem that the algorithm treat examples in the same circle as equal possibility?
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/problemforad.png)
Multivariate Gaussian distribution:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/multivariategd.png)
Some examples:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/vary.png)
We can change mu to alter the center of the image and change sigma to alter the shape. Larger sigma(1,1) makes x_1 smoother, larger sigma(2,2) makes x_2 smoother. It is also worth noticing that change the diagnose(sigma(1,2) and sigma(2,1)) to positive value makes x_1 and x_2 positively related(y = x) while change them to negative makes them negatively related(y = -x).
#### Apply multivariate Gaussian distribution:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/details.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/details2.png)

### Comparison between origin model and multivariate model:
The main difference is that the original model is always symmetric while the multivariate one has some angle.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/comparisonmodels.png)
Advantage/disadvantage:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/commondels.png)
Normally we use original model, even manually create unusual features. However when m is very large we may also consider multivariate Gaussian. When the matrix is singular thus non-invertible, check redundant data.

## Recommender system: application of machine learning
A few settings that you can have algorithm just to learn what feature to us
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/movies.png)
We want an algorithm that automatically fill in missing values for us so that we can predict what movies the user has not seen yet and recommend it to users.

### First approach: content based recommendations
Assume we have features e.g. how romantic is this movies, how much action is in this movie? Etc.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/cbr.png)
Apply a different copy of essentially linear regression for each user. Each user have a different linear function.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/problemformulation.png)
Note that for the equation at the bottom, the constant 1/m is canceled since it has nothing to do with the result.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/cbroptimization.png)
Add extra summation for all users.

And the gradient descent:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gd.png)

### Second approach: collaborative filtering
The case where we don’t know the content.

It has the property of feature learning, that is, it starts to learn for itself what features to use

So assume that we can get to users and each user tells us what is the value of theta_j for us. If we get the parameter theta of each user, we can get values x_1 and x_2 for each movie.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/thetatox.png)
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/algorithm.png)

Theta and x just like egg and chicken, we want chicken to lie eggs and eggs to incubate chickens. One way to optimize is to guess an initial value for theta first(collaborative filtering algorithm).
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/collaborative.png)
It is only possible when each user rates multiple movies and each movie is rated by multiple users.

### Collaborative filtering algorithm:
Actually there is no need to do iterations, we can compute theta and x simultaneously by combining optimization objective together.
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/collaborative2.png)
And we get rid of X_0 and theta_0 because the algorithm can get 1 by themselves if needed.
In summary, the procedure of collaborative filtering algorithm is as followed:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/collaborative3.png)
Notice that we already get rid of x_0, so there is no need to break out a special case for k = 0 in gradient descent.

#### Vectorization implementation(low rank matrix factorization):
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/lrmf.png)

### Application: find related movies:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/recommend.png)

### Technique: Mean normalization:
An unusual example:
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/norate.png)
Mean normalization fix the problem: 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/meannormalization.png)
The idea is that if an user who hasn’t  rate anything, we will assign it a mean value of that movie.
