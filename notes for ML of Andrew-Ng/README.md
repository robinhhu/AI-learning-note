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
Matrix is a rectangular array of numbers. Dimension of Matrix is the number of rows * number of columns(e.g. 4 * 3 matrix)

Matrix Elements(entries of matrix): A = [1402, 191] A_ij = “i,j entry” in the i^th row, j^th column. So A_11 = 1402, A_12 = 191.

#### Identity matrix: 
Denoted I(or I_i*i). For any matrix A, A * I = I * A = A.

#### Matrix inverse: 
If A is an m * m matrix, and if it has an inverse, A * A^-1 = A^-1 * A = I. (Only square matrix has an inverse. Matrices don’t have an inverse are “singular” or “degenerate”

#### Matrix transpose: 
A is an m*n matrix, and let B = A^T. Then B is an n*m matrix, and B_ij = A_ji.

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

Then x^(2) can be a n-dimensional matrix. That is the same as x^(2)_j, where7 j is greater than 1 and less than n.

Our hypothesis becomes h_theta(x) = theta_0 + theta_1*x_1 + ..... theta_n * x_n
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/multivariatelinearregression.png)

### Gradient descent for linear regression for multiple variables:
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent%20for%20mv.png)
 ![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/gradient%20descent%20for%20mv2.png)
#### How to make gradient descent work well:
##### Feature scaling: 
![Image text](https://github.com/robinhhu/AI-learning-note/blob/master/image/featurescaling.png)
make sure features are on a similar scale(so that gradient descent can converge much more faster): Get every features into approximately in range between -1 and 1 by divide or multiply a number.

There are two ways to implement. We can divide by the max possible value. For example, x in range 0 to 1000, then divide x by 1000. Or use mean normalization. That is, replace x with (x-u)/s to make features have approximately zero mean, where u is the average value and s is the range of the value.

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

