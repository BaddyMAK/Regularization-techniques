#### In this script I had used the same dataset used for linear regression (Parkinson disease dataset)
# Regularization-techniques
For each machine learning problem, the provided dataset is not perfect means data points can have some noises acquired during the collection phase. So, in this case the trained model will learn not only our clean data but a mix between data and noise and thus may fail to predict the correct pattern which we aim. Means The trained model may give zero error for training set but will give huge errors in predicting the correct target values for test dataset.

The solution to avoid such problem is to adopt the regularization. Regularization is a technique implemented to tuning our trained model via penalizing (adding an additional penalty term) the predicted data point with high error return. This terms will be useful to restrict the fluctuations of  model's coefficients (when they take an extremely large value). This technique will keep a checking to reduce the value of error coefficients are called shrinkage methods. In this notebook we are going to talk about the Ridge Regression as a regularization methods. 

## 1- What is Ridge Regression?
Ridge regression is a regression technique that adopts the L2 regularization. It is one of the most used techniques and is particularly useful to mitigate the problem of multicollinearity in linear regression. Multicollinearity, or collinearity, is the existence of near-linear relationships among the independent variables. 

## 2- Ridge Regression implementation
We assume that our target feature is y=Xw + ν where ν is a noise or error term.
If the noise or error term has large values then it is possible that the vector ŵ takes very large values. In this case, it is better to solve the following new problem:

min|(|y-Xw|)|²+μ||w||² , μ must be set correctly and conveniently.

Assuming that: 

y (n) =x (n)T w + v(n) 

where w is a set of random Gaussian parameters independent and identically distributed with zero mean  and a variance s² (prior probability density function).

After finding the gradient of an objective function and set it to zero then solve the equation I have found the following optimum ŵ for the measured values ymeas:

$Ŵ= (X^T X +λI)-1X^T ymeas$  
$A^T$

Knowing that different values of λ give different degrees of over-fitting that can be checked through the k-fold cross validation technique. In this lab, λ has been changed to four values 3, 6, 14, 20 and a comparison between those choices will be detailed later. 
Another way to obtain the ridge regression is the minimization of the square error under the constraint of ||w||=1 (in this case λ is the Lagrange multiplier)
