---
layout: post
title: "Logistic Regression"
description: ""
---
{% include JB/setup %}

## Logistic Regression

Now we have a different type of data (experience): \\( D = {(X_i,Y_i)|X_i \in R^n, y_i \in C } \\), where \\( C \\) is categorical. For now, lets
assume \\( C = \{ 0,1\} \\) . For an unseen data \\( x \in R^n \\), we need a function \\( F:R^n \rightarrow  C \\) which predicts its category label whether
it belongs to class '0' or to class '1', formally: \\( F(x) \in {0,1} \\)
For instance:
1. Email classification: Spam or not spam
2. Medical diagnosis: cancer or not,
3. Fraud detection: employer is cheating or not
4. Customer will pay back the loan or not.



One could fit a regression line and rounding the outcome, the regressand to 0 or 1. This approach would not work here.
A large value of \\(x\\), say \\(x=14\\), would indicate a strong relation to the class labeled by '1'. However, the corresponding
outcome of the regression function at point x could be large, far from 1.0 which would dominate in the loss function. In order to abate the overall
error the optimizer would choose another \\( \theta_{i} \\) parameters so that the \\(F(x)\\) would be closer to 1. This could increase much more 
classification error around the decision boundary.

Instead of using a linear regression, we fit an S-shaped function on the data. Many S-shaped functions are available, e.g.
arc tangent. However the most frequently used one is the sigmoid function or also called logistic function, defined as: \\( g(x) =  \\)

![Figure 1](Figure1.png)

```python
examples = [1,2,3,4,5,6]
print(examples)
```