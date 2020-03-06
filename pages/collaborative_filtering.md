---
layout: post
title: "Collaborative Filtering"
description: ""
---
{% include JB/setup %}
Content based recommendation systems offers new contents or products for customers based on their personal preference (in order to increase selling and revenue). It is extensively used by online retail stores such as Amazon, eBay, Netflix.

**Example:** Consider a movie rental store which has some movies rated by some users, organized in a table as follows(illustration):

|                 | Alice | Beata | Cecil | Dennis | Emily |
|-----------------|-------|-------|-------|--------|-------|
| Terminator      | 1     | 1     | 2     | 2      | 4     |
| Star Wars       | 2     | 2     | 5     | 3      | 5     |
| Batman          | 1     | 2     | 4     | 2      | 2     |
| Nothing Hill    | 4     | 2     | 1     | 2      | 1     |
| Amelie          |       | 5     | 2     | 3      | 2     |
| The notebook    | 4     | 5     | 2     | 2      | 2     |

The task is to recommend some new movies for the users what they might like (and increase the revenue).
First, let's introduce some feature vectors to represent the products.

|                  | \\(x_{1}\\) (action) | \\(x_{1}\\) (romantic) |
|------------------|----------------------|------------------------|
| Terminator       | 0.9                  | 0.01                   |
| Star Wars        | 0.7                  | 0.6                    |
| Batman           | 0.7                  | 0.1                    |
| Nothing Hill     | 0.1                  | 0.8                    |
| Amelie           | 0.05                 | 0.9                    |
| The notebook     | 0.1                  | 0.85                   |

So, if we include the usual constant 1 for the bias parameter then each movie can be represented by a feature vector \\(x = [1, x_{1}, x_{2}^{T}]\\), e.g.: Terminator = \\([1,0.9,0.0.01]^{T}\\).
The idea is that we model the rating of each user by a linear regression, that is, the rate of a user on a product is a linear combination of the features. Let's denote the number of users \\(U\\) and number of products \\(n\\) and the number of features \\(m\\). Then we want \\(\theta_{u} = [\theta_{0},\theta_{1},...,\theta_{m}]^{T}\\) for each user \\(1 \leq u \leq U\\).

This can be learned by minimizing the following error function.
\\[J(\theta_{u}|D) = \frac{1}{n}\sum_{i=0; y_{i},u \neq ?}^{n}(\theta_{u}^{T}x_{i} - y_{i,u})^{2} + \lambda\sum_{j=1}^{m}(\theta_{u,j})^{2} \\]

The parameters \\(\theta_{u}\\) for user \\(u\\) is independet and thus they could be learned using a single optimization:
\\[ J(\theta_{1}, \theta_{2}, ..., \theta_{U}|D) = \sum_{u=1}^{U} \frac{1}{n} \sum_{i=0; y_{i},u \neq ?}^{n} (\theta_{u}^{T}x_{i} - y_{i,u})^{2} + \lambda\sum_{u=1}^{U}\sum_{j=1}^{m}(\theta_{u,j})^{2}  \\]

When the parameters \\(\theta_{u}\\) are given, the one can learn the features for the products to indentify the best parameters vector associated with a product:
\\[ J(x_{1}, x_{2}, ..., x_{n}|D) = \sum_{i=1}^{n} \frac{1}{U} \sum_{u=1; y_{i},u \neq ?}^{U} (\theta_{u}^{T}x_{i} - y_{i,u})^{2} + \lambda\sum_{i=1}^{n}\sum_{j=1}^{m}(x_{i,j})^{2}  \\]

When neither \\(\theta_{u}\\) nor \\(x_{i}\\) feature vectors are given (but only some user preference), one can randomly initialize both parameters and can exacute the two optimization back and forth and updateing the parameters after each optimization. However, the optimization can be solved simultaneiously:
\\[ J(x_{1}, x_{2}, ..., x_{n}, \theta_{1}, \theta_{2}, ..., \theta_{U}|D) = \sum_{i=1}^{n}\sum_{u=1; y_{i},u \neq ?}^{U} (\theta_{u}^{T}x_{i} - y_{i,u})^{2} + \lambda\sum_{i=1}^{n}\sum_{j=1}^{m}(x_{i,j})^{2} + \lambda\sum_{u=1}^{U}\sum_{j=1}^{m}(\theta_{u,j})^{2} \\]

The procedure, above, for learning product feature parameters \\(x_{i}\\) and user preference parameters \\(\theta_{u}\\) called **collaborative filtering**. 