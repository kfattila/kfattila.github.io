---
layout: page
title: "Regression"
description: ""
---
{% include JB/setup %}


**Regression:** Let \\(D = \{ X_{i}, Y_{i}|X_{i} \in R^{n},y_{i} \in R \} \\) be some data (experience). Aim: model the relationship between the *dependent variable* \\(Y_{i}\\) and *independent variables* \\(X_{i}^{T} = [x_{i,1}, x_{i,2}, ..., x_{i,n}]\\).
\\(x_{i}\\) are also called:  regressors, exogenous variables, explanatory variables, covariates, input variables, predictor variables, or independent variables.
\\(Y_{i}\\) is also called: regressand, endogenous variable, response variable, measured variable, criterion variable, or dependent variable.
<span style="text-decoration: underline">Linear regression:</span> linear relationship is assumed between \\(Y\\) and \\(X\\). Then the <span style="text-decoration: underline">linear regression model, or hypothesis</span> which represents the linear relationship between \\(X\\) and \\(Y\\) can be formulated as follows:

\\[\Y = f(X) = \theta_{0} + \sum_{j=1}^{n}(x_{j}\theta_{j})\]

where coefficients \\(\theta_{j}\\)'s called unknown parameters, or model parameters. These parameters are often arranged in a vector and we denote it as \\(\theta = [\theta_{0}]\\)