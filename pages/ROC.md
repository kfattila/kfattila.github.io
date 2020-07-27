---
layout: post
title: "ROC"
description: ""
---
{% include JB/setup %}


**Receiver Operator Characteristics (ROCs).**

Each binary classifier, Logistic regression, Neural Networks, and Support Vector Machines, and in general many other binary classifiers, provides a score for a data \\(x\\) calculated as \\(h_{\theta}(x)\\), then the classification is based on whether this score is above or under a certain pre-defined threshold. 
We can calculate these scores for all positive data instances, and we get a distribution of these scores. Similarly, we can calculate the scores for all data instances belonging to the negative classes, and we get another distribution for the negative class. 
The performance of a binary classifier is actually depends on how well these distributions are separated from each other. The ROC analysis is used to characterize how well these two distribution can be separated.

![roc1](./images/roc1.png)

**Figure 1.** Example of such score distribution obtained from Logistic Regression Model.

Score distributions:

![roc2](./images/roc2.png)

Overlapping groups.

![roc3](./images/roc3.png)

Indistinguishable groups

![roc4](./images/roc4.png)



#### Calculating FPR, TPR

![roc5](./images/roc5.png)

![roc6](./images/roc6.png)


\\[True\ Positive\ Rate = \frac{TP}{(TP + FN)}\\]

\\[False\ Positive\ Rate = \frac{FP}{(FP + TP)}\\]

![roc7](./images/roc7.png)

![roc8](./images/roc8.png)

ROC curve:

![roc9](./images/roc9.png)

Area Under the curve (AUC):

![roc10](./images/roc10.png)

Examples for ROC curves.

![roc11](./images/roc11.png)

Good classification:

![roc12](./images/roc12.png)

Poor classification:

![roc13](./images/roc13.png)

No classification:

![roc14](./images/roc14.png)

No separation:

![roc15](./images/roc15.png)

Local humps:

![roc16](./images/roc16.png)


Notes:
1. The ROC analysis in a non-parametric method, it does not require the distributions in any specific parametric form. It is very closely related to the Wilcoxson statistics.
2. ROC analysis is not hindered by the class-imbalance problems.
3. The AUC can be interpreted as the probability of that in randomly chosen two data the positive data has higher score than a negative data. 