---
layout: post
title: "Logistic Regression"
description: ""
---
{% include JB/setup %}

## Logistic Regression

Now we have a different type of data (experience): $$ D = \{(X_i,Y_i)|X_i \in R^n, y_i \in C \} $$, where $$C$$ is categorical. For now, lets
assume $$ C = \{ 0,1\} . For an unseen data $$ x \in R^n $$, we need a function $$ F:R^n \rightarrow  C $$ which predicts its category label whether
it belongs to class '0' or to class '1', formally: $$ F(x) \in \{0,1\}
For instance:
1. Email classification: Spam or not spam
2. Medical diagnosis: cancer or not,
3. Fraud detection: employer is cheating or not
4. Customer will pay back the loan or not.