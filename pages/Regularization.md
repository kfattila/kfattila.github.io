---
layout: post
title: "Regularization"
description: ""
---
{% include JB/setup %}

Regularization methods, here, will mean any kind of methods that aims to improve the generalization performance, even though, it increases the accuracy on the training error.

Regularization works as trading increased bias for reduced variance, and a good regularizer is one that makes a profitable trade that is reduces variance significantly, while increases bias moderately.

"In his story “Funes the Memorious,” Jorge Luis Borges tells of meeting a youth with perfect memory. This might at first seem like a great fortune, but it is in fact an awful curse. Funes can remember the exact shape of the clouds in the sky at an arbitrary time in the past, but he has
trouble understanding that a dog seen from the side at 3:14 p.m. is the same dog seen from the front at 3:15 p.m. His own face in the mirror surprises him every time he sees it. Funes can’t generalize; to him, two things are the same only if they look the same down to every last detail. An unrestricted rule learner is like Funes and is equally unable to function. Learning is forgetting the details as much as it is remembering the important parts. Computers are the ultimate idiot savants: they can remember everything with no trouble at all, but that’s not what we want them to do.
The problem is not limited to memorizing instances wholesale. Whenever a learner finds a pattern in the data that is not actually true in the real world, we say that it has overfit the data. Overfitting is the central problem in machine learning. More papers have been written
about it than about any other topic. Every powerful learner, whether symbolist, connectionist, or any other, has to worry about hallucinating patterns. The only safe way to avoid it is to severely restrict what the learner can learn, for example by requiring that it be a short conjunctive concept. Unfortunately, that throws out the baby with the bathwater, leaving the learner unable to see most of the true patterns that are visible in the data. Thus a good learner is forever walking the narrow path between blindness and hallucination. Humans are not immune to overfitting, either. You could even say that it’s the root cause of a lot of our evils. Consider the little white girl who, upon seeing a Latina baby at the mall, blurted out “Look, Mom, a baby maid!” (True event.) It’s not that she’s a natural-born bigot. Rather, she overgeneralized from the few Latina maids she has seen in her short life. The world is full of Latinas with other occupations, but she hasn’t met them yet. Our beliefs are based on our experience, which gives us a very incomplete picture of the world, and it’s easy to jump to false conclusions. Being smart and knowledgeable doesn’t immunize you against overfitting, either. Aristotle overfit when he said that it takes a force to keep an object moving. Galileo’s genius was to intuit that undisturbed objects keep moving without having visited outer space to witness it firsthand. 
The Bible Code, a 1998 bestseller, claimed that the Bible contains predictions of future events that you can find by skipping letters at regular intervals and assembling words from the letters you land on. Unfortunately, there are so many ways to do this that you’re guaranteed to find “predictions” in any sufficiently long text. Skeptics replied by finding them in Moby Dick and Supreme Court rulings, along with mentions of Roswell and UFOs in Genesis. John von Neumann, one of the founding fathers of computer science, famously said that “with four parameters I can fit an elephant, and with five I can make him wiggle his trunk.” Today we routinely learn models with millions of parameters, enough to give each elephant in the world his own distinctive wiggle. It’s even been said that data mining means “torturing the data until it confesses.” " [ref: P. Domingo: The Master Algorithm]


###### Bias-Variance trade-off control

Let's consider the following regularized learning.
\\[\theta_ = argmin_{\theta \in G}\{L_{theta}(X,Y) + \lambdaR(\theta)\}\\]

In the optimization of the cost function (above) the regularization part can be used to control the variance (in the model selection). Loosely speaking, if \\(\lambda\\) is zero or small, then the optimizer tries to achive a small error on fitting the model to the current data. If \\(\lambda\\) is big then the optimizer tries to find small parameter values of \\(\theta\\) for the model and it will consider the model fitting less. Thus, if the parameter values are supressed then it will reduce the variance in the model selection.

###### Example for over and underfit: 
Revisit linear regression and generate more features \\(x,x^{2},x^{3},x^{4},...,x^{n},x_{1}x_{2}, x_{2}x_{3},...,x_{n-1}x_{n},...\\)

\\[J_{1}(\theta|D) = \frac{1}{m}\sum_{i=1}^{m}(Y_{i} - (\sum_{j=1}^{n}x_{i,j}\theta_{j} + \theta_{0}))^{2} + \frac{\lambda}{2n}\sum_{j=1}^{n} \theta_{j}^{2}\\]

Then the model is a polynomial function of degree n. If the regularization parameter low, close to zero, or zero, then the coefficient of the variables can be high. The higher the coefficient the "curlier" the polynomial function. Figure below shows an example.


However, if the regularization parameter is high, then it will suppress the coefficients. The lower the coefficients are the "flatter" the polynomial function is. Figure x shows an example.


An optimal choice of the regularization parameter \\(\lambda\\) might give a reasonably "curvy" model. 



Note that, this parameter penalty regularization does not control the degree of a polynomial model nor the complexity of the model. It only can suppress the value of the coefficients in the model. In generally speaking, smaller the coefficients the "flatter" the model.

If the coefficients are suppressed then the models \\(\theta_{i}\\) (represented by red dots) and the mean of \\(\theta\\) (blue square in the figure below) is shifted toward zero, hence the bias is increased but the variance of \\(\theta\\) is reduced.



When overfitting occurs: reduce the number of the parameters of the model. Increase the data if you can.
When underfitting occurs: increase the number of the parameters of the model.


###### Regularization of k-Nearest Neighbors (kNNs).
Consider an k-NN model shown below, where the black dashed line represents the true (usually unknown) model. 


k = 1: overfitting:

k = 100, underfitting:

k = 29 would give a well regularized model.


Figures taken from: 
Source: [http://scott.fortmann-roe.com/docs/BiasVariance.html](http://scott.fortmann-roe.com/docs/BiasVariance.html)

###### Regularization for Decision Trees (DTs)

Decision trees are also prone to overfitting. DTs are actually capable of learning all data examples in the training data (especially in the case of wide data). If the decision tree is too deep then the decision boundary it represents can become too non-linear. 


**Figure:** Decision tree for regression.

Tree pruning methods are used to decrease the complexity (the size) of the tree. 
These methods cut the branches of the decision tree and replace a branch with a leave node, where the label of the new leaf is chosen by the majority of the corresponding samples.


###### Parameter norm penalty

The training of the parameters \\(\theta\\) of a model is done via optimizing a cost functions (also called learning objective):
\\[J(\theta;D) = \frac{1}{m}\sum_{i}L_{\theta}(x_{i},y_{i}) + \lambda\Omega(\theta)\\]

where \\(L(.)\\) is a cost function, \\(\Omega\\) is a regularization function on model parameters, \\(\lambda\\) the trade-off parameter between cost and regularization.

The objective function for linear regression with \\(l_{2}\\) regularization (reminder):



The objective function for linear regression with \\(l_{1}\\) becomes: 



###### \\(l_{2}\\) norm

One of the most commonly used regularization is the \\(l_{2}\\) norm, also called as ridge regression or Tikhonov regularization, and it is defined as: \\(\Omega(\theta) = \frac{1}{2}\theta^{T}\theta = \frac{1}{2}||\theta||_{2}^{2}\\)

This gives us the learning objective:
\\[J(\theta;D) = \sum_{i}L_{\theta}(x_{i},y_{y}) + \frac{\lambda}{2}\theta^{T}\theta\\]

The corresponding gradient and the update can be written as:
\\[\nabla_{\theta}J(\theta;D) = \sum_{i}\nabla_{\theta}L_{\theta}(x_{i}, y_{i}) + \lambda\theta\\]

The update rule:
\\[\theta^{l+1} = \theta^{(l)} - \epsilon(\lambda\theta^{(l)} + \sum_{i}\nabla_{\theta}L_{\theta}(x_{i}, y_{i}))\\]
That is:
\\[\theta^{l+1} = (1 -\epsilon\lambda)\theta^{(l)} - \epsilon(\sum_{i}\nabla_{\theta}L_{\theta}(x_{i}, y_{i}))\\]

This shows that during the training the parameter term exponentially decays.  (cf. Momentum methods).
Let us approximate the learning objective function via quadratic function via Taylor series at its minimum location \\(\theta^{\ast}(\theta^{\ast} = argmin_{\theta}\{\sum L_{\theta}(x_{i}, y_{i})\})\\). Then:
\\[J(\theta) = J(\theta^{\ast}) + \frac{1}{2}(\theta - \theta^{\ast})^{T} H(\theta - \theta^{\ast}) + \alpha\frac{1}{2}\theta^{T}\theta \\]
where \\(H\\) is the hessian matrix of \\(J\\) w.r.t. \\(\theta\\) at \\(\theta^{\ast}\\). There is no first order term, because this is defined at the minimum location where gradients (the first order term) vanish.
Moreover, because \\(\theta^{\ast}\\) is the local minimum, therefore the Hessian is positive semidefinite.

If we set the derivatives of the new function to zero and solve it:


###### References:
1. [http://arxiv.org/pdf/1207.0580v1.pdf](http://arxiv.org/pdf/1207.0580v1.pdf)
2. [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
3. [http://www.deeplearningbook.org/contents/regularization.html](http://www.deeplearningbook.org/contents/regularization.html)
4. On Large-batch training: [https://arxiv.org/pdf/1609.04836.pdf](https://arxiv.org/pdf/1609.04836.pdf)