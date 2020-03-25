---
layout: post
title: "Analogizers"
description: ""
---
{% include JB/setup %}

Learning by analogy.

"Frank Abagnale Jr. is one of the most notorious con men in history. Abagnale, portrayed by Leonardo DiCaprio in Spielberg’s movie Catch Me If You Can, forged millions of dollars’ worth of checks, impersonated an attorney and a college instructor, and traveled the world as a fake
Pan Am pilot—all before his twenty-first birthday. But perhaps his most jaw-dropping exploit was to successfully pose as a doctor for nearly a year in late-1960s Atlanta. Practicing medicine supposedly requires many years in med school, a license, a residency, and whatnot, but Abagnale managed to bypass all these niceties and never got called on it.
Imagine for a moment trying to pull off such a stunt. You sneak into an absent doctor’s office, and before long a patient comes in and tells you all his symptoms. Now you have to diagnose him, except you know nothing about medicine. All you have is a cabinet full of patient files: their symptoms, diagnoses, treatments undergone, and so on. What do you do? The easiest way out is to look in the files for the patient whose symptoms most closely resemble your current one’s and make the same diagnosis. If your bedside manner is as convincing as Abagnale’s, that might just do the trick. 
Analogy was the spark that ignited many of history’s greatest scientific advances. The theory of natural selection was born when Darwin, on reading Malthus’s Essay on Population, was struck by the parallels between the struggle for survival in the economy and in nature. Bohr’s
model of the atom arose from seeing it as a miniature solar system, with electrons as the planets and the nucleus as the sun. Kekulé discovered the ring shape of the benzene molecule after daydreaming of a snake eating its own tail."[ref: The Master Algorithm by P. Domingo.]

**Instance-based learning, k-Nearest Neighbor (kNN).** The idea is that, classify a new object as the class of the "closest" instance-reference.
It requires 1) a distance function to measure the distance between data instances, and 2) and a policy to determine the closeness of a data instance to a class based on the distance between new instance and the members of the given class.
	1. For example, one can use the Euclidean distance defined for \\(x, y \in R^{n}\\):
	\\[d(x,y) = ||x - y||_{2} = \sqrt{\sum_{i=1}^{m}(x_{i} - y_{i})^{2}}\\]
	2. Given a dataset \\(D = \{X_{i}, Y_{i}|X_{i} \in R^{n}, y_{i} \in C\}\\), where \\(C\\) is categorical, and let's define a set of indices indicating the index of the k-nearest data as: \\(I_{k} = argmin_{i}^{k}\{d(t, x_{i})\}\\), where \\(argmin^{k}\\) denotes the set of indeces of the \\(k\\) minimal elements in its argument. Now the class label of the new elemet can be chosen on the majority of \\(\{y_{i}: i \in I_{k}(t)\}\\).

k-Nearest Neighbor (kNN) can be considered as a voting algorithm where each of the k nearest neighbors votes to its own class label.



This method using a distance metric function and the given database gives a partition of the space into regions (for k=1):

This approach can be easily defined for multi-class classification problems:

The main benefits of this algorithm are:
	- Fast, if the reference dataset is small
	- Non-parametric,
	- Does not require learning, the dataset itself represents the model/knowledge.
The disadvantage of this method:
	- Slow, if the reference dataset is large
	- it does not work well in very high dimensional spaces.
Sensitive to class-imbalanced cases.

Note also that, kNN does not require proper distance metric and it also can be used with any similarity-like measure (just change argmin to argmax). However, in this case, some error guarantees cannot be held anymore and perhaps these functions will not provide a proper Voronoi-partition of the space. But on the other hand, sometimes application specific similarity measures can provide better classification results with kNNs.

"One of the reasons researchers were initially skeptical of nearest neighbor was that it wasn’t clear if it could learn the true borders between concepts. But in 1967 Tom Cover and Peter Hart proved that, given enough data, nearest-neighbor is at worst only twice as error-prone as
the best imaginable classifier. If, say, at least 1 percent of test examples will inevitably be misclassified because of noise in the data, then nearest-neighbor is guaranteed to get at most 2 percent wrong. Up until then, all known classifiers assumed that the frontier had a very specific form, typically a straight line. This was a double-edged sword: on the one hand, it made proofs of correctness possible, as in the case of the perceptron, but it also meant that the classifier was strictly limited in what it could learn. Nearest-neighbor was the first algorithm in history that could take advantage of unlimited amounts of data to learn arbitrarily complex concepts. No human being could hope to trace the frontiers it forms in hyperspace from millions of examples, but because of Cover and Hart’s proof, we know that they are probably not far off the mark." [ref: The Master Algorithm by P.D.]

We can observe that, the decision boundary does not change if you erase the data points surrounded by instances of the same type:



The support vector machine (SVM), loosely speaking, looks a lot like weighted k-nearest-neighbor: the frontier between the positive and negative classes is defined by a set of examples and their weights, together with a similarity measure. A test example belongs to the positive class if, on average, it looks more like the positive examples than the negative ones. The average is weighted, and the SVM remembers only the key examples required to pin down the frontier. These examples are called support vectors because they’re the vectors that “hold up” the frontier: remove one, and a section of the frontier slides to a different place. You may also notice that the frontier is a jagged line, with sudden corners that depend on the exact location of the examples. Real concepts tend to have smoother borders, which means
nearest-neighbor’s approximation is probably not ideal. But with SVMs, we can learn smooth frontiers, more like this:




**Figure.** Decision Boundaries made by kNN (A) and by SVM (B).

Now let us have some data: \\(D = \{X_{i}, Y_{i}|X_{i} \in R^{n}, y_{i} \in C\}\\), where \\(C\\) is categorical. For now, let's assume \\(C=\{0,1\}\\). In the case of Logistic Regression, parametrized by \\(\theta\\) (the bias is included), we used a sigmoid function to represent the relationship between \\(X\\) and \\(Y\\), and fitted an S-shaped sigmoid function over the data, and the LogReg provided us a linear decision boundary by solving the minimization of the following cost functions:

\\[J(\theta|D) = \frac{1}{m}\sum_{i=1}^{m}(-y_{i} \log(g(\theta^{T}x_{i})) - (1-y_{i}) \log(1 - g(\theta^{T}x_{i}))) + \lambda\sum_{j=1}^{n}\theta_{j}^{2}\\]

where \\(g(x) = \frac{1}{1+e^{-x}}\\) is the sigmoid function.
Now, let us change the log-functions \\(log(g(z))\\) which measured the cost of miss-classification and let us choose the following ones:
\\[(z)_{+} = max(z,0)\\]

For a positive data \\((x_{i}, y_{i} = 1)\\), we want \\(\theta x_{i} \geq 1\\), hence we deinfe the cost function as \\((1 - \theta x_{i})\\)_{+} and
For a negative data \\((x_{i}, y_{i} = 0)\\), let us want \\(\theta x_{i} \leq -1\\), hence we deinfe the cost function as \\((\theta x_{i} + 1)\\)_{+}.
The choise of \\(\pm 1\\) is arbitrary but it must be non-zero. This cost function is somewhat similar to the cost function used in Logistic Regression.

**Figure 2.** Cost functions of Logistic Regression and Support Vector Machines for positive (left) and negative (right) classes, respectively.

Using this cost functions we can formulate the objective function of SVM classification.
\\[J(\theta|D) = \frac{1}{m}\sum_{i=1}^{m} y_{i}(1 - \theta^{T}x_{i})_{+} + (1 - y_{i})(\theta^{T}x_{i} + 1)_{+} + \lambda\sum_{j=1}^{n}\theta_{j}^{2}\\]