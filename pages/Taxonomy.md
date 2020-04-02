---
layout: post
title: "Taxonomy"
description: ""
---
{% include JB/setup %}

The main five branches of machine learning are:
	1. Connectionism (Neural Networks, RNNs)
	2. Analogizers, (kNN, SVM)
	3. Symbolic, (Decision Trees, Rule-based, Random Forests)
	4. Evolutionary (Genetic Algorithms)
	5. Bayesian


|           | Connectionist                          | Analogizer                  | Symbolic             | Evolutionary        | Bayesian                   |
|-----------|----------------------------------------|-----------------------------|----------------------|---------------------|----------------------------|
| Model     | Neural Networks, <br>RNNs              | Support<br>Vectors Machines | Logic                | Genetic<br>Programs | Graphical<br>Models        |
| Objective | Squared Error, <br>Cross-entropy(Xent) | Maximal<br>Margin           | Accuracy             | Fitness             | Posterior<br>Probability   |
| Optimizer | Gradient<br>Descent                    | Constraint<br>Optimization  | Inverse<br>Deduction | Genetic<br>Search   | Probabilistic<br>Inference |
{:.mbtablestyle}

Note that these branches are not completely independent or disjunct. Some graphical models (E.g. Restricted Boltzmann Machines or Belief Networks) also belong to connectionist models.

Let \\(D = \{X_{i}, Y_{i}_{i=1}^{m} |+X_{i} \in R^{n}, y_{i} \in C\} \\) be a dataset.
Neural Networks (and deep models) perform well on so called deep data. This data means when data has significantly more data instances than features (\\(m \gg n\\)), for instance in speech recognition or image processing (Note that an image can be high dimensional, but the convolutional kernels at the first layer greatly reduces the number of parameters).

The connectionist models are justified by the Hebbian principle (in which associations between an environmental stimulus and a response to the stimulus can be encoded by strengthening of synaptic connections between neurons.) This also related to Behaviorism which is a systematic approach to understanding the behavior of humans and other animals. It assumes that all behaviors are either reflexes produced by a response to certain stimuli in the environment, or a consequence of that individual's history, including especially reinforcement and punishment, together with the individual's current motivational state and controlling stimuli. Although behaviorists generally accept the important role of inheritance in determining behavior, they focus primarily on environmental factors.

Deep learning, as it is primarily used, is essentially a statistical technique for classifying
patterns, based on sample data, using neural networks with multiple layers. It heavily use of statistical techniques to pick regularities in masses of data to better mine and predict data
However, it is unlikely to yield the explanatory insight and unlikely to yield general principles about the nature of intelligent beings or about cognition. 

More on this topic:
1. [https://arxiv.org/ftp/arxiv/papers/1801/1801.00631.pdf](https://arxiv.org/ftp/arxiv/papers/1801/1801.00631.pdf)
2. [https://www.theatlantic.com/technology/archive/2012/11/noam-chomsky-on-where-artificial-intelligence-went-wrong/261637/](https://www.theatlantic.com/technology/archive/2012/11/noam-chomsky-on-where-artificial-intelligence-went-wrong/261637/)


The big deal of deep learning, however, is not that we can increase the number of the hidden layers, but the fact that we can train a model having millions of parameters so that it performs reasonably well. This achievement is mainly due to big data and better computational architectures (GPUs).

Analogizers perform well on the so called wide data. This type of dataset has more features than instance (\\(m \ll n\\)). Typical examples include medical and biological data (excluding images). It is easier to extract several thousand of features from a biological/medical experiments (e.g. gene expression microarrays, shotgun proteomics)  than to collect millions of patients for every single disease. Moreover, kernel functions provide a non-linear decision boundary. SVMs, decision trees, and other shallow models still perform much better on these type of datasets than deep nerual networks.

In fact, shallow models still perform better on UCI datasets than deep neural nets in general. Looking at Kaggle challenges that are not related to vision or sequential tasks, gradient boosting, random forests, or SVMs are winning most of the competitions. Deep Learning is notably absent, and for the few cases where ANNs won, they are shallow. For example, the HIGGS challenge, the Merck Molecular Activity challenge, and the Tox21 Data challenge were all won by ANNs with at most four hidden layers. Surprisingly, it is hard to find success stories with ANNs that have many hidden layers, though they would allow for different levels of abstract representations of the input.

Genetic Algorithms are also inspired by the nature (like connectionist models) and turned to be efficient in finding good model structures (i.e. a structure of an ANN), but gradient descent algorithms proved to be more efficient on optimizing the model parameters (weights).

###### Software 2.0 paradigm

The classic software 1.0 is that is known in programs written in languages such as Python, C++, R, Matlab, Ruby, etc.
Andrej Karpathy introduces an interesting concept termed Software 2.0. In this concept, programs are written in much more abstract way, more human unfriendly language, such as the weights of a neural network. Interestingly, Humans are not involved in the programming. Instead, general aims and goals are formulated on the desirable program behavior and the model parameters are coded with backpropagation.  Thus, the program designer specifies the input and the desired output and lets the back propagation (the optimizer) write the code (the weights of the neural network). 
For instance, try to implement a speech recognition system without machine learning or training data. A very related, often cited humorous quote attributed to Fred Jelinek from 1985 reads “Every time I fire a linguist, the performance of our speech recognition system goes up”.

For more discussion see:
[https://medium.com/@karpathy/software-2-0-a64152b37c35](https://medium.com/@karpathy/software-2-0-a64152b37c35)
