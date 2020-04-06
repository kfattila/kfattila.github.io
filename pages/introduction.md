---
layout: post
title: "Introduction"
description: ""
---
{% include JB/setup %}



#### Definition (Machine Learning):

Machine learning is - generally speaking - about solving problem with computers where the computer algorithm is calibrated using real world data. In other words, the parameters of the computer algorithm is adjusted (tuned) to the problem using ereal world data.

<u>Tasks</u>: the problem to be solved; such as facial recognition or speech recognition.

<u>Experiences</u>: database or data from real world, used to adapt the algorithm to solve the task.

<u>Performance measure</u>: Measures how well the algorithm solves the task  with respect to the experience.
A computer programs is said to learn a function (i.e. calculate tens of thousands parameters of an algorithm) from experience E in order to perform task T as well as "possible" with respect to performance measure P.

Related definitions:
https://goo.gl/SIOXgM
https://goo.gl/9BV1dT


<u>**Features**</u> describe a real world entity such as a car, picture, bank account.

Objects are represented as features, attributes, and real-life entities first undergo some sort of digitalization.
Consider a banking system which predicts whether a person will be able to pay a bank loan back in the near future.  Clearly, no computational method accepts any person as "input". First, some characteristics are extracted from the person's payment history than his/her credibility can be predicted. Now, the feature extraction can be good or poor. It requires domain specific knowledge and expertise. Clearly, features to predict credibility cannot be used to predict if the person has cancer or diabetes.

types of features can be: 
	a) categorical: only '=' is defined, no ordering. 
		a. A,B,C,
		b. binary 'Y/N'
		c. weather = {'sunny', 'rainy', 'windy'}
	b) ordinal:' =','<','>' are defined. 
		a. 1,2,3,...
	c) real-valued,
	d) unstructured: text.
	e) Pixels (of pictures)


The feature extraction can be considered as a function which maps a real word entity to a digital data vector. This feature mapping can lose lot of information if it is not designed properly. Usually, feature extraction is done by domain experts. When the feature extraction is poor, then no method can save you.

When design features for  a Machine Learning system, think if you could perform the task only from the given features and ask" When I see these features, could I really perform the task well?"
Examples:
Feature extraction for disease diagnosis: Hearth disease risk (3 features: x=weight, y=height, z=blood pressure) (toy data, only for illustration)

![intro1](./images/intro1.png)

Predict Car price from mileage.  (Real data, source is below)

![intro2](./images/intro2.png)

Source: [http://www.amstat.org/publications/jse/v16n3/kuiper.xls](http://www.amstat.org/publications/jse/v16n3/kuiper.xls)

Examples for poor feature extraction:

![intro3](./images/intro3.png)

![intro4](./images/intro4.png)

**Figure 3:** Examples for poor features in classification (top) and regression (bottom).

#### Tasks

Set of data given as \\(D = \\{(x_{i}, y_{i})\_{n=1}^{n}\mid x_{i} \in R^{n}, y_{i} \in C \\} \\), where \\(C\\) can be
1. either categorical labels such as \\(C  = \\{0,1,2,...,k\\}\\) without ordering, \\(C = \\{+1, -1\\}, C = \\{'true', 'false'\\}\\) or \\(C = \\{\tikzcircle[green]{2pt}, \tikzcircle[red]{2pt}, \tikzcircle[blue]{2pt} \\}\\)
2. or real valued (i.e. \\(C = R\\))
3. structure such as tree, sentence, sequence of symbols

<u>Inductive Learning Problem:</u> construct a function \\(F\\) such that \\(F: R^{n} \rightarrow C\\), \\(F(x) = y\\).

When \\(C\\) is categorical we talk about classification problems.

When \\(C\\) is real valued we talk about regression problems.

#### Examples for tasks:

1. Classification: the goal is to find the appropriate label for a given query data.
	a. Disease diagnosis: Hearth disease risk (3 features: x=weight, y=height, z=blood pressure) (toy data, only for illustration)

	![intro5](./images/intro5.png)

	b. Face recognition

2. Regression: Predict Car price from mileage.  (Real data, source is below)

![intro6](./images/intro6.png)

Source: [http://www.amstat.org/publications/jse/v16n3/kuiper.xls](http://www.amstat.org/publications/jse/v16n3/kuiper.xls)

3. Transcription:  speech recognition, video or image capturing, machine translation, video description
4. Structured output: outputs a structure containing relationships between the input elements.  e.g. transforms a sentence to a tree that describes its grammatical structure. Natural Language Processing, Information Retrieval,
5. Anomaly detection: analyses same elements and flags the some as being unusual. e.g. credit card detection, engine failure, 
6. Synthesis and sampling: text to speech synthesis.
7. Imputation of missing values: predict value of the missing elements of \\(x \in R^{n}\\).
8. Recommendation systems
9. Denoising: remove noise from corrupted \\(x \in R^{n}\\) obtained by an unknown corruption.
10. Representation learning or unsupervised learning when \\(C\\) is not available.
11. Object tracking, face detection on images or in videos, 
12. Public surveillance, crime detection on public security cameras
13. Neural cell detection on medical light microscopic images.
14. Pneumonia detection on X-ray images
15. Autonomous car driving
16. Human activity recognition (that is to predict if the user is cooking, running, walking, travelling, etc.)