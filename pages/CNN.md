---
layout: post
title: "CNN"
description: ""
---
{% include JB/setup %}


#### Neural Networks.

**Limitation of simple neural networks.** We saw that a simple logistic regression can learn to classify the hand-written digits from the MNIST dataset. The digits are represented in a 28x28 pixels. Thus the data lie in a 784 dimensional space. One logistic regression can recognize only one class, say, the logistic regression for the digit 2 can recognize whether the little image contains a hand-written digit 2 or not. To do this, it builds an S-shaped sigmoid function in the 784 dimensional space which looks like this:

![cnn1](./images/cnn1.png)


Each image is a dot in the 784 dimensional space. Thus, the logistic regression learns a distributions of the digits '2' and non-'2's represented in the 784 dimensional space.

What is wrong with this? What are the limitations of this approach?

The first problem is that this system works only with 28x28 images. The second problem is that it cannot handle, for instance, the following situation:
If all training image samples look like this:

![cnn2](./images/cnn2.png)

Then a neural network cannot recognize samples like this:

![cnn3](./images/cnn3.png)

because the information (the non-zero features) lies now in different dimension and there are no proper corresponding weights to capture this information. In this case, it is said the system is not invariant to translation. LogRegs and vanilla neural networks are not invariant to translation and are not invariant to rotation as well. One way to overcome this issue is to generate more training samples in which original training samples are shifted and rotated randomly.

The third problem is that it cannot exploit information about the correlation of the neighboring pixels. This means that if we randomly permutate the columns and the rows of all the images (in the same way) then the performance of the logistic regression does not change.

![cnn4](./images/cnn4.png)

Figure. LHS: an original sample from MNIST. RHS: the same sample but rows and columns are randomly permutated.

If we randomly permutate the images (permutate all images in the same way) and the weights of the LogReg model in the same way, its performance does not change. This is due to the fact the logistic regression independently weights the input pixels and sums them up: \\(\sum_{i}\theta_{i}x_{i}\\) (before applying the sigmoid functions). So Logreg and vanilla neural nets do not care about the order of the input pixels. 
This actually may be a good property in certain application in which the order of the features are not important. For instance, in bioinformatics application when features represent gene expression levels. In this case the order of the genes are totally indifferent.
But this is not the case for image recognition or some other pattern recognition problems.

From now on, we consider image data and we consider the input in matrix form (instead of a vector form. So, in case of MNIST data, the input data is 28x28 matrix instead of 784 dimensional vector).

Now, we discuss a method to handle the shortcomings mentioned above.

##### Convolution.

The convolution of \\(f\\) and \\(g\\) is written \\(f \ast g\\),  using an asterisk or star. It is defined as the integral of the product of the two functions after one is reversed and shifted. As such, it is a particular kind of integral transform:
\\[f \ast g(t) = \int f(t)g(t - \alpha)d\alpha = \int f(t - \alpha)g(t)d\alpha\\]
Now, let us consider a (possible large) two dimensional image \\(I\\) and \\(\alpha\\) (possible smaller) two-dimensional kernel \\(K\\) of size \\(m \times n\\). Then, their convolution is defined as:
\\[S(i,j) = (I \ast K)(i,j) = \sum_{m}\sum_{n} I(m,n)K(i - m, j - n) = \sum_{m}\sum_{n} I(i - m, i - n)K(m,n)\\]

The second formula is more straightforward to implement. As the indices of the kernel increases, the indices in the input image decreases. This is a bit counter intuitive, thus in the common machine learning libraries they flip the kernel and the following is used:
\\[S(i,j) = (I \ast K)(i,j) = \sum_{m}\sum_{n} I(i + m, i + n)K(m, n)\\]
In this case, this function is called cross-correlation (which is the same as convolution but without flipping the kernel) but it runs under the name convolution.

How is it used?
The kernel is iteratively applied over the image and at each position it convolves with the receptive part of the image. The parameters of the kernels do not change during scanning. In the following example, the kernel is of size 2x2 and its parameters are:
\\[K(1,1) = w; K(1,2) = x; K(2,1) = y; K(2,2) = z\\]

![cnn5](./images/cnn5.png)

Image is from the Deep Learning book by Goodfellow. The shape of the input images is: 3x8, the size of the kernel: 2x2, the size of the output: 2x3


What does this mean?
A kernel \\(K\\) is considered as a filter, and it scans the whole input image for certain patterns. If the receptive field (sub-image) does not match to the kernel, the output is small. If the receptive field matches to the kernel, then it outputs a large value. Therefore, the output (known as feature map or activation map) will indicate whether a pattern has been found in the image or not. 
For an input image of size \\(k \times l\\) and a kernel of size \\(m \times n\\), the output activation map is of size: \\((k - m) \times (l - n)\\).

![cnn6](./images/cnn6.png)

(source: [https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/))

This little pattern is also called feature. In this sense, the convolutional layer using a kernel can be considered as feature extraction. Each point in the output layer indicates whether the filter found a pattern or not.


Let us consider the following example:

![cnn7](./images/cnn7.png)

Now, the filter (kernel) iteratively scans over the whole image. When the filter matches to a part of the images, then it outputs a large score.

![cnn8](./images/cnn8.png)

If it does not match, then it outputs a small score.

![cnn9](./images/cnn9.png)

It is important to note that the weights of a kernel does not change during scanning. Thus the number of the parameters is much less than in the case of a fully connected neural network. If an input image is of size \\(k \times l\\)