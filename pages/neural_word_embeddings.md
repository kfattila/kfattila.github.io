---
layout: post
title: "Neural Word Embeddings"
description: ""
---
{% include JB/setup %}

We have seen that, the collaborative filtering is capable of learning a vector representation of data instances based on their relative relation to each other and to the users.

In this part, we will see other methods for learning real-valued, fixed dimensional representation for words and other real-world entities.  These are often called 'word-embedding' (to real-valued vector space), or 'word representation', or also called 'distributed representations' of words and phrases, etc. Neural word embedding refers to the fact the embedding is done using neural networks.
When a word representation is available, then documents or sentences can be encoded by the list of the word representations of the corresponding words. Word embedding provides a way to represent words with numerical vectors so that words and texts can be processed with using machine learning methods.

#### One-hot-encoding for words representation

Perhaps, this is the simplest method as it works as follows. 
1. Take a large collection of unstructured text from, for instance, Wikipedia or Twitter. 
2. Collect or identify all unique words from the text. This process is called tokenization, therefore unique words are called tokens, sometimes.
3. Give some ordering over the tokens and index each word (token) by their ordinal number. Let us suppose the total number of different tokens is N.
4. Create an N-dimensional vector for each token which contains '1' at the index of the token, all other vector components are zero.

For example:
Consider the following words, 'a', 'abbreviation', apple, 'auto', 'bear', …, 'zoology'.
The corresponding word-embeddings are:

![neuralword1](./images/neuralword1.png)

The problem with this approach is that it consumes too much memory and the representation is too sparse. In a typical real-world application, the number of the tokens can be several million (N=2-3 million). Thus, there is a several million dimensional vector for every token, where there is only one component with non-zero element. 
The other problem with one-hot-encoding is that, the vector space does not yield a semantic similarity measure corresponding to the original words. That is, the distance (or the scalar product) of the one-hot vectors does not correspond to the semantic differences of the corresponding words.


#### Word2vec method

This approach aims at providing real-valued, dense representation for (English) words from a very large collection of text. This approach takes into account some semantic relationship between the words. When the words tend to appear in similar semantic context, then they obtain similar real-valued representation. One of the advantages of word2vec method is that it does not require a well-annotated, labeled data, word2vec requires only a large unstructured text. This large amount of text can be taken from, e.g. all articles from Wikipedia or all messaged from Twitter. 

The word2vec method actually refers to two different approaches: 'skip-gram' model and 'Continuous Bag of Words (CBOW)', both types uses a neural network to learn the word embedding.

#### CBOW method

The main idea of this method is to predict the probability of a word given a context. The length of the context can vary from one to few tens. Let us demonstrate how this works through an example. Let us suppose, we have a small text:

"Hey this is sample corpus using only one context word." 

First, the words are represented via one-hot-encoding embedding. The number of the tokens is: V=10
One-hot-Encoding for every word:

![neuralword2](./images/neuralword2.png)

The training input-output pairs are constructed from the word and from its context (window). The context of every word is one of the adjacent words in the sentence. Therefore our data looks like this:

![neuralword3](./images/neuralword3.png)

If larger context (context=2)were used, then the data would have look like this:

![neuralword4](./images/neuralword4.png)

Etc.

The goal of CBOW is to predict the probability of the target word from the context words. That is, we want to calculate:
\\[p(target word \mid context_{1}, context_{2}, ..., context_{K}) \\]
This is achieved with a three-layer neural network (shown for one context):

![neuralword5](./images/neuralword5.png)

**Figure 1.** The three-layer neural network used in the CBOW model. Input is one context word and the output is the target word. Context and target words are represented with one-hot-encoding.


In real-world applications, the number of the hidden units is around 300 (N=300).
The input and the output layer, both are one-hot encoded of size [1×V].
There is no activation function at the hidden layer.
There is a soft-max normalization at the output layer.
The error between the true output and the observed output is back-propagated.
The word representation (embedding) of the target word is taken from the weights between the hidden and the output layer.

When more than one context words are given, then the model is defined as:

![neuralword6](./images/neuralword6.png)

**Figure 2.** CBOW for k context words. In this case the hidden data are averaged.

**Forward propagation in the CBOW model.**

The inputs are a one-hot-encoding of the context words: \\(x_{1},x_{2},...,x_{K}\\). The hidden activation is given as: \\(h = \frac{1}{K}\sum_{K}(Wx_{k}) = \frac{1}{K}W(\sum_{k}x_{k})\\).

The output vector is then calculated as \\(u = W'h\\). Remember, the shape of the output vector \\(u\\) is of \\([1,V]\\).
The output is put through a soft-max layer to convert the output vector to a discrete probability distribution. The probability of the *j*th word is given by:
\\[y_{j} = p(target word with index j \mod x_{1},x_{2},...,x_{K}) = \frac{exp(u_{j})}{\sum_{i=1}^{V}\exp(u_{i})}\\]

**The learning objective** of CBOW is defined by a soft-max function.

\\[W,W' \leftarrow arxmax_{\Theta}\\{p(target word j \mid context_{1}, context_{2}, ..., context_{K}) = \frac{exp(u_{j})}{\sum_{i=1}^{V}\exp(u_{i})} \\}\\]

#### Model dissection

Note that the context words and the target words are one-hot encoded. So, there is only one non-zero element in their corresponding vectors. 

![neuralword7](./images/neuralword7.png)

The word representation of the target word (with index \\(j\\)) is composed from the weights of the *j*th column of \\(W'\\) matrix, and it is denoted as \\(W'\_{j}\\). Note that the word representation is of shape \\([N,1]\\).

For a context word with index \\(k\\) we have \\(h = W_{k}\\), where \\(W_{k}\\) is the *k*th row from weight matrix \\(W\\). The probability of the target word with index \\(j\\) with respect to one context word with index \\(k\\):
\\[p(jth target word \mod x_{k}) = \frac{\exp(W_{j}'^{T}h)}{\sum_{i=1}^{V}\exp(W_{i}'^{T}h} = frac{\exp(W_{j}'^{T}W_{k}}{\sum_{i=1}^{V}\exp(W_{i}'^{T}W_{k}}\\]

For K different context words the hidden activation is given: \\(h = \sum_{k=1}^{K}W_{k}\\).
The probability of the target word with index j with respect to K different context words:
\\[p(jth target word \mod x_{1},x_{2},...,x_{k}) = \frac{\exp(W_{j}'^{T}h)}{\sum_{i=1}^{V}\exp(W_{i}'^{T}h} = \frac{}{} \\]
Advantage: This model is of low-memory cost. 

#### Skip-gram model

Here, the aim is to calculate some distributed word representation but the model structure and the aim is the opposite. In this approach, we want to predict the context words for a given target word that is \\(p(context_{x}\mid target_{\downarrow})word\\) Consider the previous example: *"Hey this is sample corpus using only one context word."*

Now, for a target word "corpus" we want to get high probability for the context words: "sample" and "using". This approach does start with one-hot encodings of the words as before.

![neuralword8](./images/neuralword8.png)

Note that, there is no activation function here either.

#### Forward propagation

Consider a target word with index \\(k\\). The input vector is V-dimensional, there is one non-zeros component at index \\(k\\). Therefore, the hidden activation is \\(h = x_{k}^{T}W = W_{k}\\) (of shape \\([1,N]\\)), that is: the first layer will just select the kth row from the first weight matrix \\(W\\). Now, we calculate the outputs at the last layer as follows:
\\[u = hW'\\]

These examples (above) were taken from [3].

#### References:
1. [http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)
2. [http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_I_The_Skip-Gram_Model.pdf](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_I_The_Skip-Gram_Model.pdf)
3. [http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)
4. [https://arxiv.org/pdf/1411.2738.pdf](https://arxiv.org/pdf/1411.2738.pdf)