---
layout: post
title: "NTM"
description: ""
---
{% include JB/setup %}


On the previous models, the memory and the state was stored inside a unit. **Neural Turing Machines (NTMs)** use an external memory and it updates it (i.e. reads old content, adds new content or removes old content from it) based on the current input etc. The key element in NTMs is that the reading and the writing in the memory depends on the input and the learning where to read from and where to write to is based on gradient descent. This means that the reading/writing operations are differentiable.

The NTMs are based on vectors, it reads vectors and outputs vectors, and it operates with vectors within its internal procedures. Let us assume we are given a single data sequence \\(x = x_{1}, ..., x_{T}\\), where each \\(x_{i} \in R^{n}\\) and the desired outputs are \\(y = y_{1}, ..., y_{T}\\)

The memory is an array consisting of vectors. Let \\(M \in R^{k \times l}\\) denote the memory. This memory has \\(k\\)  memory slots and each slot is an l dimensional real-valued vector. The memory slot at position \\(i\\) will be indexed as \\(M(i)\\). The reading and the writing has to be differentiable. So the key idea is that the reading/writing happens at everywhere in the memory, but at different extent. 
Let us index the memory at each time as \\(M_{t}\\).  This means just the memory state at step \\(t\\).

#### Reading: 

The reading from the memory is defined as:
\\[ r_{t} \leftarrow \sum_{i} w_{t}(i)M_{t}(i) \\]
where \\(w_{t}(i)\\) is the weight of the memory slot at position \\(i\\).  We have an additional constraint: \\(\sum_{i} w_{t}(i) = 1\\). That is, the reading will be a convex combination of the memory slots. 
Note that, this operation is differentiable with respect to both the memory and the weighting. Note that, this is very similar to the context vector generation with attention models.

#### Writing:

The writing consists of two steps. First erasing an old element from the memory and then adding new information \\(a_{t}\\). The idea is the same, we erase from everywhere and add to everywhere, but at different extent. Let \\(e_{t} \in R^{l}\\) be the erase vector at time \\(t\\) whose elements lie in the range (0,1). The memory vectors are updated as follows:
\\[M_{t}(i) \leftarrow M_{t-1}(i)\textdegree [\1 − w_{t}(i)e_{t}] + w_{t}(i)a_{t}\\]
where **1** is a vector of all 1-s. 
Note that, the weights can be considered as attention. A large value of the weight indicates the location in the memory, where the system should focus on.

#### Addressing mechanisms:

Now, the question is how the weights are produced. There are two main mechanisms: content-based and location-based focusing. The addressing mechanism uses several inputs. These inputs are \\(k_{t} \in R^{1}, \beta_{t} \in R, g_{t} \in R, s_{t} \in R^{k}, \gamma \in R\\). These data are coming from the controller.

1. Content-based Focusing. Let \\(k_{t} \in R^{l}\\) be a key (querry). This key is then iteratively compared to all memory slots, then the similarity between the key \\(k_{t}\\)  and the memory slot will determint the weight. \\[w_{t}^{c} \leftarrow softmax(\beta_{t}s(k_{t}, M_{t}(i))) \\] where \\(\beta_{t}\\) is the temperature parameter of the soft-max function and \\(s(.,.)\\) is the cosine similarity. Therefore, the weigth vector w_t  will indicate the similarity between the querry vector and the vector in the memory.
2. Location-based focusing: This step consists of several parts.
  a. Gated-weighting.
Combines the weight vector from the previous state with the current state. The interpolation scalar weight is \\(g_{t} \in (0,1)\\)
\\[w_{t}^{g}(i) \leftarrow g_{t}w_{t}^{c} + (1 - g_{t})w_{t-1} \\]
If the interpolation weight is zero, then the content weighting entirely ignored and it will use the attention weight vector from the previous step. If the interpolation weight is one \\((g_{t} = 1)\\), then the attention weight from the previous step is completely ignored.
  b. Convolutional Shift.
\\[w_{t}^{s}(i) \leftarrow \sum_{j=1}^{k}w_{t}^{g}(j)s_{t}(i - j)\\]
where all index arithmetic is computed modulo \\(k\\). The shift vector \\(s_{t}\\) is of k-dimensional vector.
  c. Sharpening. This is done by using a temperature parameter \\(\gamma_{t} \geq 1\\) 
 \\[w_{t}(i) \leftarrow \frac{w_{t}^{s}(i)^{\gamma_{t}}}{\sum_{j}w_{t}^{s}(j)^{\gamma_{t}}}\\]
This step ensures that the system will focus on a small region of the memory, instead of a blurred region over several memory slots. 



The methods above described how the memory is used. Now we will discuss the controller. The controller can be an LSTM as well (or a standard feedforward neural network) with the following mechanisms. The LSTM has four inputs now:
- the input data \\(x_{t}\\),
- the vector from the hidden unit \\(h_{t-1}\\)
- the state vector \\(c_{t-1}\\), and 
- a new input, the information \\(r_{t}\\) from the memory.

The first three input is the same as in a vanilla LSTM.
The controller has to deal with the data read from the memory. The gates of the LSTM are now defined as follows:
\\(f_{t} = \sigma(W_{f}[h_{t-1};x_{t};r_{t}] + b_{f})\\)
Note the new term related to the data read from the memory : \\(r_{t}\\)
\\(i_{t} = \sigma(W_{i}[h_{t-1};x_{t};r_{t}] + b_{i}) \\)
\\(\widetilda{c}\_{t} = tanh(W_{c}[h_{t-1};x_{t};r_{t}] + b_{c}) \\)
\\(c_{t} = c_{t-1}\textdegree f_{t} + \widetilda{c}\_{t}\textdegree i_{t} \\)
\\(o_{t} = \sigma(W_{0}[h_{t-1};x_{t};r_{t}] + b_{0}) \\)
\\(h_{t} = o_{t}\textdegree tanh(c_{t}) \\)

Now, the outputs from the LSTM are the following:
\\(y_{t} = a(W_{y}h_{t} + b_{y})\\) as before.
And the LSTM emits vectors that defines the interactions with the memory at the current time step.
\\(k_{t}^{w} = W_{k}^{w}h_{t} + b_{k}^{w} \in R^{l}\\), the key for memory writing.
\\(k_{t}^{r} = W_{k}^{r}h_{t} + b_{k}^{r} \in R^{l}\\), the key for memory reading.
\\(\Beta_{t}^{r} = W_{\Beta}h_{t} + b_{\Beta} \in R\\), the temperature for the softmax.
\\(\gamma_{t} = W_{\gamma}h_{t} + b_{\gamma} \in R\\), the parameter for the shaprening.
\\(g_{t} = W_{g}h_{t} + b_{g} \in R\\), the interpolation weight.
\\(s_{t} = W_{s}h_{t} + b_{s} \in R^{k}\\), the rotation (shift) vector.
\\(a_{t} = W_{a}h_{t} + b_{a} \in R^{l}\\), the write vector.
\\(e_{t} = W_{e}h_{t} + b_{e} \in R^{l}\\), the erase vector.

Differentiable neural computer (DNC) is an upgraded version of NTMs having more complicated memory addressing mechanisms.


References:
[https://distill.pub/2016/augmented-rnns/](https://distill.pub/2016/augmented-rnns/)
[https://www.nature.com/articles/nature20101](https://www.nature.com/articles/nature20101)
[http://arxiv.org/pdf/1410.5401.pdf](http://arxiv.org/pdf/1410.5401.pdf)