---
layout: post
title: "RNN"
description: ""
---
{% include JB/setup %}

The RNNs unrolled in time can be represented as follows:


The long-short term memory (LSTM) is also somewhat like this, but it has a more complicated  memory update mechanism. LSTMs use a memory and a state. This was done in order to make the model capable of learning long-term dependencies. The core idea of the LSTMs is the state which runs through the whole time. The state is denoted by \\(C\\) here. An LSTM performs four steps on its state using so called gates. The gates are: input modulation, input gate, forget gate and output gate. The main idea in LSTMs is that each gate is differentiable so their operation can be learned from data.

Let the data be \\(x = x_{1}, ..., x_{T}\\), where each \\(x_{i} \in R^{n}\\) are column vectors, and the desired outputs are \\(y = y_{1}, ..., y_{T}\\).
Let the state vector be \\(c \in R^{k}\\) and the hidden unit \\(h \in R^{l}\\). Before, we discuss the operations, let us take a look at the structure of the cell.

###### The forget gate:

It will calculate which information should be removed from the state \\(c\\), based on the the input from the previous hidden unit \\(h_{t-1}\\) and the current input \\(x_{t}\\). These vectors are combined and transofmed as follows:
\\[f_{t} = \sigma\\]