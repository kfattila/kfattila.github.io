---
layout: post
title: "NCD"
description: ""
---
{% include JB/setup %}

The information distance (ID) for binary strings x and y is defined as the shortest binary program \\(p\\) on a reference universal prefix computer \\(U\\) with input \\(y\\), which computes \\(x\\), and vice versa; formally:

\\[ID(x, y) = min\{l(p) : U(p,x) = y, U(p,y) = x\}\\]

This is a symmetric function and it has been proven that it satisfies the triangle inequality up to an additive constant. But it is incomputable. It has been shown that ID can be computed by the so-called max distance, up to an additive logarithmic term:

\\[E(x, y) = max\{K(x|y), K(y|x)\}\\]

where \\(K(x|y)\\) denotes the conditional Kolmogorov complexity defined as \\(K(x|y) = min\{l(p) : U(p,y) = x\}\\).

In general, the “up to an additive logarithmic term” means that the information required to reconstruct \\(x\\) from \\(y\\) is always maximally correlated with the information required to reconstruct y from \\(x\\) that is dependent on the former amount of information. Thus \\(E\\) is also a suitable approximation for the information distance.

The non-normalized information distance is not a proper evolutionary distance measure
because of the length factor of strings. For a given pair of strings \\(x\\) and \\(y\\) the
normalized information distance is defined by

\\[D(x,y) = \frac{max\{K(x|y),K(y|x)\}}{max\{K(x|\epsilon),K(y|\epsilon)\}} \\]
where \\(\epsilon\\) denotes the empty string.

In [2] it was shown that this satisfies the triangle inequality and vanishes when \\(x = y\\) with a negligible error term. The proof of its universality was given in [3], and the proof that it obeys the normalization property is more technical (for details, see [1; 2]).

The numerator can be rewritten in the form \\( max\{K(xy) - K(x); K(yx) - K(y)\} \\) within logarithmic additive precision due to the additive property of prefix Kolmogorov complexity [1]. Thus we get
\\[ D(x,y) = \frac{K(xy)-min\{K(x),K(y)\}}{max\{K(x),K(y)\}} \\]

Since the Kolmogorov complexity cannot be computed, it has to be approximated, and for this purpose, common file compressors are employed. Let \\(C(x)\\) be the length of the compressed string compressed by a particular compressor like gzip or arj. Then the approximation for the information distance \\(E\\) can be obtained by using the following formula:

\\[CBD(x,y) = \frac{C(xy)−min⁡\{C(x), C(y)\}}{max⁡\{C(x), C(y)\}}\\]

A CBM is a metric up to an additive constant and satisfies the normalization property if \\(C\\) satisfies the following properties up to an additive term:
1. Idempotency: \\(C(xx) = C(x)\\)
2. Symmetry: \\(C(xy) = C(yx)\\)
3. Monotonicity: \\(C(xy) \geq C(x)\\)
4. Distributivity: \\(C(xy) + C(z) \leq C(xy) + C(yz)\\)

The proof can be found in Theorem 6.2 of [1]. A compressor satisfying these properties is called a normal compressor.
There is no bound for the difference between the information distance and its approximation; that is, |E-CBD| is unbounded. For example, the Kolmogorov complexity of the first few million digits of the number \\(\pi\\), denoted by pi as a string, is a small constant because its digits can be generated by a simple program but \\(C(pi)\\) is proportional to \\(l(pi)\\) for every known text compressor \\(C\\).

###### Hierarchical Clustering

It is a greedy algorithm to construct a hierarchy of clusters. This algorithm start with a set of items and considers each items as a cluster containing a single element. Then it recursively merges the two closest clusters into a new one in each iteration until a single cluster is obtained. In this method there are two criteria: distance function to measure the dissimilarity over data and a linking criterion which finds the two closest clusters. Distance functions have been discussed on the previous lecture. The linkage criterion defines the distance between two clusters. It is often defined as either 1) maximal distance between the elements of two clusters (\\(d(A,B) = max\{d(a,b): a \in A, b \in B\}\\), \\(A\\) and \\(B\\) are two clusters), 2) minimal distance between the elements of two clusters (\\(d(A,B) = min\{d(a,b): a \in A, b \in B\}\\), \\(A\\) and \\(B\\) are two clusters), or the average (\\(d(A,B) = \frac{1}{|A||B|}\sum_{a\in A,b \in B}d(a,b)\\)), \\(A\\) and \\(B\\) are two clusters).

The pseudocode of hierarchical clustering:
	1. Start by assigning each item to its own cluster, so that if you have \\(N\\) items, you now have \\(N\\) clusters, each containing just one item. Let the distances (similarities) between the clusters equal the distances (similarities) between the items they contain.
	2. Find the closest (most similar) pair of clusters and merge them into a single cluster, so that now you have one less cluster.
	3. Compute distances (similarities) between the new cluster and each of the old clusters.
    4. Repeat steps 2 and 3 until all items are clustered into a single cluster of size \\(N\\).

(Source: [http://www.analytictech.com/networks/hiclus.htm](http://www.analytictech.com/networks/hiclus.htm))

Hierarchical clustering is often represented by a tree structure, called dendrogram. In this representation, leaves represent the objects itself, each inner node of the tree represents a cluster to which the data belong. 


##### Applications of Compression-based Distances (CBDs)

###### Language Trees:

The text corpora "The Universal Declaration of Human Rights" were downloaded in 52 Asian-European Languages from the website of the United Nations. The Lempel-Ziv compressor was used in the distance metric.
The resulted dendrogram of the hierarchical classification can be seen below. Note that how well the language groups can be recognized.

###### Russian Writers:

Some set of novels on the original Cyrillic letters written by few Russian writer were clustered. The resulted hierarchical clusters clearly shows that the novels written by the same author are grouped together.

###### Russian writers translated to English:

A bunch of novels written by Russian writes and translated to English were clustered by compression based methods. Notice that how well the novels translated by the same translator grouped together.

###### Clustering hand written digits.

The compression based distance measure has been evaluated on 30 hand written digits taken from the NIST Special Database 19. dataset. The images of the digits are black-and-white pictures '#' represents black and '.' represents white pixels. Each digit is 128x128 pixel.
The data which was used are:

The clustering obtained is:



###### Clustering of mitochondrial DNA.

The evolutionary tree has been reconstructed from the mitochondrial DNA of some mammals. Such reconstruction traditionally required a multiple alignment of the DNA of the species which is often cumbersome. The compression based distance measure does not require the absence of such multiple alignment and it can work on the raw DNA sequences. The reconstructed phylogenetic tree of some mammals can be seen below.


(Note, that for evolutionary reconstruction, often the mitochondrial DNA is used. This is inherited only from the mother, and this DNA does not mix with the DNA of the fathers).


###### The Google Similarity Distance (2007).

The idea is similar to CBDs, but instead of measuring how two strings are related to each other based on the mutual information in them, the Google Similarity Distance (GSD) uses the Google to measure how closely two strings (words) related to each other. Let x be a search term to Google, and let \\(\Gamma\\) is denoted by \\(M = |\Gamma|\\). Let \\(S_{x} \subseteq \Gamma\\) the set of documents which contains the term \\(x\\), i.e. the documents which are retrieved by Google when searching for \\(x\\) (note that the ordering is not considered here), and let \\(f(x) = |S_{x}|\\). For a pair of search term, \\(x,y\\) let \\(S_{x,y}\\) the set of pages containing both search terms, and \\(f(x,y) = |S_{x,y)\\). Let \\(N = \sum_{x,y} f(x,y)\\), the total number of pages containing two search terms. Note that, since one document contains at least one search term (indexed by Google) some documents counted more then once and hence \\(N \geq M\\).
In contrast to the compression based distances, where \\(C(x)\\) represents the length of the compressed strings, here we define the Google code \\(G\\) as
\\(G(x) = G(x,x), and G(x,y) = \log(\frac{1}{g(x,y)})\\), where \\(g(x,y) = \frac{f(x,y)}{N}\\).
The normalized Google distance NGD for search terms \\(x\\) and \\(y\\) is defined as follows:
\\[NGD(x,y) = \frac{G(x,y) - min(G(x), G(y))}{max(G(x), G(y))} = \frac{max\{\log f(x), \log f(y)\} - \log f(x,y)}{\log N - min\{\log f(x), \log f(y)\}}\\]

Some properties:
1. If \\(x = y\\), or \\(x \neq y\\) but \\(f(x) = f(y) = f(x,y) > 0\\), then \\(NGD(x,y) = 0\\)
2. If \\(f(x) = 0\\), then for each search term \\(y: f(x,y) = 0\\), then \\(NGD(x,y) = \frac{\inf}{\inf}\\)
3. \\(NGD(x,x) = 0\\), for all search term \\(x\\).
4. NGD is symmetric
5. \\(NGD(x,y) = 0\\) for some \\(x \neq y\\). Choose \\(x\\) and \\(y\\) such that \\(x \neq y\\) but \\(S_{x} = S_{y}\\)



###### Some conclusions

In general, compression based distances have been demonstrated working well on a broad range of applications, and string compressors are very fast compared to e.g. edit distance-based measures. However, in my opinion, problem specific measures (such as Needleman-Wunch, or Smith-Waterman algorithms for biological sequences) perform better because usually they contain some domain specific information encoded into algorithm, while compression based distances do not utilize any specific information.


##### References:
	1. R. Cilibrasi and P. M. B. Vitányi. Clustering by compression. IEEE Transactions on Information Theory, 51(4):1523–1545, 2005.
	2. Ming Li, Xin Chen, Xin Li, Bin Ma, and Paul Vitányi. The similarity metric. In SODA ’03: Proceedings of the fourteenth annual ACM-SIAM symposium on Discrete algorithms, pages 863–872, Philadelphia, PA, USA, 2003. Society for Industrial and Applied Mathematics.
	3. Ming Li. Information distance and its applications. In Oscar H. Ibarra and Hsu-Chun Yen, editors, CIAA, volume 4094 of Lecture Notes in Computer Science, pages 1–9. Springer, 2006.
	4. Cilibrasi and P. M. B. Vitányi. The Google Similarity Measures, IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 19, NO 3, MARCH 2007, 370–383