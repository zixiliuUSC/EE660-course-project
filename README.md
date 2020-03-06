# Amazon Review Sentiment Analysis
> This project is to build a natural language processing system fitting the relation between Amazon review text and the coresponding product rating. 

[![sklearn Version][sklearn-image]][sklearn-url]
[![Downloads Stats][sklearn-downloads]][sklearn-url]
[![Gensim Version][gensim-image]][gensim-url]
[![NLTK Version][NLTK-image]][NLTK-url]
[![numpy version][numpy-image]][numpy-url]

In this project, I use two different language models as feature extraction methods which are TF-IDF and word2vec. Among them, word2vec model is to learn semantic vectors of words by using an unsupervised machine learning model based on 2-layer perceptron classification machine. TF-IDF model is term frequency counting model and I will use PCA algorithm to reduce feature dimension. In order to deal with class imbalance, I use SMOTE techniques to do re-sampling. In the classification, I fine tune and compare the performance between logistic regression, linear regression, decision tree, Adaboost and Gaussian Naive Bayes. In some of this technique, I also use regularization method to do feature reduction. Finally, the evaluation is mainly use F1-macro score for overall performance comparison and F1 score for comparing the performance in each class.

![](header.png)

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[sklearn-image]: https://img.shields.io/badge/sklearn-0.21.3-blue
[sklearn-url]: https://scikit-learn.org/stable/
[sklearn-downloads]: https://img.shields.io/badge/sklearn-download-blue
[wiki]: https://github.com/yourname/yourproject/wiki
[gensim-image]:https://img.shields.io/badge/gensim-3.8.1-brightgreen
[gensim-url]: https://radimrehurek.com/gensim/
[NLTK-image]:https://img.shields.io/badge/NLTK-3.4.5-yellow
[NLTK-url]:https://www.nltk.org/
[numpy-image]:https://img.shields.io/badge/numpy-1.16.2-orange
[numpy-url]:https://numpy.org/
