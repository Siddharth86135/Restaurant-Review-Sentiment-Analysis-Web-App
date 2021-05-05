# Restaurant Review Sentiment Analysis Web Application

## **Predicting Restaurant Review Sentiment**

<img src="https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/blob/master/README-Resources/Restaurant.jpg" width=600>

Restaurant Review Sentiment Analysis is a project made for San Francisco State University CSC 849 Search Engines graduate course. Machine learning techniques and classical supervised learning algorithms were used. We first implemented a restaurant category classifier in order to predict the restaurant category label for a given set of restaurant reviews. We then use sentiment analysis to output the overall sentiment of a given user text review.

## Getting Started

The project uses Python 3.7.0 and the scikit-learn and pandas libraries. You must first install Python 3.7.0 in order to be able to run the project locally. You can download Python 3.7.0 here:    
https://www.python.org/downloads/release/python-370/

## **Predicting Restaurant Review Sentiment**

<img src="https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/blob/master/README-Resources/Restaurant.jpg" width=600>


---
## **Understanding The Project**

### **The Dataset**
***
![Dataset](https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/blob/master/README-Resources/Screenshot%20(96).png)

---
## **Overview**
* Dataset has **10000 rows** and **8 columns**.
* We have to predict whether a review is **"Positive"** or **"Negative"**.
* **PortStemmer** method has been used for **Stemming**.
* I have also tried **WordEmbedding** with **LSTM**.
* I have applied many different algorithms **LSTM**, **Bi-Directional LSTM**, **RandomForestClassifier**, **MultinomialNB**, **SVM** and **KNN**.

---


## Problem Definition and Algorithm

###  Task Definition

> To develop a machine learning model to detect different types of sentiments contained in a collection of English sentences or a large paragraph.

I have chosen Restaurant reviews as my topic. Thus, the objective of the model is to correctly identify the sentiments of the users by reviews which is an English paragraph and the result will be in positive or negative only.

For example, 

If the review given by the user is:
> “ We had lunch here a few times while on the island visiting family and friends. The servers here are just wonderful and have great memories it seems. We sat on the oceanfront patio and enjoyed the view with our delicious wine and lunch. Must try! ”

Then the model should detect that this is a positive review. Thus the output for this text will be **Positive**.

### Algorithm Definition

The data set which I chose for this problem is available on Kaggle. The sentiment analysis is a classification because the output should be either positive or negative. That is why I tried 3 of the classification algorithms on this data set.

* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Logistic Regression

i) Multinomial Naive Bayes:
Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of a feature.
Bayes theorem calculates probability P(c|x) where c is the class of the possible outcomes and x is the given instance which has to be classified, representing some certain features.

> P(c|x) = P(x|c) * P(c) / P(x)

Naive Bayes is mostly used in natural language processing (NLP) problems. Naive Bayes predict the tag of a text. They calculate the probability of each tag for a given text and then output the tag with the highest one.


ii) Bernoulli Naive Bayes
BernoulliNB implements the naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions; i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be represented as binary-valued feature vectors; if handed any other kind of data, a BernoulliNB instance may binarize its input (depending on the binarize parameter).

> The decision rule for Bernoulli naive Bayes is based on:- P ( x i ∣ y ) = P ( i ∣ y ) xi + ( 1 − P ( i ∣ y ) ) ( 1 − x i )

which differs from multinomial NB’s rule in that it explicitly penalizes the non-occurrence of a feature that is an indicator for class, where the multinomial variant would simply ignore a non-occurring feature.

In the case of text classification, word occurrence vectors (rather than word count vectors) may be used to train and use this classifier. BernoulliNB might perform better on some datasets, especially those with shorter documents. It is advisable to evaluate both models if time permits.


iii) Logistic Regression
Logistic regression is a supervised classification algorithm. In a classification problem, the target variable(or output), y, can take only discrete values for the given set of features(or inputs), X.

Contrary to popular belief, logistic regression is a regression model. The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the sigmoid function.
