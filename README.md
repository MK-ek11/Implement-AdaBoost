# Implement AdaBoost with Decision Stump
## Description
Course: Data Mining and Knowledge Discovery (Fall 2021) <br />
Task: Write own program


>Boosting is an iterative procedure used to adaptively change the distribution of training examples for learning base classiﬁers so that they increasingly focus on examples that are hard to classify. Unlike bagging, boosting assigns a weight to each training example and may adaptively change the weight at the end of each boosting round.
>
> *Tan, Pang-Ning, et al. Introduction to Data Mining EBook: Global Edition, Pearson Education, Limited, 2019.*

>If a tuple was incorrectly classified, its weight is increased. If a tuple was correctly classified, its weight is decreased. A tuple’s weight reflects how difficult it is to classify—the higher the weight, the more often it has been misclassified. These weights will be used to generate the training samples for the classifier of the next round. The basic idea is that when we build a classifier, we want it to focus more on the misclassified tuples of the previous round. Some classifiers may be better at classifying some “difficult” tuples than others. In this way, we build a series of classifiers that complement each other.
>
> *J. Han, Jian Pei, and Micheline Kamber, Data mining: concepts and techniques. S.l: Elsevier Science, 2011.*


### Task:
![Screenshot 2023-03-10 164234](https://user-images.githubusercontent.com/101310529/224269020-51824782-cacb-41dd-85df-94e276e8f536.png)


### Answer:
![Screenshot 2023-03-10 165110](https://user-images.githubusercontent.com/101310529/224269256-aed414b9-d8aa-49be-a87b-d33787837fa0.png)
