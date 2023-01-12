# Homework 6

The purpose of the sixth homework is to learn about the method of calculating fairness statistics. Calculate these statistics for a dataset of 'credit scoring' or 'adult'

## Deadline 

2022-11-24 EOD

## Task

If there are no age, gender or race attributes in the selected dataset then for the following tasks train models on one of the datasets

- credit scoring https://www.kaggle.com/competitions/GiveMeSomeCredit/
- adult https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

Prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW6`.

1. train a model for the selected dataset
2. for the selected protected attribute (age, gender, race) calculate the following coefficients: Statistical parity, Equal opportunity, Predictive parity.
3. Train another model and see how the coefficients Statistical parity, Equal opportunity, Predictive parity behave for it.
4. Apply the selected bias mitigation technique on the first model. Check how Statistical parity, Equal opportunity, Predictive parity coefficients behave after this correction.
5. Compare for the three models quality (performance) with fairness coefficients. Is there any correlation?



## **Important note:**

The submitted homework should consist of two parts (try to render html file out of your jupiter notebook). 

The first part is the key results and comments from points 3,4,7. In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (IMAGES, COMMENTS).**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make this homework more readable, 2. will create good habits related to reporting.

