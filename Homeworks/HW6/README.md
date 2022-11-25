# Homework 6

The purpose of the sixth homework is to learn about the method of calculating fairness statistics. Calculate these statistics for a dataset of 'credit scoring' or 'adult income'.

## Deadline 

2022-12-08 23:59

## Task

For this homework, train models on one of the following datasets:

- credit scoring https://www.kaggle.com/competitions/GiveMeSomeCredit/
- adult income https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

Prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW6`.

1. Train a model for the selected dataset.
2. For the selected protected attribute (age, gender, race) calculate the following fairness coefficients: Statistical parity, Equal opportunity, Predictive parity.
3. Train another model (different hyperparameters, feature transformations etc.) and see how the coefficients Statistical parity, Equal opportunity, Predictive parity behave for it.
4. Apply the selected bias mitigation technique on the first model. Check how Statistical parity, Equal opportunity, Predictive parity coefficients behave after this correction.
5. Compare the quality (performance) of the three models with their fairness coefficients. Is there any correlation?
6. ! COMMENT on the results obtained in (2)-(5)



## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (2)-(5). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5).

Such division: 1. will make this homework more readable, and 2. will develop good habits related to reporting.

