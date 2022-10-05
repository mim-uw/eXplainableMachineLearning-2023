# Homework 2

The goal of the second homework is to become familiar with the SHAP method. Calculate these explanations on the model from the first homework and gather conclusions from the results.

Add your second homework to this folder.

## Deadline 

2022-10-20 EOD

## Task

For selected models (you can use models created in Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW2`.

1. For the selected data set, train at least two tree-based ensemble models (random forest, gbm, catboost or any other boosting)
2. for some selected observations (two or three) from the selected dataset, calculate predictions for models selected in point (1)
3. for observations selected in (2), calculate the decomposition of model prediction using SHAP, Break Down or both (packages for python: `dalex`, `shap`, packages for R: `DALEX`, `iml`).
4. find two observations in the data set, such that they have different variables of the highest importance (e.g. age and gender are the most important for observation A, but race and class for observation B)
5. (if possible) select one variable and find two observations in the data set such that for one observation this variable has a positive effect and for the other a negative effect
6. train a second model (of any class, neural nets, linear, other boosting) and find an observation for which BD/shap attributions are different between the models
7. Comment on the results for points (4), (5) and (6)


## **Important note:**

The submitted homework should consist of two parts (try to render html file out of your jupiter notebook). 

The first part is the key results and comments from points 3,4,7. In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (IMAGES, COMMENTS).**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make this homework more readable, 2. will create good habits related to reporting.
