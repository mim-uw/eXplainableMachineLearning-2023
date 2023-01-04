# Homework 4

The purpose of the fourth homework is to become familiar with the PDP method and its variants. Calculate these explanations on the model from the first homework and gather conclusions from the results.

Add your fourth homework as a pull request to this folder.

## Deadline 

2022-11-10 EOD

## Task

For selected models (you can use models from Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW4`.


1. For the selected data set, train at least one tree-based ensemble model (random forest, gbm, catboost or any other boosting)
2. for some observations from this dataset, calculate the model predictions for model (1)
3. for observations selected in (2), calculate the explanation of model prediction using Ceteris paribus profiles (packages for python: `AIX360`, `dalex`, `PDPbox`, own code, packages for R: `DALEX`, `ALEPlot`).
4. find two observations in the data set, such that they have different CP profiles (e.g. model response is growing with age for one observation and lowering with age for another). Note that you need to have a model with interactions to have such differences
5. for the selected model calculate PDP explanations
6. train the second model (of any class, neural nets, linear, other boosting) and find an observation for which PDP profiles are different between the models
7. Comment on the results for points (4), (5) and (6)


## **Important note:**

The submitted homework should consist of two parts (try to render html file out of your jupiter notebook). 

The first part is the key results and comments from points 3,4,7. In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (IMAGES, COMMENTS).**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make this homework more readable, 2. will create good habits related to reporting.
