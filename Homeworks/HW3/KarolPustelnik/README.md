# Homework 3

The purpose of the third homework is to become familiar with the LIME method. Calculate these explanations on the model from the first homework and gather conclusions from the results.

Add your third homework to this folder.

## Deadline 

2022-10-27 EOD

## Task

For selected models (you can use models from Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW3`.

1. For the selected data set, train predictive models (or use models from Homework 1)
2. for some selected observations from this dataset, calculate the model predictions for model (1)
3. for an observation selected in (2), calculate the decomposition of model prediction using `LIME` or a similar technique (packages for python: `lime`, packages for R: `localModel`, `iml`).
4. compare LIME decompositions for different observations in the dataset. How stable are these explanations? 
5. Compare LIME explanations with explanations obtained with the SHAP method 
6. train a second model (of any class, neural nets, linear, other boosting) and find an observation for which LIME attributions are different between the models
7. Comment on the results obtained in (4), (5) and (6)


## **Important note:**

The submitted homework should consist of two parts (try to render html file out of your jupiter notebook). 

The first part is the key results and comments from points 3,4,7. In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (IMAGES, COMMENTS).**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make this homework more readable, 2. will create good habits related to reporting.
