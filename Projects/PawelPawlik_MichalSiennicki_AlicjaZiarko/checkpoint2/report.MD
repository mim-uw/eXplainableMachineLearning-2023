# 10. *Test CRP (and LRP) on ResNets classifying medical images, e.g. lung or skin lesions
## Checkpoint 2 - report
Paweł Pawlik, Michał Siennicki, Alicja Ziarko

# What was done
 - Implementing of datasets loading
 - Training pipeline with pytorch lightning, hydra, wandb
 - LRP explanation

# What are the difficulties
- On covid lungs dataset the training is not working - seems like the task of predicting if it is covid or not is too hard
- On menaloma dataset the training is not as stable as we wish. We don't want to use the tabular data (image only) as opposed to best kaggle solutions. 

# What will be done next
- Improving the menaloma training variance - more stable results are wanted
- Rerun LRP results
  - Idea: Check TTA in interpretation (random rotations of input, calculate intepretations, calculate their average)
- Get CRP explanations
- Evaluate with the Quantus package