# Report

I randomly selected two indices from the test dataset. The indices were 28 and 38.

## Random Forrest

I used radom forest as tree based model.

Probailty of heart attack for patient 28 equals: 55.38228180752807%
Probailty of heart attack for patient 38 equals: 23.03908323870259%

![Dalex chart for Random Forrest](https://user-images.githubusercontent.com/60492340/197866722-6211b7f6-dbc5-4340-b2bb-a9fcc506b3b8.png)

Because for both charts above the top variable has the highest absolute shaply value, so both chosen samples have different top two featrues. For the patient 28 these are *oldpeak* and *slp* and for patient 38 *oldpeak* and *thall*. This occures because for one sample one feature value can determine the output and in the other it will not. For example if you have a pool in your garden your very likely to be wealthy but not having a pool does not imply your poor beacause you may live in not warm enough climate.

One can see that for patient 28 the *thalachh* feature has positive shapley values and for patient 38 has negative. This kind of situations happen when different values of some input goes with different results of classification. For example if one person has low salary it implies that the person is not wealthy on the other hand high salary suggest that the person is wealathy.

![Shap chart for Random Forrest](https://user-images.githubusercontent.com/60492340/197866967-c39d7ae7-be38-43f6-b67b-9ed9a1beb6c1.png)

In this method (shap library) results are the same as in dalex library. I guess that for many features in the dataset there might be some differences between those libraries depending on the method of the estimation. Probably for explaining neural networks working on image input both metods may show different results.

## SVM

Probailty of heart attack for patient 28 equals: 32.8056375699065%
Probailty of heart attack for patient 38 equals: 39.61395034409673%

![Shap chart for SVM](https://user-images.githubusercontent.com/60492340/197867149-27d9c3ed-543f-4c49-a0b6-a29bff782caf.png)

Method SVM with rbf kernel works differentelly than decision trees. This causes different learning process, different classification conditions and different results. Probability scores calculated by two methods for patient 28 are extremly different. This also refers to shap values. Different features have main impact on the final output and that is why shaple values may differ. Many features are both negative or both positive for both methods which means that sometimes models work similarly but on the other hand for SVM second most important feature for patient 38 is *restecg2* which not apperas in top 9 impacing features of random forest. 
