# eXplainable Machine Learning / Wyjaśnialne Uczenie Maszynowe - 2023

[**eXplainable Machine Learning**](https://usosweb.uw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1000-319bEML) course for Machine Learning (MSc) studies at the University of Warsaw. 

Winter semester 2022/23 [@pbiecek](https://github.com/pbiecek) [@hbaniecki](https://github.com/hbaniecki)


## Design Principles

The course consists of lectures, computer labs and a project.

The design of this course is based on four principles:

- Mixing experiences during studies is good. It allows you to generate more ideas. Also, in mixed groups, we can improve our communication skills,
- In eXplainable AI (XAI), the interface/esthetic of the solution is important. Like earlier Human-Computer Interaction (HCI), XAI is on the borderline between technical, domain and cognitive aspects. Therefore, apart from the purely technical descriptions, the results must be grounded in the domain and should be communicated aesthetically and legibly. 
- Communication of results is important. Both in science and business, it is essential to be able to present the results concisely and legibly. In this course, it should translate into the ability to describe one XAI challenge in the form of a short report/article.
- It is worth doing valuable things. Let's look for new applications for XAI methods discussed on typical predictive problems.


## Meetings

Plan for the winter semester 2022/2023. UW classes are on Fridays. 


* 2022-10-07  -- Introduction, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/TrustworthyMachineLearning-2023/main/Lectures/01_introduction.html#/title-slide), [audio](https://youtu.be/1UkrvKyvMDw)
* 2022-10-14  -- Break-Down / SHAP, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2023/main/Lectures/02_shap.html#/title-slide), [audio](https://youtu.be/SJQWAJLhMas), [code examples](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw2_shap_with_xgboost_on_titanic.html)
* 2022-10-21  -- LIME / LORE, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2023/main/Lectures/03_lime.html), [audio](https://youtu.be/l5I1uwoKrME), [code examples](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw3_lime_with_xgboost_on_titanic.html)
* 2022-10-28  -- CP / PDP, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2023/main/Lectures/04_pdp.html#/title-slide), [code examples](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw4_cp_and_pdp_with_xgboost_on_titanic.html)
* 2022-11-04  -- PROJECT: **First checkpoint** - Choose a topic and be familiar with the attached materials.
* 2022-11-18  -- VIP / MCR, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2023/main/Lectures/05_vip.html#/title-slide), [audio](https://youtu.be/6IU4kMv2x9Y), [code examples](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw5_pvi_with_xgboost_on_titanic.html)
* 2022-11-25  -- Fairness, [slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2023/main/Lectures/06_fairness.html#/title-slide), [audio](https://youtu.be/OdPW06tx_Yk)
* 2022-12-02  -- Explanations for neural networks & Evaluation of explanations 
* 2022-12-09  -- PROJECT: **Second checkpoint** - Provide initial experimental results and/or code implementation.
* 2022-12-16  -- Counterfactual explanations (?)
* 2022-12-22  -- Concept based explanations (?)
* 2023-01-13  -- Student presentations
* 2023-01-20  -- Student presentations
* 2023-01-27  -- PROJECT:  **Final presentation** - Present final experimental results and/or code implementation.


## How to get a good grade

From different activities, you can get from 0 to 100 points. 51 points are needed to pass this course.

Grades:

* 51-60: (3) dst
* 61-70: (3.5) dst+
* 71-80: (4) db
* 81-90: (4.5) db+
* 91-100: (5) bdb


There are four key components:

* Homeworks (0-24)
* Presentations (0-10)
* Project (0-36)
* Exam  (0-30)

## Homeworks (24 points)

 - [Homework 1](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW1)  for 0-4 points. **Deadline: 2022-10-13**
 - [Homework 2](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW2)  for 0-4 points. **Deadline: 2022-10-20** 
 - [Homework 3](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW3)  for 0-4 points. **Deadline: 2022-10-27**
 - [Homework 4](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW4)  for 0-4 points. **Deadline: 2022-11-17**
 - [Homework 5](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW5)  for 0-4 points. **Deadline: 2022-11-24**
 - [Homework 6](https://github.com/mim-uw/TrustworthyMachineLearning-2023/tree/main/Homeworks/HW6)  for 0-4 points. **Deadline: 2022-12-08**

## Presentations (10 points)

Presentations can be prepared by one or two students. Each group should present a single paper related to XAI published in the last 3 years (journal or conference). Each group should choose a different paper. Here are some suggestions:


* J. DeYoung et al. [ERASER : A Benchmark to Evaluate Rationalized NLP Models](https://aclanthology.org/2020.acl-main.408.pdf). ACL, 2020.
* D. Slack et al. [Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods](https://dl.acm.org/doi/10.1145/3375627.3375830). AIES, 2020.
* S. Lundberg et al. [From local explanations to global understanding with explainable AI for trees](https://www.nature.com/articles/s42256-019-0138-9). Nature Machine Intelligence, 2020.
* U. Bhatt et al. [Evaluating and Aggregating Feature-based Model Explanations](https://www.ijcai.org/Proceedings/2020/0417). IJCAI, 2020.
* J. Adebayo et al. [Debugging Tests for Model Explanations](https://proceedings.neurips.cc/paper/2020/hash/075b051ec3d22dac7b33f788da631fd4-Abstract.html). NeurIPS, 2020.
* S. Sinha et al. [Perturbing Inputs for Fragile Interpretations in Deep Natural Language Processing](https://arxiv.org/abs/2108.04990). EMNLP BlackboxNLP Workshop, 2021.
* L. Yang et al. [Synthetic Benchmarks for Scientific Research in Explainable Machine Learning](https://openreview.net/forum?id=R7vr14ffhF9). NeurIPS Datasets and Benchmarks, 2021.
* M. Neely et al. [Order in the Court: Explainable AI Methods Prone to Disagreement](https://arxiv.org/abs/2105.03287). ICML XAI Workshop, 2021.
* F. Poursabzi-Sangdeh et al. [Manipulating and Measuring Model Interpretability](https://arxiv.org/abs/1802.07810v5). CHI, 2021.
* X. Zhao et al. [BayLIME: Bayesian Local Interpretable Model-Agnostic Explanations](https://proceedings.mlr.press/v161/zhao21a.html). UAI, 2021.
* Y. Zhou et al. [Do Feature Attribution Methods Correctly Attribute Features?](https://ojs.aaai.org/index.php/AAAI/article/view/21196). AAAI, 2022.
* J. Adebayo et al. [Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation](https://openreview.net/forum?id=xNOVfCCvDpM). ICLR, 2022.
* Papers from: [NeurIPS 2021 Workshop on eXplainable AI approaches for debugging and diagnosis](https://xai4debugging.github.io/)

// More suggestions for computer vision:

* Z. Huang & Y. Li. [Interpretable and Accurate Fine-grained Recognition via Region Grouping](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_Interpretable_and_Accurate_Fine-grained_Recognition_via_Region_Grouping_CVPR_2020_paper.html). CVPR, 2020.
* K. Hanawa et al. [Evaluation of Similarity-based Explanations](https://openreview.net/forum?id=9uvhpyQwzM_). ICLR, 2021.
* A. Kapishnikov et al. [Guided Integrated Gradients: An Adaptive Path Method for Removing Noise](https://arxiv.org/abs/2106.09788). CVPR, 2021.
* V. Petsiuk et al. [Black-Box Explanation of Object Detectors via Saliency Maps](https://openaccess.thecvf.com/content/CVPR2021/html/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.html). CVPR, 2021.
* L. Arras et al. [CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations](https://doi.org/10.1016/j.inffus.2021.11.008). Information Fusion, 2022.
* A. Khakzar et al. [Do Explanations Explain? Model Knows Best](https://openaccess.thecvf.com/content/CVPR2022/html/Khakzar_Do_Explanations_Explain_Model_Knows_Best_CVPR_2022_paper.html). CVPR, 2022.
* S. Chen & Q. Zhao. [REX: Reasoning-Aware and Grounded Explanation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_REX_Reasoning-Aware_and_Grounded_Explanation_CVPR_2022_paper.html). CVPR, 2022.
* T. Makino et al. [Differences between human and machine perception in medical diagnosis](https://www.nature.com/articles/s41598-022-10526-z). Scientific Reports, 2022.
* M. Watson et al. [Agree to Disagree: When Deep Learning Models With Identical Architectures Produce Distinct Explanations](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9706847). WACV, 2022.
* Q. Zheng et al. [Shap-CAM: Visual Explanations for Convolutional Neural Networks based on Shapley Value](https://arxiv.org/abs/2208.03608). ECCV, 2022.
* XAI-related papers from:
    * [CVPR 2022 Workshop on Fair, Data-Efficient, and Trusted Computer Vision](https://openaccess.thecvf.com/CVPR2022_workshops/FaDE-TCV)
    * [CVPR 2021 Workshop on Fair, Data-Efficient, and Trusted Computer Vision](https://openaccess.thecvf.com/CVPR2021_workshops/TCV) 

## Project (36 points)

[List of topics](https://docs.google.com/document/d/15lqyxRtolxBgZjDWs81ISXnJ6y0vUOXp8KcQeug1s3g/edit?usp=sharing)

XAI stories ebook (previous editions): 
- first edition https://pbiecek.github.io/xai_stories/
- second edition https://pbiecek.github.io/xai_stories_2/
<!-- - this edition https://github.com/pbiecek/xai_stories_3/ -->

## Exam (30 points)

A written exam will consist of simple exercises based on the materials from Lectures and Homeworks.

## Literature

We recommend to dive deep into the following books and explore their references on a particular topic of interest:

* [Explanatory Model Analysis. Explore, Explain and Examine Predictive Models](https://pbiecek.github.io/ema/) by Przemysław Biecek, Tomasz Burzykowski
* [Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/) by Solon Barocas, Moritz Hardt, Arvind Narayanan
* [Interpretable Machine Learning. A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar
