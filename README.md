# Problem: Stamp image classification

### 해커톤 홈페이지
https://ai-rush.com/
### 깃헙 페이지
https://github.com/ai-rush-2019

<br>

## Overview
About 700,000 [LINE sticker](https://store.line.me/stickershop/home/user/en) images are manually classified by 350 tags that indicate their characteristics.
Build a model that can predict one tags for each sticker image.

<br>

![image](https://user-images.githubusercontent.com/4004593/62457739-5e90f580-b7b6-11e9-96fb-10c4abae39f1.png)

![image](https://user-images.githubusercontent.com/4004593/62457778-6e103e80-b7b6-11e9-86a9-b135471bba33.png)

<br>

* pytorch code for image classification problem. you can find simple notebook version [here](https://www.kaggle.com/yangsaewon/pytorch-baseline-updated-7-10).

* setup - docker image (https://hub.docker.com/r/nsml/ml)

* trained 5 kfold ensemble of resnet50 models with StepLR learning rate scheduler. can be reused for other image classification problems.

* note that only 1 tag label is used for training and evaluation not multi labeling (dataset not provided due to license) 

* code is based on **NSML platform**.
   * https://n-clair.github.io/airush-docs/_build/html/ko_KR/index.html
