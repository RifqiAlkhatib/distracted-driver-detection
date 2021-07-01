# Distracted Driver Detection

This project aims to create an image classification model using transfer learning & fine-tuning of pre-trained Convolutional Neural Network (CNN) models.

## Background

Compared to other similar cities in first world countries, Singapore has one of the highest [road fatality rates per motor vehicle](https://www.budgetdirect.com.sg/car-insurance/research/road-accident-statistics-in-singapore). One of the main causes of this is distracted driving - in 2016, over 40% of road traffic fatalities here occurred due to [drivers failing to keep a proper lookout](https://www.budgetdirect.com.sg/blog/car-insurance/distracted-driving-the-cost-and-6-ways-to-stay-safe) on roads. Drivers who are using their mobile phones, grabbing a drink or even talking to their fellow passengers tend to pay less attention to their surroundings while driving.

Despite enforcement efforts such as the [ban of using mobile phones while driving](https://kwiksure.sg/blog/distracted-driving/), the problem seems to be quite entrenched here. In fact, a survey showed that over 80% of drivers admitted to [using their mobile phones while driving](https://www.todayonline.com/singapore/83-singapore-drivers-use-their-mobile-phones-while-driving-survey). Ironically, the same survey also showed that over 90% of drivers felt that it was an unsafe practice.

## Problem Statement

In order to improve these statistics and increase road safety, we want to explore whether dashboard cameras can be used to detect drivers who are distracted. We will attempt to create an image classifer model that can detect various distraction states of the driver based on images of the driver.

## Executive Summary

We built a Convolutional Neural Network (CNN) model based on data from a [Kaggle competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview) by State Farm. The data consists of images of drivers split into 10 classes, each representing various states of distraction. Our model aims to predict which class the images belong to and will be evaluated on the test dataset from Kaggle.

We first preprocessed our train and validation datasets in order for our model to achieve the best performance on the test dataset. We used transfer learning with fine-tuning on two pre-trained CNN models - VGG16 & EfficientNetB4. The models were evaluated on multiclass log loss and EfficientNetB4 was chosen as the best model, achieving a score of 0.599. An ensemble method was then used to produce the final predictions for Kaggle, achieving a score of **0.301**.

The performance of the model was then tested on unseen external images to further evaluate its practicality for real world usage.

## Data Dictionary

This repo comprises 6 .csv files - `drivers_img_list.csv` consists of information about the train dataset, while the other 5 files are the predictions for submission to Kaggle.

| File               | Column    | Type   | Description                                                                       |
|--------------------|-----------|--------|-----------------------------------------------------------------------------------|
| `drivers_img_list` | subject   | string | Unique ID of the driver in the image                                              |
| `drivers_img_list` | classname | string | Class of distraction state the image belongs to                                   |
| `drivers_img_list` | img       | string | File name of the .jpg image                                                       |
| `submission`       | img       | string | File name of the .jpg image                                                       |
| `submission`       | c0 - c9   | float  | Predicted probability of the image belonging to class `c0` to   `c9` respectively |
