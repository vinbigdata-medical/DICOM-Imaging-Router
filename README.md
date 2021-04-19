# Automatic Classification of Human Body Parts from X-ray Images Using Deep Convolutional Neural Networks


This repository contains the training code for our paper entitled "Automatic Classification of Human Body Parts from X-ray Images Using Deep Convolutional Neural Networks", which was submitted and under review by [Medical Imaging with Deep Learning 2021 (MIDL2021)](https://2021.midl.io/).

# Abstract  
X-ray imaging is the most commonly used imaging modality in clinical practice, resultingin vast,  non-normalized databases of X-ray images.  This leads to an obstacle in the de-ployment of artificial intelligence (AI) solutions for analyzing medical images, which oftenrequires identifying the right body part before feeding the image into a specified AI model.This  challenge  raises  the  need  for  an  automated  and  efficient  approach  to  classify  bodyparts from X-ray scans.  To tackle this problem, this paper aims to deploy deep convolu-tional neural networks (CNNs) for categorizing unknown X-ray images into 5 anatomicalgroups:  abdominal, adult chest, pediatric chest, spine, and others.  To this end, a large-scale X-ray dataset consisting of 16,093 images has been collected and manually classified.We then trained a set of state-of-the-art deep CNNs using a training set of 11,263 images.These networks were then evaluated on an independent test set of 2,419 images and showedsuperior performance in classifying the body parts.  Specifically, our best performing model(i.e.  MobileNet-V1) achieved a recall of 0.982 (95% CI, 0.977–0.988), a precision of 0.985(95%  CI,  0.975–0.989)  and  a  F-1  score  of  0.981  (95%  CI,  0.976–0.987),  whilst  requiringless computation for inference (0.0295 second per image).  This remarkable performanceindicates that deep CNNs can accurately and effectively differentiate human body partsfrom X-ray scans, thereby providing potential benefits for a wide range of applications inclinical settings.  To encourage new advances, we will make the dataset, codes and trainedmodels publicly available upon the publication of this paper.  
# Preprocess X-ray image  

# Architecture  
![title](images/Pipeline.jpg)
