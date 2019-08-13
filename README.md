# :tophat:RedHat-OSPO:tophat:
This repository is created for my internship project in Red Hat AICoE collaborated with OSPO. For a detailed report, please check *OSPO-report.pdf*.      

![Alt Text](https://msdnshared.blob.core.windows.net/media/2016/04/shadowman.png)

# OSPO Diversity: Open-source Projects Contributors Affiliation    

## :one:Background    

Red Hat is an active leader in open source projects and Red Hat is also involved in maintaining a healthy open source community. The health and sustainability of an open source community is co-determined by a lot of factors, as CHAOSS project metrics displays. Diversity of community contributors within projects is pertinent to Red Hat’s business as well as open-source projects that are beneficial to the ecosystem. 
Usually, the level of participation of each company’s employees (especially Redhatters) in open-source projects can be issued by the email domain of participants. However, Red Hat employees are not required to use “redhat.com” email domain when they’re contributing to projects so not every contributor uses their work email in project participation. Also, Redhatters may have moved between companies during time. 
Therefore, it’s desirable to improve the ability to determine and classify contributors’ identity and affiliated organization with machine learning methods. 

## :two:Project Goals    

The project consists of building a better identification of the following aspects:
* Is the contributor a full-time employee or a volunteer
* Is the contributor a Redhatter or a non-Redhatter
* What’s the contributor’s affiliated company if he/she is a non-Redhatter

## :three:Exploratory Analysis
The original dataset is a 22-gigabytes git log data. I've completed following steps for insights exploration:    
- Committer and author check
- Affiliation assignment based on email domain
- Commit date parsing
- Popular projects 
- Comparison between different groups 
- Clustering 
- String comparison on email address    

## :four:Feature Engineering    
Based on those previous steps of exploratory analysis, I’ve created or added some potentially influencing features that can be missed out if I simply apply original dataset to model. The insights of creating and transforming are from the initial exploration results and common sense.     
- Percentage made on each day / each part of day 
- Committer’s contribution in each project
- Nearest neighbor's affiliation
- Encodings for repos name    

## :five:Modeling 

### "Naive" Classification    
As for modeling, this problem is defined to be a classification problem. With experiments with SVM and Neural Network with sigmoid function, I prove that simple classification structure doesn’t work.     

### Generative Adversarial Networks    
This method works perfectly in our project setting. Firstly, the current dataset only has one side of data labeled, which is the Red Hat employee side. This will solve imbalanced problem and will not require more data on Volunteer side. Secondly, the goal of model is to output the probability of each committer being a Red Hat employee, and GAN’s output is probabilities.     

### Pseudo-Labeling     
When more data labeled as volunteer is gathered. Pseudo-Labeling can be another scope of analysis besides GAN, and can be used to compare with GAN’s results. Pseudo-Labeling is one of the most efficient and famous methods in semi-supervised learning. 

## :six:Acknowledgement    
This project is supported by OSPO in Red Hat. I’d like to thank Brian Profitt, Sherard Griffin, Sanjay Aurora, Michael Clifford, and Prasanth Anbalagan who provided insight and expertise that greatly assisted the project.     

## :seven:Reference

* Background    
Community Health Analytics Open Source Software Project: https://chaoss.community/    
Red Hat Projects on Github: https://redhatofficial.github.io/#!/main    
* Modeling     
Tutorials on GAN: https://github.com/rionaldichandraseta/gan-on-structured-data/blob/master/gan/notebooks/gan-toy.ipynb; https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py    
Tutorial on TensorBoard: https://www.tensorflow.org/guide/summaries_and_tensorboard    
Tutorials on Pseudo-Labeling: https://datawhatnow.com/pseudo-labeling-semi-supervised-learning/


