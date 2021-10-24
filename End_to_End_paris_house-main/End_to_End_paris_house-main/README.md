# Paris House Price Predictions
### This is End to End project

## Life Cycle of Data Scientist:
#### i) EDA
#### ii) Feature Engineering
#### iii) Feature Selection
#### iv) Model Building
#### v) Model Deployment




## Overview of my Model.
### EDA
###### * In EDA,I have identified  Target variable is increasing linearly based on increasing independent variable. 
![image](https://user-images.githubusercontent.com/85152278/132151730-4092ad90-c7a7-46ca-a56e-17a097557e92.png)
###### * So, I have planned to find whose Feature follows the Gaussian Distribution, and not follow the Normal Distribution.

### Feature Engineernig 
##### * In, Feature Enginneing, I have scale down the values based on the feature following Gaussian, and not following the normal distribution. 
##### * And, I have Identified features are not correlated with each other, so it reduce our working load on feature enginnering. 
![image](https://user-images.githubusercontent.com/85152278/132151904-d2a8d912-b2f5-4e19-a736-32a96dd6ce51.png)

### Featrure Selection
##### * In feature selection, I used mutual info regreesion coz it works well in regression problems well. 
##### * And I choosed first 8 based on my Threshold value. 

### Model Building
##### * In model Building, I used various models like linear, kmeans, xgboost and many.......
##### * And maintained a DataFrame of r2 and accuarcy score for all the models. 
##### * And choosed the model, and saved it in local system as a pickle file. 

### Model Deployment
##### * And, using the Pickle file created a one model.
##### * Created a Flask with Swagger API and tested in the local machine works well.
##### * And deployed in my local machine.
##### The Paris zip file is pikle file, you can download or if you run the whole ipynb notebook it will automatically save in local device. 

