# Telecom-Churn
KonnectIndia Cellular Services is a leading telecom services provider in India. It has a vast subscriber base, with most of them being pre-paid subscribers. The service provider has decided to predict the customer churn. 
The purpose of this assignment is to develop a supervisory machine learning model to meet the above objective. Once the model has been built, the service provider would be able to predict whether or not a customer would churn depending upon key factors such as consumption, bill payment criteria etc. Hence the provider can improve the retention rate by focussing more on the customers who are about to churn out. 
Method:
The following steps are taken to perform analytics on the data set
1.	Understanding the data
2.	Pre-processing
3.	Identifying the key metrics needed for the problem
4.	Applying the models
5.	Results
6.	Comparing the outcomes of the applied models
7.	Shortlisting the model
8.	Inferences

Understand the data:
•	The data provided has 110 attributes along with a target attribute (Embedding the data set for reference). All the independent attributes were numeric in nature. Each record in the data set refers to a customer data such as consumption, recharge value etc.
 

Bivariate Analysis: This is the analysis performed to understand the dependency between the independent variables (two at a time). It is observed that there are 51 variables which have extremely high collinearity. Please find the list of variables in the sheet embedded below. 



Determining Outliers and Influential Observations:
Outlier:
An outlier is an observation with unusually large positive or negative residual. Positive residuals indicate that the model is underestimating the response value, whereas Negative residuals indicate that the model is overestimating the response. 

outlierTest(<fit>) would help us understanding the outliers for a given distribution. 

Influential Observation:
An Influential observation is a data point which would have disproportionate impact on the outcome of the model. If these observations are removed, the outcome of the model changes significantly. 

Cook’s Distance would help us understand the influential observations. If the distance is greater than 4 / (n-k-1) where n is sample size and k is the number of predictor variables, then such entities would indicate Cook’s Distance.

Pre-processing:
The ones with extremely high positive or negative collinearity have been discarded. The cut off values are set to > 0.9 for positive correlation and < -0.9 for negative correlation.
All the attributed including the target are in numeric format. Changing the target variable to factor. 
The entire data set would be split into two portions: train and test. Train data represents the portion on which the model has to be built. Test represents the portion on which the model has to be tested. Train data considered for this project is 60% of the data and the Test data is 40% of the data.
Identifying the key metrics needed for the problem
The metrics which could be computed and their interpretation in the current business context are as follows:
•	Recall – How many of the actual churners are correctly predicted by the model?
•	Specificity - How many of the actual non-churners are correctly predicted by the model?
•	Precision - How many of the model predicted churners are actually churned?
•	Accuracy – How good is the prediction of the model when compared to the actual (True as True & False as False)

Applying the models
The approach is to predict the outcome using different classification models and finalize the best model based on the relative comparison of Recall and Accuracy. Once the metrics are computed for all the models, the simplest approach with the best outcome could be chosen.
The following models are considered for prediction:
a.	Apply vif function to remove the collinear variables. Do a logistic regression on these variables and predict the outcome.
b.	Apply Principal Component Analysis to remove the collinearity. Apply auto encoder on the original data set to fetch the non-linear combination of attributes. Pick the minimum number of components from PCA which can account for maximum variance and bind them with auto encoder non-linear attributes to perform a random forests. 
c.	Perform a Generalized linear model grid search to predict the outcome

Model 1: Logistic Regression after applying Variance Inflation Factor (vif)
Model 2: Support Vector Machines
Model 3: Random Forests on the combination of PCA and Auto Encoder Components
Model 4: Generalized Linear Model Grid Search

Explanation of the Models:
Model 1: Logistic Regression after applying Variance Inflation Factor (vif)
Variance Inflation Factor would help us understand the collinearity between the variables. The vif function would discard the ones with high collinearity and retain the ones with low value. For this assignment, we have considered the ones with a cut off value of < >
After applying the function, 51 out of 111 variables have been discarded due to their high collinearity.
To cross check the above outcome, the data set has been fed as input to correlation function in excel and the attributed with high correlation are highlighted. The result proved to be the same in both the cases.
Logistic Regression:
Key assumptions of Logistic Regression:
•	The model should be fitted correctly.  Neither over fitting nor under fitting should occur.  That is only the meaningful variables should be included, but also all meaningful variables should be included.  
•	The error terms need to be independent.  Logistic regression requires each observation to be independent.  That is that the data-points should not be from any dependent samples design, e.g., before-after measurements, or matched pairings.  
•	Also the model should have little or no multi-collinearity.  That is that the independent variables should be independent from each other.  However, there is the option to include interaction effects of categorical variables in the analysis and the model.  Hence vif function is applied on the data set prior to regression.
•	Logistic regression assumes linearity of independent variables and log odds.  Whilst it does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.  
•	Lastly, it requires quite large sample sizes.  Because maximum likelihood estimates are less powerful than ordinary least squares (e.g., simple linear regression, multiple linear regression).
How to Interpret the output?
This algorithm helps in the classification of the outcome. The linear equation which is the resultant of the model is known as LOGIT.
Logit is the log of odds ratio. Odds ratio is the ratio of probability of success to probability of failure. 
In this regression, the coefficients give the change in log(odds) in the response for a unit change in the predictor variable, holding other predictor variables constant.
Other key parameters in the output of the model are Deviance, Coefficients, and AIC.
There are two Deviance parameters : Residual Deviance and Null Deviance.
•	Residual Deviance measures the amount of variance which remains unexplained by the model.
•	Null Deviance shows how well the model predicts the response with only the intercept as parameter.

AIC represents Akaike’s Information Criterion and it penalizes for adding more parameters to the model.
Key metrics for this project when Logistic regression is applied are as follows:
# Accuracy on train data is 77.64%
# Recall on train data is 75.64 %
# Accuracy on test data is 78.17%
# Recall on test data is 76.11 %


Model 2: Support Vector Machines
SVM selects the hyper-plane which classifies the classes accurately before even maximizing the margin. SVM has a feature to ignore outliers and find the hyper-plane that has maximum margin. 
Kernels are functions which takes low dimensional input space and transform it to a higher dimensional space i.e. it converts not separable problem to separable problem. 
The default kernel is the radial basis. The following are the other frequently used ones:
•	Linear 
•	Polynomial
•	Gaussian
•	Sigmoid

Dimension of hyperplane is always (n-1) i.e. for 2-dimensional plane hyperplane will be a line. for 3-dimensional space 2-dimensional plane will be a hyperplane and so on
We should specify the Kernel function, cost and the gamma functions while deploying the SVM. We can tune the SVM using the function tune.svm to obtain the optimum combination of gamma and the cost values. 
The kernel function used in this project is the Polynomial.

Model 3: Random Forests on the combination of PCA and Auto Encoder Components
Principal Components Analysis: 
The main goals of principal component analysis is :
•	to identify hidden pattern in a data set
•	to reduce the dimensionality of the data by removing the noise and redundancy in the data
•	to identify correlated variables

This algorithm transforms the original axes into new space where all the attributes are orthogonal to each other. 
Eigenvector with the highest eigenvalue is the principle component of the data set. Eigenvectors are ordered by eigenvalue, highest to lowest. This gives you the components in order of significance. We can choose to ignore the components of lesser significance. We do lose some information, but if the eigenvalues are small, the data loss is insignificant. 
In this project, it is observed that with the help of 10 components the entire information could be captured. These 10 components would be a linear combination of all the attributes in the original space.
Auto Encoder Components:
Auto encoders are self-learning unsupervised algorithms which aim to transform inputs into outputs with the least possible amount of distortion. The auto encoder tries to learn an approximation to the identity function, so as to generate the output that is similar to input. 
For this project, two hidden layers have been considered and 50 features have been extracted in the output layer.

Random Forests:
Random Forest is an ensemble learning based classification and regression technique.
For a building a decision tree, samples of a data frame are selected with replacement along with selecting a subset of variables for each of the decision tree. Both sampling of data frame and selection of subset of the variables are done randomly. 
For this project, the output of PCA (10 components) and the output of AEC (50 features) are merged into a dataset before applying the random forests. 


# Key metrics for this project when Logistic regression is applied are as follows 
# Accuracy on train data is 77.64%
# Recall on train data is 75.64 %
# Accuracy on test data is 78.17%
# Recall on test data is 76.11 %



The top 30 important features extracted from the Random Forests are shown below:
 
Model 4: Generalized Linear Models Grid search
Generalized linear models (GLMs) are an extension of traditional linear models. GLM models are fitted by finding the set of parameters that maximizes the likelihood of the data.
The elastic net parameter α ∈ [0, 1] controls the penalty distribution between the lasso and ridge regression penalties. 
When α = 0, the `1 penalty is not used and a ridge regression solution with shrunken coefficients is obtained. 
If α = 1, the Lasso operator soft-thresholds the parameters by reducing all of them by a constant factor and truncating at zero. This sets a different number of coefficients to zero depending on the λ value.
The Grid search suggests the optimum values of Lambda and Alpha. 
For this project, alpha of the best model was 0.25 and Lambda was 0.001

Key metrics for this project when Logistic regression is applied are as follows:
# Accuracy on train data is 77.64%
# Recall on train data is 75.64 %
# Accuracy on test data is 78.17%
# Recall on test data is 76.11 %


Results
 

As a last step, we plotted the ROC curve (Receiver Operating Characteristics) and calculate the AUC (area under the curve) which are typical performance measurements for a binary classifier.
The ROC is a curve generated by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings while the AUC is the area under the ROC curve. 
As a rule of thumb, a model with good predictive ability should have an AUC greater than 90%.
Inferences
After comparing the performance of all the models on the test data, it is observed that the range of key metrics lies between 70% and 80% in most of the cases. Hence the simplest model has been chosen as the best fit i.e. Logistic Regression after removing the high collinear variables.

 
Road Ahead – Making further inroads
This analysis has been performed on the masked data. If the original attributes and data could be made available, more interesting patterns could be derived. 


