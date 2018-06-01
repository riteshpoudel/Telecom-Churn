rm(list = ls())

#reading data into R
data <- read.csv("C:/INSOFE/Project/Telecom Churn Data/Data.csv", header = T,sep = ",")
summary(data)
head(data)
str(data)

# Split data to train and test
set.seed(4567)
train_rows = sample(x = nrow(data),size = 0.7*nrow(data))
train = data[train_rows,]
test = data[-train_rows,]

# creating train data set for different models
logit_train = data[train_rows,]
pca_train = data[train_rows,]
rf_train  = data[train_rows,]
aec_train = data[train_rows,]
glm_train = data[train_rows,]


# creating test data set for different models
logit_test = data[-train_rows,]
pca_test = data[-train_rows,]
rf_test  = data[-train_rows,]
aec_test = data[-train_rows,]
glm_test = data[-train_rows,]

########################## Approach 01 ############################
#Apply vif function to remove the collinear variables. 
#Applying logistic regression on these variables and compute the accuracy

library(usdm)

vif_train <- vifstep(data[,-18],th = 10)
summary(vif_train)
temp_data <- exclude(data,vif_train)
summary(temp_data)
dim(temp_data)

target <- data[,18]
temp_data_train <- cbind(target,temp_data)
dim(temp_data_train)
rm(vif_train,temp_data)

# applying logistic regression on data after VIF
mod_appr1 = glm(target ~ ., data = temp_data_train, family = 'binomial')
summary(mod_appr1)

# Predict on train
pred_appr1 = predict(mod_appr1,type = "response")
head(pred_appr1)
conf_Mtrx_appr1_trn = table(temp_data_train$target,ifelse(pred_appr1<0.3,0,1))
conf_Mtrx_appr1_trn

#Error Metrics on train data set
accuracy_train_logit = sum(diag(conf_Mtrx_appr1_trn))/sum(conf_Mtrx_appr1_trn)
precision_train_logit = conf_Mtrx_appr1_trn[2,2]/sum(conf_Mtrx_appr1_trn[,2])
recall_train_logit = conf_Mtrx_appr1_trn[2,2]/sum(conf_Mtrx_appr1_trn[2,])
# Accuracy on train data is 76.68 %
# Recall on train data is 78.6%

# Prediction on test data set
pred_appr1_test = predict(mod_appr1,newdata = test, type = "response")
conf_Mtrx_appr1_test = table(test$target,ifelse(pred_appr1_test<0.3,0,1))

#Error Metrics on test
accuracy_test_logit = sum(diag(conf_Mtrx_appr1_test))/sum(conf_Mtrx_appr1_test)
precision_test_logit = conf_Mtrx_appr1_test[2,2]/sum(conf_Mtrx_appr1_test[,2])
recall_test_logit = conf_Mtrx_appr1_test[2,2]/sum(conf_Mtrx_appr1_test[2,])

# Accuracy on test data is 77.57 %
# Recall on test data is 79.63%
################################# Approach 2 ########################################
#Apply Support Vector machines on the data set to classify the target variable

library(e1071)
svm_train = subset(train, select = -target)
y = as.factor(train$target)
svm_model = svm(svm_train,y, method = "C-classification", kernel = "polynomial", cost = 10, gamma = 0.1)

summary(svm_model)

pred = predict(svm_model, svm_train)
table(pred, y)

svm_test = subset(test, select = -target)
b = as.factor(test$target)
pred= predict(model, svm_test)
table(pred, b)

conf_Mtrx_appr2_trn = table(temp_data_train$target,ifelse(pred<0.5,0,1))

#Error Metrics on train data set
accuracy_train_svm = sum(diag(conf_Mtrx_appr2_trn))/sum(conf_Mtrx_appr2_trn)
recall_train_svm = conf_Mtrx_appr2_trn[2,2]/sum(conf_Mtrx_appr2_trn[2,])
# Accuracy on train data is 76.68 %
# Recall on train data is 78.6%

# Prediction on test data set
conf_Mtrx_appr2_test = table(temp_data_test$target,ifelse(pred<0.5,0,1))

#Error Metrics on test
accuracy_test_svm = sum(diag(conf_Mtrx_appr2_test))/sum(conf_Mtrx_appr2_test)
recall_test_svm = conf_Mtrx_appr2_test[2,2]/sum(conf_Mtrx_appr2_test[2,])

################################# Approach 3 ########################################
#Apply PCA & auto encoder on the data set to fetch the non linear combination of attributes.
#Combining both PCA linear components & Auto encoder non-linear features and perform a random forests.

#Apply PCA
pcad <- princomp(pca_train)
summary(pcad)
pcad$loadings[,1:10]

pca_train_scores <- pcad$scores[,1:10]
dim(pca_train_scores)

pca_data_tst <- predict(pcad, pca_test)
pca_test_scores <- pca_data_tst[,1:10]
head(pca_test_scores)
dim(pca_test_scores)

# Apply autoencoder
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "1g")

data.hex_train <- as.h2o(x = aec_train , destination_frame = "data.hex_train")
summary(data.hex_train)

data.hex_test <- as.h2o(x = aec_test , destination_frame = "data.hex_test")
summary(data.hex_test)

y = "target"
x = setdiff(colnames(data.hex_train), y)

aec <- h2o.deeplearning(x = x, autoencoder = T, 
                        training_frame=data.hex_train,
                        activation = "RectifierWithDropout",
                        hidden = c(90,75),
                        epochs = 50)

# Extract features on train data
features_train <- as.data.frame(h2o.deepfeatures(data = data.hex_train[,x], object = aec, layer = 2))
head(features_train)
dim(features_train)

# Extract features on test data
features_test <- as.data.frame(h2o.deepfeatures(data = data.hex_test[,x], object = aec, layer = 2))
head(features_test)
dim(features_test)

# Apply Random forests on the combined data set

rf_data <- cbind(features_train,pca_train_scores, target = aec_train[,18])
dim(rf_data)
head(rf_data)
rf_data.hex <- as.h2o(x = rf_data , destination_frame = "rf_data.hex")
dim(rf_data.hex)
y
x_rf <- setdiff(names(rf_data.hex),y)
require(randomForest)
rf_DL <- h2o.randomForest(x_rf,y,training_frame = rf_data.hex)
summary(rf_DL)

# importance of attributes
imp <- c("Comp.4", "DF.L2.C75", "DF.L2.C35", "DF.L2.C45", "Comp.7", "Comp.3", "Comp.9", "DF.L2.C38",
         "Comp.8", "DF.L2.C9", "Comp.2", "Comp.5", "Comp.1", "Comp.10", "DF.L2.C37", "Comp.6", "DF.L2.C20",
         "DF.L2.C48", "DF.L2.C22", "DF.L2.C18", "DF.L2.C41", "DF.L2.C29", "DF.L2.C69", "DF.L2.C60",
         "DF.L2.C14", "DF.L2.C43", "DF.L2.C31", "DF.L2.C24", "DF.L2.C25", "DF.L2.C50", "DF.L2.C73",
         "DF.L2.C61", "DF.L2.C40", "DF.L2.C59", "DF.L2.C58", "DF.L2.C5", "DF.L2.C57", "DF.L2.C2",
         "DF.L2.C44", "DF.L2.C65", "DF.L2.C30", "DF.L2.C3", "DF.L2.C64", "DF.L2.C42", "DF.L2.C32",
         "DF.L2.C23", "DF.L2.C34", "DF.L2.C21", "DF.L2.C54", "DF.L2.C6", "DF.L2.C27", "DF.L2.C62","DF.L2.C63")

train_Imp = rf_data[,imp]
dim(train_Imp)
train_Imp_w.target = cbind(train_Imp,target = pca_train$target)
dim(train_Imp_w.target)
train_Imp_w.target.hex <- as.h2o(train_Imp_w.target, destination_frame = "train_Imp_w.target.hex")
rf_model <- h2o.randomForest(imp,y,training_frame = train_Imp_w.target.hex)

test_rf = cbind(features_test, pca_test_scores)
test_Imp = test_rf[,imp]
test_Imp_w.target = cbind(test_Imp, target = aec_test$target)
dim(test_Imp_w.target)
test_Imp_w.target.hex <- as.h2o(test_Imp_w.target, destination_frame = "test_Imp_w.target.hex")
dim(test_Imp_w.target.hex)

# Predict on train
pred_train_rf.hex = h2o.predict(rf_DL,train_Imp_w.target.hex)
pred_train_rf <- as.data.frame(pred_train_rf.hex)
length(train_Imp_w.target$target)
conf_Matrix_rf_train = table(train_Imp_w.target$target,ifelse(pred_train_rf<0.5,0,1))
conf_Matrix_rf_train

#Error Metrics on train
accuracy_train_rf = sum(diag(conf_Matrix_rf_train))/sum(conf_Matrix_rf_train)
precision_train_rf = conf_Matrix_rf_train[2,2]/sum(conf_Matrix_rf_train[,2])
recall_train_rf = conf_Matrix_rf_train[2,2]/sum(conf_Matrix_rf_train[2,])
# Accuracy on train data is 98.48 %
# Recall on train data is 95.38 %

# Predict on test
pred_test_rf.hex = h2o.predict(rf_DL,newdata = test_Imp_w.target.hex, type = "response")
pred_test_rf = as.data.frame(pred_test_rf.hex)
conf_Matrix_rf_test = table(test_Imp_w.target$target,ifelse(pred_test_rf<0.5,0,1))
conf_Matrix_rf_test

#Error Metrics
accuracy_test_Imp = sum(diag(conf_Matrix_rf_test))/sum(conf_Matrix_rf_test)
precision_test_Imp = conf_Matrix_rf_test[2,2]/sum(conf_Matrix_rf_test[,2])
recall_test_Imp = conf_Matrix_rf_test[2,2]/sum(conf_Matrix_rf_test[2,])

# Accuracy on test data is 79.05 %
# Recall on train data is 56.99 %

###############################Approach 4#########################
# Apply glm on the data set with various combinations of alpha and lamba and determine the appropriate generalization to be used (lasso, ridge or elastic net). Compute the accuracy
# Import a local R train data frame to the H2O cloud

train.hex <- as.h2o(x = train, destination_frame = "train.hex")

# Lambda search
model_LS = h2o.glm(y = "target", 
                   x = setdiff(names(train.hex), "target"),
                   training_frame = train.hex, 
                   family = "binomial",
                   lambda_search = TRUE)

print(model_LS)


# Prepare the parameters for the for H2O glm grid search
lambda_opts = list(list(1), list(.5), list(.1), list(.01), 
                   list(.001), list(.0001), list(.00001), list(0))
alpha_opts = list(list(0), list(.25), list(.5), list(.75), list(1))

hyper_parameters = list(lambda = lambda_opts, alpha = alpha_opts)

# Build H2O GLM with grid search
grid_GLM <- h2o.grid("glm", 
                     hyper_params = hyper_parameters, 
                     grid_id = "grid_GLM.hex",
                     y = "target", 
                     x = setdiff(names(train.hex), "target"),
                     training_frame = train.hex, 
                     family = "binomial")

# Remove unused R objects
rm(lambda_opts, alpha_opts, hyper_parameters)

# Get grid summary
summary(grid_GLM)

# Fetch GBM grid models
grid_GLM_models <- lapply(grid_GLM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

for (i in 1:length(grid_GLM_models)) 
{ 
  print(sprintf("regularization: %-50s auc: %f", grid_GLM_models[[i]]@model$model_summary$regularization, h2o.auc(grid_GLM_models[[i]])))
}

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GLM_model = find_Best_Model(grid_GLM_models)

rm(grid_GLM_models)

# Get the auc of the best GBM model
best_GLM_model_AUC = h2o.auc(best_GLM_model)

# Examine the performance of the best model
best_GLM_model

# View the specified parameters of the best model
best_GLM_model@parameters

# Important Variables.
h2o.varimp(best_GLM_model)

# Computing model efficiency on train data
predict_glm_train.hex = h2o.predict(best_GLM_model, 
                                    newdata = train.hex[,setdiff(names(train.hex), "target")])
predict_glm_train <- as.data.frame(predict_glm_train.hex)
conf_Matrix_GLM = table(train$target, predict_glm_train$predict)

accuracy_glm_train = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)
precision_glm_train = conf_Matrix_GLM[2,2]/sum(conf_Matrix_GLM[,2])
recall_glm_train = conf_Matrix_GLM[2,2]/sum(conf_Matrix_GLM[2,])

# Accuracy on train data is 77.64%
# Recall on train data is 75.64 %

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test, destination_frame = "test.hex")


# Predict on same training data set
predict.hex = h2o.predict(best_GLM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "target")])

data_GLM = h2o.cbind(test.hex[,"target"], predict.hex)

# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)

# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$target, pred_GLM$predict) 

accuracy_glm = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)
precision_glm = conf_Matrix_GLM[2,2]/sum(conf_Matrix_GLM[,2])
recall_glm = conf_Matrix_GLM[2,2]/sum(conf_Matrix_GLM[2,])

# Accuracy on test data is 78.17%
# Recall on test data is 76.11 %

# Shutdown H2O
h2o.shutdown(F)

