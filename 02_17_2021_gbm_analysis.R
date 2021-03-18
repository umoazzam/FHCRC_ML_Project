# GENERAL SET UP

## Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(dplyr)
library(caret)
library(gbm)
library(pROC)
covars_df <- read.csv("fake_covars.csv", 
                      stringsAsFactors = FALSE)

## Set up smoke column as binary for each category of smoker

covars_df <- covars_df %>% 
  mutate(smoke1 = case_when(smoke == 1 ~ 1, smoke == 2 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke2 = case_when(smoke == 2 ~ 1, smoke == 1 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke3 = case_when(smoke == 3 ~ 1, smoke == 1 ~ 0, smoke == 2 ~ 0)) %>% 
  select(-X, -clinic, -smoke, -idno, seqid)

## Set stroke and smoke variables as factors (DONT NEED THIS)

covars_df$smoke1 <- as.factor(covars_df$smoke1)
covars_df$smoke2 <- as.factor(covars_df$smoke2)
covars_df$smoke3 <- as.factor(covars_df$smoke3)

## Partition data for training and testing

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, 
                             p = 0.9, 
                             list = FALSE, 
                             times = 1)

train_val_df <- covars_df[index,]
test_df <- covars_df[-index,]

# BASIC GBM SET UP

## Split training data into training and validation

index2 <- createDataPartition(train_val_df$stk_isc, 
                             p = 0.8, 
                             list = FALSE, 
                             times = 1)

train_df <- train_val_df[index2,]
validation_df <- train_val_df[-index2,]

## Run basic gbm model

boost.covars <- gbm(stk_isc~.,
                    data = train_df,
                    distribution = "bernoulli",
                    n.trees = 5000,
                    shrinkage = 0.1,
                    interaction.depth = 4,
                    cv.folds = 10)

summary(boost.covars)

## Find index for number of trees with minimum CV error

best <- which.min(boost.covars$cv.error)

## Get MSE and compute RMSE (standard way to measure error in model predicting quantitative data)

rmse_val <- sqrt(boost.covars$cv.error[best])

## Compute best number of trees and find ROC

best.iter <- gbm.perf(boost.covars, method = "cv")
best.iter

pred_example <- predict.gbm(boost.covars, validation_df)
roc_example <- gbm.roc.area(validation_df$stk_isc, pred_example)

# TUNING PARAMETERS: LEARNING RATES, INTERACTION>DEPTH

## Create Grid Search

lr_range <- c(0.3,0.1,0.05,0.01,0.005)
id_range <- c(3, 5, 7)

model_ext_list <- list()
counter <- 1
roc_list <- list()

for (i in lr_range) {
  for (j in id_range) {
    model_int <- list()
    model_int$m <- gbm(stk_isc~.,
                       data = train_df,
                       distribution = "bernoulli",
                       n.trees = 1000, 
                       shrinkage = i,
                       interaction.depth = j,
                       cv.folds = 10)
    
    predictions <- predict.gbm(model_int$m, validation_df)
    # roc <- roc(validation_df$stk_isc, predictions)
    roc <- gbm.roc.area(validation_df$stk_isc, predictions)
    
    model_int$rmse <- sqrt(min(model_int$m$cv.error))
    model_int$learning_rate <- i
    model_int$interaction_depth <- j
    model_int$roc <- roc
    
    roc_list[counter] <- roc
    
    model_ext_list[[counter]] <- model_int
    
    counter <- counter + 1
  }
}

## Find best model

max(unlist(roc_list))
best_index <- which.max(unlist(roc_list))
best_model <- model_ext_list[[best_index]]$m

# TESTING BEST MODEL

summary(best_model)
predict_best <- predict.gbm(best_model, test_df)
roc_best <- gbm.roc.area(test_df$stk_isc, predict_best)
roc_best
