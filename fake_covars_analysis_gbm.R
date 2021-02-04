## Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(dplyr)
library(caret)
library(gbm)
covars_df <- read.csv("fake_covars.csv", 
                      stringsAsFactors = FALSE)

## Set up smoke column as binary for each category of smoker

covars_df <- covars_df %>% 
  mutate(smoke1 = case_when(smoke == 1 ~ 1, smoke == 2 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke2 = case_when(smoke == 2 ~ 1, smoke == 1 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke3 = case_when(smoke == 3 ~ 1, smoke == 1 ~ 0, smoke == 2 ~ 0)) %>% 
  select(-X, -clinic, -smoke)

## Set stroke and smoke variables as factors (DONT NEED THIS)

### covars_df <- as.data.frame(covars_df)
### covars_df$stk_isc[covars_df$stk_isc==1] <- "stroke"
### covars_df$stk_isc[covars_df$stk_isc==0] <- "no_stroke"
### covars_df$stk_isc <- as.factor(covars_df$stk_isc)
covars_df$smoke1 <- as.factor(covars_df$smoke1)
covars_df$smoke2 <- as.factor(covars_df$smoke2)
covars_df$smoke3 <- as.factor(covars_df$smoke3)

## Partition data for training and testing

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, 
                             p = 0.8, 
                             list = FALSE, 
                             times = 1)

train_df <- covars_df[index,]
test_df <- covars_df[-index,]

## Run basic gbm model

boost.covars <- gbm(stk_isc~.,
                    data = train_df,
                    distribution = "bernoulli",
                    n.trees = 5000,
                    shrinkage = 0.1,
                    interaction.depth = 4,
                    cv.folds = 10)

## Find index for number of trees with minimum CV error

best <- which.min(boost.covars$cv.error)

## Get MSE and compute RMSE

sqrt(boost.covars$cv.error[best])

## Plot error curve

best.iter <- gbm.perf(boost.covars, method = "cv")
best.iter

## Create Grid Search

summary(boost.covars)

