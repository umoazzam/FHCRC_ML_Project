# Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(caret)
library(dplyr)
library(gbm)
covars_df <- read.csv("fake_covars.csv", 
                      stringsAsFactors = FALSE)

# Set up smoke column as binary for each category of smoker

covars_df <- covars_df %>% 
  mutate(smoke1 = case_when(smoke == 1 ~ 1, smoke == 2 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke2 = case_when(smoke == 2 ~ 1, smoke == 1 ~ 0, smoke == 3 ~ 0)) %>% 
  mutate(smoke3 = case_when(smoke == 3 ~ 1, smoke == 1 ~ 0, smoke == 2 ~ 0)) %>% 
  select(-X, -clinic, -smoke)
  
# Set stroke and smoke variables as factors

covars_df <- as.data.frame(covars_df)
covars_df$stk_isc[covars_df$stk_isc==1] <- "stroke"
covars_df$stk_isc[covars_df$stk_isc==0] <- "no_stroke"
covars_df$stk_isc <- as.factor(covars_df$stk_isc)
# covars_df$smoke1 <- as.factor(covars_df$smoke1)
# covars_df$smoke2 <- as.factor(covars_df$smoke2)
# covars_df$smoke3 <- as.factor(covars_df$smoke3)

# Partition data for training and testing

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, 
                             p = 0.8, 
                             list = FALSE, 
                             times = 1)

train_df <- covars_df[index,]
test_df <- covars_df[-index,]

# Set training method with cross validation, set seed

ctrl_specs <- trainControl(method = "cv", 
                           number = 5, 
                           savePredictions = "all", 
                           classProbs = TRUE)
set.seed(345678)

# Train model using GBM

covars_gbm <- train(stk_isc ~ 
                      chblmod 
                    + chdblmod 
                    + afbl 
                    + agebl 
                    + gend01 
                    + race01 
                    + ptfv1 
                    + bmi 
                    + sbp
                    + htnmed06
                    + dm
                    + chol
                    + smoke1
                    + smoke2
                    + smoke3,
                    data = train_df,
                    method = "gbm",
                    distribution = "bernoulli",
                    trControl = ctrl_specs)

# ran into an issue: "There were missing values in resampled performance measures."



