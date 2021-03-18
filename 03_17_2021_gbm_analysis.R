## METHODS ##

# imports library, sets up workspace, etc
set_up <- function() {
  rm(list = ls())
  setwd("~/FHCRC_ML_Project")
  library(dplyr)
  library(caret)
  library(gbm)
  library(pROC)
}

# cleans dataset for modelling
data_clean <- function(df) {
  df <- df %>% 
    mutate(smoke1 = case_when(smoke == 1 ~ 1, smoke == 2 ~ 0, smoke == 3 ~ 0)) %>% 
    mutate(smoke2 = case_when(smoke == 2 ~ 1, smoke == 1 ~ 0, smoke == 3 ~ 0)) %>% 
    mutate(smoke3 = case_when(smoke == 3 ~ 1, smoke == 1 ~ 0, smoke == 2 ~ 0)) %>% 
    select(-X, -clinic, -smoke, -idno, seqid)
  df$smoke1 <- as.factor(df$smoke1)
  df$smoke2 <- as.factor(df$smoke2)
  df$smoke3 <- as.factor(df$smoke3)
  return(df)
}

# generates model grid with info on lr, id, rmse, and roc value
grid_search <- function(t_df, v_df) {
  
  lr_range <- c(0.3,0.1,0.05,0.01,0.005)
  id_range <- c(3, 5, 7)
  
  model_ext_list <- list()
  counter <- 1
  
  for (i in lr_range) {
    for (j in id_range) {
      model_int <- list()
      model_int$m <- gbm(stk_isc~.,
                         data = t_df,
                         distribution = "bernoulli",
                         n.trees = 1000, 
                         shrinkage = i,
                         interaction.depth = j,
                         cv.folds = 10)
      
      predictions <- predict.gbm(model_int$m, v_df)
      
      model_int$rmse <- sqrt(min(model_int$m$cv.error))
      model_int$learning_rate <- i
      model_int$interaction_depth <- j
      model_int$roc <- gbm.roc.area(v_df$stk_isc, predictions)
      
      model_ext_list[[counter]] <- model_int
      
      counter <- counter + 1
    }
  }
  
  return(model_ext_list)
}

# finds and returns best model based on max roc value
find_best_model <- function(model_list) {
  roc_list <- list()
  for (i in 1:15) {
    roc_list[i] <- model_list[[i]]$roc
  }
  roc_index <- which.max(unlist(roc_list))
  return(model_list[[roc_index]]$m)
}

## DATA SET UP ## creating train/validation/test df

set_up()
covars_df <- data_clean(read.csv("fake_covars.csv", stringsAsFactors = FALSE))

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, 
                             p = 0.9, 
                             list = FALSE, 
                             times = 1)

train_val_df <- covars_df[index,]
test_df <- covars_df[-index,]
index2 <- createDataPartition(train_val_df$stk_isc, 
                             p = 0.8, 
                             list = FALSE, 
                             times = 1)

train_df <- train_val_df[index2,]
validation_df <- train_val_df[-index2,]

## TUNING PARAMETERS: LEARNING RATES, INTERACTION-DEPTH ## finding/analyzing best model from tuning parameter grid search

models <- grid_search(train_df, validation_df)
best_model <- find_best_model(models)

summary(best_model)
final_predictions <- predict.gbm(best_model, test_df)
final_roc <- gbm.roc.area(test_df$stk_isc, final_predictions)
