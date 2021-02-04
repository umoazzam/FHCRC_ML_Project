# GENERAL SET UP

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

# BASIC GBM SET UP

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

## Get MSE and compute RMSE

rmse_val <- sqrt(boost.covars$cv.error[best])

## Plot error curve

best.iter <- gbm.perf(boost.covars, method = "cv")
best.iter

# TUNING PARAMETERS: LEARNING RATES

## Create Grid Search

hyper_grid <- expand.grid(learning_rate = c(0.3,0.1,0.05,0.01,0.005),
                          RMSE = NA,
                          trees = NA,
                          time = NA)

## Execute Grid Search

for(i in seq_len(nrow(hyper_grid))) {
  set.seed(345678) 
  train_time <- system.time({
    m <- gbm(stk_isc~.,
             data = train_df,
             distribution = "bernoulli",
             n.trees = 18, 
             shrinkage = hyper_grid$learning_rate[i],
             interaction.depth = 3,
             cv.folds = 10)
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

## Grid search results

arrange(hyper_grid, RMSE)

# TUNING PARAMETERS: INTERACTION.DEPTH AND N.MINOBSINNODE

## Search grid

hyper_grid <- expand.grid(n.trees = 18,
                          shrinkage = 0.01,
                          interaction.depth = c(3, 5, 7),
                          n.minobsinnode = c(5, 10, 15))

## Create model fit function

model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(345678)
  m <- gbm(stk_isc~.,
           data = train_df,
           distribution = "bernoulli",
           n.trees = n.trees,
           shrinkage = shrinkage,
           interaction.depth = interaction.depth,
           n.minobsinnode = n.minobsinnode,
           cv.folds = 10)
  sqrt(min(m$cv.error))
}

## Perform search grid with functional programming

hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(n.trees = ..1,
              shrinkage = ..2,
              interaction.depth = ..3,
              n.minobsinnode = ..4
  )
)

arrange(hyper_grid, rmse)

# FINAL 

## Run basic gbm model

boost.covars <- gbm(stk_isc~.,
                    data = train_df,
                    distribution = "bernoulli",
                    n.trees = 18,
                    shrinkage = 0.1,
                    interaction.depth = 5,
                    n.minobsinnode = 5,
                    cv.folds = 10)

summary(boost.covars)


















