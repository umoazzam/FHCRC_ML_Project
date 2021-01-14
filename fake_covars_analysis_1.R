# Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(caret)
covars_df <- read.csv("fake_covars.csv", 
                      stringsAsFactors = FALSE)

# partitioning data (80% training data, 20% testing data): create index matrix of selected values
# relabel/convert outcome variable stk_isc as factor, 1 = stroke, 0 = no stroke

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, 
                             p = 0.8, 
                             list = FALSE, 
                             times = 1)
covars_df_new <- as.data.frame(covars_df)

covars_df_new$stk_isc[covars_df_new$stk_isc==1] <- "stroke"
covars_df_new$stk_isc[covars_df_new$stk_isc==0] <- "no_stroke"
covars_df_new$stk_isc <- as.factor(covars_df_new$stk_isc)

covars_train_df <- covars_df_new[index,]
covars_test_df <- covars_df_new[-index,]

# specify the type of training method used & number of folds

ctrl_specs <- trainControl(method = "cv", 
                           number = 5, 
                           savePredictions = "all", 
                           classProbs = TRUE)

# set new random seed for k-fold x-val

set.seed(345678)

# specify logistic regression model and check model

model_1 <- train(stk_isc ~ 
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
                 + smoke
                 + clinic,
                 data = covars_train_df,
                 method = "glm", family = binomial, trControl = ctrl_specs)

print(model_1)

## accuracy: ~ 0.64
## kappa: ~0.28

# check output in terms of regression coefficients

summary(model_1)

# comparative variable importance (predictor variables)

varImp(model_1)

# linear modeling attempt

lm.fit <- lm(stk_isc ~ 
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
             + smoke
             + clinic,
             data = covars_train_df)

print(lm.fit)
