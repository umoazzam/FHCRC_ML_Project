# Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(caret)
covars_df <- read.csv("fake_covars.csv", stringsAsFactors = FALSE)

# partitioning data (80% training data, 20% testing data): create index matrix of selected values

set.seed(345678)
index <- createDataPartition(covars_df$stk_isc, p = 0.8, list = FALSE, times = 1)
covars_df_new <- as.data.frame(covars_df)
covars_train_df <- covars_df_new[index,]
covars_test_df <- covars_df_new[-index,]


#


lm.fit <- lm(stk_isc, data = covars_df)

