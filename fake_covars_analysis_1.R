# Set up workspace/initialize dataset

rm(list = ls())
setwd("~/FHCRC_ML_Project")
library(caret)
covars_df <- read.csv("fake_covars.csv", stringsAsFactors = FALSE)

# partitioning data: create index matrix of selected values

lm.fit <- lm(stk_isc, data = covars_df)

