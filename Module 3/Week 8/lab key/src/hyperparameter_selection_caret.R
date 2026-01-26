######################################################
### Run cross-validation test leaving full years   ###
### of data out fo the presence absence data set.  ###
###                                                ###
### Compute cross-validation metric for a suite of ###
### models with different hyperparameters, by      ###
### leaving individual years out of the data set.  ###
### The best model is then tested on the           ###
### final 2 years of data.                         ###
######################################################


# Read in command line arguments
args = commandArgs(trailingOnly=TRUE)
species <- args[1] 
ntree <-  as.numeric(args[2]) 
k <- as.numeric(args[3]) 
 
# Load libraries
require(dplyr)
require(reshape2)
require(ggplot2)
require(randomForest)
require(caret)
require(doParallel)

# Load data (choose between presence or density)
rf_data <- read.csv(paste0("data/", species, "_presence.csv"))


# Final data processing for random forest model 
# remove Haul.ID, Survey yea,r, lat, lon and X
# filter rows with NANs out of the dataset
data_presence <- rf_data %>% 
  select(-X,-Haul.ID, # remove variables 
         -Start.latitude..decimal.degrees.,
         -Start.longitude..decimal.degrees.) %>% 
  filter(!is.na(presence), # filter NA values for presence and temperature
         !is.na(Bottom.temperature..degrees.Celsius.),
         !is.na(Surface.temperature..degrees.Celsius.))


# Set values of the mtry hyperparameter to test 
tunegrid <- expand.grid(
    mtry = c(1,2,3,4,5,6,7,8,9)
)

# Set up the parameters of the caret tuning process with `trainControl`
# caret training method to cross-validation ("cv") and the number of folds k.
# This function will automatically create randomly selected training and testing sets. 
train_control_rand <- trainControl(
  method = "cv",
  number = k,
  savePredictions = "final"
)


# Use the caret `train` function to train a random forest model on the data
# Comparing the performance of each value of mtry in tunegrid 

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

rf_model_rand <- train(
  as.factor(presence) ~ ., 
  data = data_presence %>% select(-Survey.year,),
  method = "rf",
  trControl = train_control_rand,
  tuneGrid = tunegrid,
  ntree = ntree
)

stopCluster(cl)

# The following block of code will create a list of indices for different training 
# and testing sets by breaking the dataset up into blocks of consecutive years. 
# Repeat the prior analysis using these blocks as the training and validation sets.
k <- 12
index <- list()
indexOut <- list()
t0 <- 1998
for (i in 1:k) {
  test_idx <- which(data_presence$Survey.year %in% c(t0:(t0+2)))
  train_idx <- which(!(data_presence$Survey.year %in% c(t0:(t0+2))))
  t0 <- t0 + 2 
  print(t0)
  index[[paste0("Fold", i)]] <- train_idx
  indexOut[[paste0("Fold", i)]] <- test_idx
}

# Set up the parameters of the caret tuning process with `trainControl`
# use the index and indexOut keywords to select the training and testing sets. 
train_control_years <- trainControl(
  method = "cv",
  number = length(index),
  index = index,
  indexOut = indexOut,
  savePredictions = "final"
)

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

rf_model_years <- train(
  as.factor(presence) ~ ., 
  data = data_presence %>% select(-Survey.year,),
  method = "rf",
  trControl = train_control_years,
  tuneGrid = tunegrid,
  ntree = ntree
)

stopCluster(cl)


# Combine the results of the model tuning procedures into a data frame and plot!
rf_years <- as.data.frame(rf_model_years$results)
rf_years$method <- "Years"
rf_random <- as.data.frame(rf_model_rand$results)
rf_random$method <- "Random"

rf_results <- rbind(rf_random,rf_years) 
ggplot(rf_results, aes(x=mtry,y=Accuracy,color=method))+
  geom_point()+geom_line()+theme_classic()

ggsave("results/parameter_selection/model_performance_plot.png",
       height = 5, width = 7)


