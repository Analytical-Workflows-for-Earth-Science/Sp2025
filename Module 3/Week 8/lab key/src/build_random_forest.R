#################################################################
### Build a random forest model for the presence and abundance ##
### of a species in the NOAA Bering Sea trawl data set.       ###
#################################################################

# load command line arguments 
# Replace the calls to args with specific values if running this file from the console
args = commandArgs(trailingOnly=TRUE)
species <- args[1]
ntree <- as.numeric(args[2])
mtry <- as.numeric(args[3])


# Load libraries
require(dplyr)
require(reshape2)
require(ggplot2)
require(randomForest)

# Load presence data for species
data_presence <- read.csv(paste0("data/", species, "_presence.csv"))

# Final data processing for random forest model 
# remove Haul.ID, Survey year, lat, lon, and X
# filter rows with NANs out of the dataset
data_presence <- data_presence %>% 
  select(-X,-Haul.ID,-Survey.year, # remove variables 
         -Start.latitude..decimal.degrees.,
         -Start.longitude..decimal.degrees.) %>% 
  filter(!is.na(presence), # filter NA values for presence and temperature
         !is.na(Bottom.temperature..degrees.Celsius.),
         !is.na(Surface.temperature..degrees.Celsius.))

# Train model on presence data 
# Train a random forest for the presence data using all columns as covariates
# Build ntree trees and try mtry variables at each split. 
rf <- randomForest::randomForest(as.factor(presence)~., data = data_presence, 
                                 ntree = ntree, mtry = mtry)

# Save rf model and related data to the resutls/rf_models file
write.csv(rf$confusion,paste0("results/rf_models/rf_confusion_",species,"_presence.csv") )
saveRDS(rf, file = paste0("results/rf_models/rf_model_",species,"_presence.rdata"))
saveRDS(data_presence, file = paste0("results/rf_models/rf_data_",species,"_presence.rdata"))

##########################################
####### Repeat for densities data ########
##########################################

# Load data
data_density <- read.csv(paste0("data/",  species, "_density.csv"))

# select variables and remove NANs
data_density <- data_density %>% 
  select(-X,-Haul.ID,-Survey.year,   # remove variables 
         -Start.latitude..decimal.degrees., 
         -Start.longitude..decimal.degrees.)%>%
  filter(!is.na(log_density), # filter NA values for presence and temperature
         !is.na(Bottom.temperature..degrees.Celsius.),
         !is.na(Surface.temperature..degrees.Celsius.))

rf <- randomForest::randomForest(log_density~., data = data_density, ntree = ntree)


# Save random foret model for densities data 
model_performacne <- data.frame(mse = rf$mse[ntree], rsq = rf$rsq[ntree])
write.csv(model_performacne,paste0("results/rf_models/rf_performance_",species,"_density.csv") )
saveRDS(rf, file = paste0("results/rf_models/rf_model_",species,"_density.rdata"))
saveRDS(data_density, file = paste0("results/rf_models/rf_data_",species,"_density.rdata"))


