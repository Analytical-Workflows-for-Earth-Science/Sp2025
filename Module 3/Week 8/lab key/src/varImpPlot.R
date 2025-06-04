#################################################################
### Create variable importance plots from a random forest     ###
### model.                                                    ###
#################################################################

# Read in command line arguments
args = commandArgs(trailingOnly=TRUE)
species <- args[1]
data_type <- args[2]

# libraries
require(dplyr)
require(ggplot2)
require(randomForest)


# Read in saved random forest model from the results/rf_models directory 
# use the paste0 function to include the species and data_type arguments
# in the file name
rf <- readRDS(paste0("results/rf_models/rf_model_", species,"_",data_type,".rdata"))


# Plot the variable importance using ggplot 
# the randomForest object saves the variable importance metrics
# as an array where each row is associated with a different variable.
# You can access the name of each variable in the array using the rownames function.
varImp <- as.data.frame(rf$importance)
varImp$variable <- rownames(varImp)
names(varImp) <- c("Importance", "Variable")
ggplot(varImp, aes(x = Importance, y = Variable))+
  geom_segment(aes(xend = 0))+
  geom_point()+theme_classic()

# Save the plot to the results/figures directory using the 
# ggsave function. You can use the paste0 function to 
# include the data_type and species in the file name. 
ggsave(paste0("results/figures/rf_varImp_",species,"_",data_type,".png"),
       height = 6, width = 6)





