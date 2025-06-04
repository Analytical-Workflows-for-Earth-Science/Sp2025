#####################################
### Make partial dependence plots ###
#####################################

# load in command line arguments
args = commandArgs(trailingOnly=TRUE)
species <- args[1]
variable <- args[2]

# load pdp package for partial dependence plots, and ggplot + ggplotify for saving
require(pdp)
require(randomForest)
require(ggplot2)
require(ggplotify) 
require(dplyr)

# Read in saved random forest model using paste0 to include the species name
rf <- readRDS(paste0("results/rf_models/rf_model_", species,"_presence.rdata"))

# Read in training data, this is required for the pdp::partial function.
training <- readRDS(paste0("results/rf_models/rf_data_", species,"_presence.rdata"))

# Make a partial dependence plot for the variable specified by the command line arguments
# using pdp::partial
# type help(partial) into the console for documentation
# Note that some additional keyword arguments will need to be passed to 
# the partial function for categorical models
plt <- partial(rf, variable, train = training, 
               prob = TRUE, which.class = "TRUE", plot = TRUE)

# Convert to ggplot and add white background
p <- as.ggplot(plt) + theme(panel.background = element_rect("white")) 

# save!
ggsave(paste0("results/figures/pdp_", species,"_", variable,"_presence.png"), 
       plot = p, height = 4, width = 5)


########################################
#### repeat for the densities model ####
########################################
# Read in saved random forest model 
rf <- readRDS(paste0("results/rf_models/rf_model_", species,"_density.rdata"))

# Read in training data this is required for the pdp::partial function
training <- readRDS(paste0("results/rf_models/rf_data_", species,"_density.rdata"))

# Make partial dependence plot using pdp::partial
# some of the key word arguments are only needed for categorization 
# models and will not apply to regression models 
plt <- partial(rf, variable, train = training, plot = TRUE)

# Convert to ggplot and add white background
p <- as.ggplot(plt) + theme(panel.background = element_rect("white")) 

# save!
ggsave(paste0("results/figures/pdp_", species,"_", variable,"_density.png"), 
       plot = p, height = 4, width = 5)


