######################################################
### Run cross-validation test leaving full years   ###
### of data out of the presence dataset.           ###
###                                                ###
### Compute cross-validation metric for several    ###
### models with different hyperparameters, by      ###
### leaving individual years out of the data set.  ###
### The best model is then tested on the           ###
### final 2 years of data.                         ###
######################################################

# load command line arguments 
args = commandArgs(trailingOnly=TRUE)
species <- args[1]

# libraries 
require(parallel)
require(randomForest)
require(dplyr)
require(purrr)
require(ggplot2)
require(reshape2)

# load useful functions from performance_metrics.R

source("src/performance_metrics.R")

########################
### define function ###
########################


# This function trains the random forest model on the dataset 
# training while setting the *choose a hyper-parameter* to h_param
# use help(randomForest) to see how the model's hyperparameters can be modified
train_presence_rf <- function(training, mtry){
  
  # Final dat processing fro rf model 
  training <- training %>% 
    select(-X,-Haul.ID,-Survey.year, # remove variables not included in RF model  
           -Start.latitude..decimal.degrees.,
           -Start.longitude..decimal.degrees.) %>% 
    filter(!is.na(presence), # filter NA values for presence and temperature
           !is.na(Bottom.temperature..degrees.Celsius.),
           !is.na(Surface.temperature..degrees.Celsius.))
  
  # train model with the mtry variables tested at each split 
  rf <- randomForest::randomForest(as.factor(presence)~., data = training, 
                                   ntree = 100, mtry = mtry)
  
  return(rf)
}


# Train the random forest model, leaving year i out of the training dataset
# set the hyperparameter value to h_param
test_model_i <- function(i,mtry,training){
  sub_training_data <- training %>% filter(Survey.year != i) # leave year i out ot the training data set
  sub_testing_data <- training %>% filter(Survey.year == i) # only include year i in the data set
  mod <- train_presence_rf(sub_training_data, mtry)
  c_mat <- confusion(mod, sub_testing_data)
  return(c_mat)
} 


# set up method to calculate teh out of sample performance of the model
# given the value of the hyper parameters h_param
test_hyper_param <- function(mtry,training,cl){
  
  # run model tests set 1:k using parLapply
  input_list <- as.list(unique(training$Survey.year))
  test_model_func <- function(i){test_model_i(i,mtry,training)}
  output_list <- parLapply(cl, input_list, test_model_func)
  # use parLapply to run the test_model_i on each year in the training data set
  # type help(parLapply) to learn about the parLapply
  # hint: you will need to create a list of years, including the training data set, the unique() function can help
  # You will also need to create a function to pass to parLapply that runs test_model_i
  # on the training data set and h_param, but only requires the year i and an argument.
    
  
  c_mat <- reduce(output_list,element_sum)
  
  # calculate model performance metrics using 
  # functions from performance_metrics.R
  metrics <- data.frame(hyper_param = mtry,
                        sensetivity = sensetivity(c_mat), 
                        specificity = specificity(c_mat),
                        error_rate = error_rate(c_mat))
  return(metrics)
}


######################
### run analysis  ###
#####################

# load data set 

# Final data processing for random forest model 
# remove Haul.ID, Survey year, lat, lon, and X
# filter rows with NANs out of the dataset
data <- read.csv(paste0("data/", species, "_presence.csv"))
  
# split data into training and final testing set
training <- data[data$Survey.year<2017,]
testing <- data[data$Survey.year>=2017,]

param_levels <- c(1,2,3,4,5,6,7) # set hyperparameter levels using a vector c(...)

### Set up virtual parallel processing to run cross validation tests in parallel
nmax <- detectCores() # detect number of cores on your machine use require(help = "parallel") to find the function for this
nuse <- nmax-2 # use all but 2 cores for the analysis
cl <- makeCluster(nuse) # make the virtual parallel computing cluster. again see require(help = "parallel") for helpful functions

# export the functions, data and variables used in the analysis to cluster using parallel::clusterExport
# Loop over the values of each hyperparameter in series 
# and save the performance metric to a data frame named df_metrics
mtry <- param_levels[1]
exp_data <- c("param_levels","train_presence_rf","test_model_i","confusion", "training","filter","select", "%>%", "mtry")# list names of functions and variables needed in the analysis as strings.
clusterExport(cl, exp_data, envir=environment())
df_metrics <- test_hyper_param(param_levels[1],training,cl)
for(i in 2:length(param_levels )){
  mtry <- param_levels[i]
  exp_data <- c("param_levels","train_presence_rf","test_model_i","confusion", "training","filter","select", "%>%", "mtry")# list names of functions and variables needed in the analysis as strings.
  clusterExport(cl, exp_data, envir=environment())
  df_metrics_i <- test_hyper_param(mtry ,training,cl)
  df_metrics <- rbind(df_metrics,df_metrics_i)
}

# close the cluster when the task completes
stopCluster(cl) 

# Plot cross-validation metrics as a function of the hyperparameter value
# and save to the `results/parameter_selection` file.
ggplot(df_metrics %>% melt(id.var="hyper_param"),
       aes(x = hyper_param, y = value, color = variable))+
  geom_point()+geom_line()+
  theme_classic()+
  scale_color_manual(values = PNWColors::pnw_palette("Bay", n=3))

ggsave(paste0("results/parameter_selection/mtry_",species,"_presence.png"), 
       height = 4, width = 5.5)

# Save cross-validation metrics 
write.csv(df_metrics,paste0("results/parameter_selection/mtry_",species,"_presence.csv"))


##########################################################
## select best model based on cross validation results ###
## and test its performance on the testing data set    ###
##########################################################

# Select the best value of the hyperparameter
ind <- df_metrics$error_rate == max(df_metrics$error_rate)
best_mtry <- df_metrics$hyper_param[ind] 


# test the selected model on the training data set
rf <- df_metrics$error_rate == max(df_metrics$error_rate) # build rf model on the complete data set. You can use the train_presence_rf  
c_mat <-  df_metrics$hyper_param[ind] # Calculate confusion matrix on testing data set using the confusion function from performance_metrics.R

# calculate final suite of performance metrics using function from performance_metrics.R
# and combine into a data frame 
mod <- train_presence_rf(training, best_mtry)
c_mat <- confusion(mod, testing)
final_metrics <- data.frame(hyperparameter = best_mtry,
                            sensetivity = sensetivity(c_mat), 
                            specificity = specificity(c_mat),
                            error_rate = error_rate(c_mat))
# save results to results/parameter_selection
write.csv(final_metrics,
          paste0("results/parameter_selection/best_mtry_",species,"_presence.csv"))





