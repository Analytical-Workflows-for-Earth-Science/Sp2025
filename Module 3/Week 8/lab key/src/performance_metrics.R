##################################################
### This file contains useful methods for      ###
### calculating model performance on out of    ###
### sample data                                ###
##################################################

# run predicitons with the trained model 
# and calcualte the confunsion matrix 
confusion <- function(model,testing){
  preds <- predict(model,testing)
  obs <- testing$presence
  tab <- table(factor(obs, levels = c("TRUE","FALSE")), factor(preds, levels = c("TRUE","FALSE")))
  return(tab)
}

element_sum <- function(x,y){x+y}
# performance metrics (True positive ratio and estimate of positive correctness)
sensetivity<-function(c){c[1,1]/sum(c[1,])}
specificity<-function(c){c[2,2]/sum(c[2,])}
error_rate<-function(c){(c[1,1]+c[2,2])/sum(c)}

