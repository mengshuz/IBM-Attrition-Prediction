#########################################
library(readxl)
library(fastDummies)
library(tree)
library(randomForest)
library(pROC)
library(paletteer)
source("DataAnalyticsFunctions.R")
source("PerformanceCurves.R")
library(class)
library(VIM)
library(ggplot2)
library(GGally)
library(readxl)
library(glmnet)
library(Matrix)
library(randomForest)


hr_original <- read.csv("HR Employee Attrition.csv")

####Data Preparation      
### Graph for missing values
aggr(hr_original, cex.axis=0.5)
aggr(hr_original,combined = TRUE, numbers=TRUE,cex.axis=.40, cex.numbers=0.9)
# we can see our dataset have no missing values


#remove duplicated rows
hr_original <- hr_original[!duplicated(hr_original), ]
#remove duplicated columns
hr_original <- hr_original[!duplicated(as.list(hr_original))]

# remove columns with non-distinct values
hr_original <-hr_original[vapply(hr_original, function(x) length(unique(x)) > 1, logical(1L))]

#create dummy variables for categorical variable 
hr_original <- dummy_cols(hr_original, c('BusinessTravel',"Department", "EducationField", "Gender", 'JobRole', 'MaritalStatus', "OverTime", "Education","EnvironmentSatisfaction","JobInvolvement","JobSatisfaction","PerformanceRating","RelationshipSatisfaction","WorkLifeBalance"),remove_selected_columns = TRUE, remove_first_dummy = TRUE)
hr_original

#scale variable MonthlyIncome to thousands
hr_original$MonthlyIncome <- hr_original$MonthlyIncome/1000

# Convert Attrition to (1,0) 
col2 <- ifelse(hr_original$Attrition=="Yes",1,0)
hr_original$Attrition <-col2

#modify variable name with space or special characters
names(hr_original)[names(hr_original) == 'Department_Research & Development'] <- 'Department_Research_and_Development'
names(hr_original)[names(hr_original) == "EducationField_Life Sciences" ] <- "EducationField_Life_Sciences" 
names(hr_original)[names(hr_original) == "EducationField_Technical Degree"] <- "EducationField_Technical_Degree"
names(hr_original)[names(hr_original) == "JobRole_Human Resources"] <- "JobRole_Human_Resources"
names(hr_original)[names(hr_original) == "JobRole_Laboratory Technician" ] <- "JobRole_Laboratory_Technician" 
names(hr_original)[names(hr_original) == "JobRole_Manufacturing Director"  ] <- "JobRole_Manufacturing_Director"  
names(hr_original)[names(hr_original) == "JobRole_Research Director" ] <- "JobRole_Research_Director" 
names(hr_original)[names(hr_original) == "JobRole_Research Scientist" ] <- "JobRole_Research_Scientist"
names(hr_original)[names(hr_original) == "JobRole_Sales Executive" ] <- "JobRole_Sales_Executive"
names(hr_original)[names(hr_original) == "JobRole_Sales Representative" ] <- "JobRole_Sales_Representative"       


##Check datatype of variables
str(hr_original)

## there are 1470 complete observations
## we will work with 1000 observations as our train data-set

set.seed(8)
sample<-sample(c(TRUE,FALSE), nrow(hr_original), replace = TRUE, prob = c(0.7, 0.3))
hr_train<-hr_original[sample,]
hr_test<-hr_original[!sample,]

###Just checking how balanced they are:
mean(hr_train$Attrition==1)
mean(hr_test$Attrition==1)
### very similar attrition rates




###EDA 
### Lets compute the (Full) PCA
pca.hr_train <- prcomp(hr_train[c(1,3,4,5,6,7,8,9,10,11,12,13)],scale=TRUE)
hr_train[c(1,3,4,5,6,7,8,9,10,11,12,13)]
summary(pca.hr_train)


### Lets plot the variance that each component explains

plot(pca.hr_train,main="PCA: Variance Explained by Factors",col=paletteer_c("grDevices::Reds 2", 12))
mtext(side=1, "Factors",  line=1, font=2)

###
###
### It seems like the first 6 components are responsible for almost all variation (0.806)
###
### We can see the PCscore of each county for each Principal Component
### we can see which county has a lot of PC1 (we will interpret PC1 in a little bit via the loadings_hr)
hr_train_pc <- predict(pca.hr_train) # scale(county)%*%pca.county$rotation
### This will give you the PC score of each county on each Principal Component
hr_train_pc

## Next we will interpret the meaning of the latent features (PCs)
## to do that we look at the "loadings_hr" which gives me
## the correlation of each factor with each original feature
## Note that it is important to look at the larger correlations (positively or negatively) 
## to see which are the original features that are more important 
## in a factor. We will do it for the first 6 factors.

### when we look at them via the PCs (i.e. latent features)
loadings_hr <- pca.hr_train$rotation[,1:6]
## pca.hr_train$rotation is a matrix. Each column corresponds
## to a PC and each row to an original feature.
## the value of the entry is the correlation between them (i.e. loading)
##
### For each factor lets display the top features that 
### are responsible for 3/4 of the squared norm of the loadings_hr

####
#### Loading 1
v<-loadings_hr[order(abs(loadings_hr[,1]), decreasing=TRUE)[1:27],1]
loadingfit <- lapply(1:27, function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]
#### Looking at which are large positive and large negative
#### YearsAtCompany    TotalWorkingYears   YearsInCurrentRole

#### Loading 2
v<-loadings_hr[order(abs(loadings_hr[,2]), decreasing=TRUE)[1:27],2]
loadingfit <- lapply(1:27, function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]
#### NumCompaniesWorked   Age    TotalWorkingYears    YearsWithCurrManager 
####




###Modeling
###Why not linear regression?

result_linear <- glm(Attrition==1~., data=hr_train)
summary(result_linear)
predict_linear <- predict(result_linear,newdata=hr_test)
predict_linear

#negative possibilities, linear regression does not work





### Need to estimate probability of Attrition
### Compare different models 
### m.lr : logistic regression
### m.lr.l : logistic regression with interaction using lasso
### m.lr.pl : logistic regression with interaction using post lasso
### m.lr.tree : classification tree
### m.RandomForest : random forest

### Because we will be concerned with providing a compensation

#### set up the data for lasso
#### the features need to be a matrix ([,-1] removes the first column which is the intercept)

Mx_hr<- model.matrix(Attrition ~ .^2, data=hr_train)[,-1]
Mx_hr
My_hr<- hr_train$Attrition == 1
lasso_hr <- glmnet(Mx_hr,My_hr, family="binomial")
lasso_hr
lassoCV_hr <- cv.glmnet(Mx_hr,My_hr, family="binomial")

num.features_hr <- ncol(Mx_hr)
num.n_hr <- nrow(Mx_hr)
num.Attrition_hr <- sum(My_hr)
w_hr <- (num.Attrition_hr/num.n_hr)*(1-(num.Attrition_hr/num.n_hr))
lambda.theory_hr <- sqrt(w_hr*log(num.features_hr/0.05)/num.n_hr)
lassoTheory_hr <- glmnet(Mx_hr,My_hr, family="binomial",lambda = lambda.theory_hr)
summary(lassoTheory_hr)
support(lassoTheory_hr$beta)

features.min_hr <- support(lasso_hr$beta[,which.min(lassoCV_hr$cvm)])
features.min_hr <- support(lassoTheory_hr$beta)
features.min_hr
length(features.min_hr)
data.min_hr <- data.frame(Mx_hr[,features.min_hr],My_hr)

###

###
### prediction is a probability score
### we convert to 1 or 0 via prediction > threshold
PerformanceMeasure <- function(actual, prediction, threshold=.5) {
  1-mean( abs( (prediction>threshold) - actual ) )  
}


###K-fold cross validation

n_hr_train <- nrow(hr_train)

nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n_hr_train/nfold))[sample(1:n_hr_train)]

### create an empty dataframe of results (OOS_new)
OOS_new <- data.frame(m.lr=rep(NA,nfold), m.lr.l=rep(NA,nfold), m.lr.pl=rep(NA,nfold), m.tree=rep(NA,nfold),m.RandomForest=rep(NA,nfold),m.average=rep(NA,nfold)) 


for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'

  ### Logistic regression
  m.lr <-glm(Attrition~., data=hr_train, subset=train,family="binomial")
  pred.lr <- predict(m.lr, newdata=hr_train[-train,], type="response")
  OOS_new$m.lr[k] <- PerformanceMeasure(actual=My_hr[-train], pred=pred.lr)
  
  ### the Post Lasso Estimates
  m.lr.pl <- glm(My_hr~., data=data.min_hr, subset=train, family="binomial")
  pred.lr.pl <- predict(m.lr.pl, newdata=data.min_hr[-train,], type="response")
  OOS_new$m.lr.pl[k] <- PerformanceMeasure(actual=My_hr[-train], prediction=pred.lr.pl)
  
  ### the Lasso estimates  
  m.lr.l  <- glmnet(Mx_hr[train,],My_hr[train], family="binomial",lambda = lassoCV_hr$lambda.min_hr)
  pred.lr.l <- predict(m.lr.l, newx=Mx_hr[-train,], type="response")
  OOS_new$m.lr.l[k] <- PerformanceMeasure(actual=My_hr[-train], prediction=pred.lr.l)
  
  ### the classification tree
  m.tree <- tree(factor(Attrition)~ ., data=hr_train, subset=train) 
  pred.tree <- predict(m.tree, newdata=hr_train[-train,], type="vector")
  pred.tree <- pred.tree[,2]
  OOS_new$m.tree[k] <- PerformanceMeasure(actual=My_hr[-train], prediction=pred.tree)
  
  ### the randomforest 
  m.RandomForest <-randomForest(Attrition~ ., data=hr_train, subset=train) 
  pred.RandomForest <- predict(m.RandomForest, newdata=hr_train[-train,], type="response")
  OOS_new$m.RandomForest[k] <- PerformanceMeasure(hr_train$Attrition[-train], prediction=pred.RandomForest)
  
  ##average
  pred.m.average <- rowMeans(cbind(pred.tree, pred.lr.l, pred.lr.pl, pred.lr,pred.RandomForest))
  OOS_new$m.average[k] <- PerformanceMeasure(actual=My_hr[-train], prediction=pred.m.average)
  
  print(paste("Iteration",k,"of",nfold,"completed"))
  
  
}    

library(VIM)
OOS_new
colMeans(OOS_new)
par(mar=c(5,5,.5,3)+0.3)

barplot(colMeans(OOS_new), las=2,xpd=FALSE , xlab="", ylim=c(0.975*min(colMeans(OOS_new)),max(colMeans(OOS_new))), ylab = "Out of sample Accuracy", main="Performance Measures for Models")

###logistic regression is the best 

train <- which(foldid!=1)

### Logistic regression
m.lr <-glm(Attrition~., data=hr_train, subset=train,family="binomial")
pred.lr <- predict(m.lr, newdata=hr_train[-train,], type="response")
par(mar=c(5.1, 4.1, 4.1, 2.1), mgp=c(3, 1, 0), las=0)

hist(pred.lr, breaks = 40,main="Prediction for Logistic Regression")


### We use the logistic model to predict test sample
### 
m.lr_train <-glm(Attrition~., data=hr_train, family="binomial")
m.lr_train
summary(m.lr_train)

pred.lr_hold_out <- predict(m.lr_train, newdata=hr_test, type="response")
pred.lr_hold_out

##round predicted probability to 2-decimal places
pred.lr_hold_out <-round(pred.lr_hold_out, digits=2)



### We can make predictions using the rule
### if hat prob >= threshold, we set hat Y= 1
### otherwise we set hat Y= 0
Mx_hr_holdout<- model.matrix(Attrition ~ .^2, data=hr_test)[,-1]
Mx_hr_holdout
My_hr_holdout <- hr_test$Attrition == 1



### threshold = .75  
PL.performance75_hr <- FPR_TPR(pred.lr_hold_out>=0.75 , My_hr_holdout)
PL.performance75_hr
### threshold = .25
PL.performance25_hr <- FPR_TPR(pred.lr_hold_out>=0.25 , My_hr_holdout)
PL.performance25_hr

### threshold = .5
PL.performance_hr <- FPR_TPR(pred.lr_hold_out>=0.5 , My_hr_holdout)
PL.performance_hr
### threshold = .5  optimal 


### confusion matrix
confusion.matrix_hr <- c( sum( (pred.lr_hold_out>=0.5) * My_hr_holdout ),  sum( (pred.lr_hold_out>=0.5) * !My_hr_holdout) , sum( (pred.lr_hold_out<0.5) * My_hr_holdout ),  sum( (pred.lr_hold_out<0.5) * !My_hr_holdout))
confusion.matrix_hr


# ROC curve
par(mfrow = c(2, 2))
roccurve1 <-  roc(p=pred.lr_hold_out, y=My_hr_holdout, bty="n")
roccurve2 <-  roc(p=pred.lr.pl, y=My_hr_holdout, bty="n")
roccurve3 <-  roc(p=pred.tree, y=My_hr_holdout, bty="n")
roccurve4 <-  roc(p=pred.m.average, y=My_hr_holdout, bty="n")



par(mar=c(4,5,1,1))


####hold out attrition rate
holdout_attrition_rate <- sum(My_hr_holdout)/sum(!My_hr_holdout)




#Cumulative response curve
cumulative <- cumulativecurve(p=pred.lr_hold_out,y=My_hr_holdout)

#lift curve
lift <- liftcurve(p=pred.lr_hold_out,y=My_hr_holdout)
##mean attrition rate is about 19%, so we get about 3.5 times than random guessing








