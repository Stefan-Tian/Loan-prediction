library(data.table)
library(caret)
library(caTools)
library(ranger)
setwd("~/Downloads")
loan_train<-fread("train_u6lujuX_CVtuZ9i.csv", stringsAsFactors = T)
loan_test<-fread("test_Y3wMUE5_7gLdaTN.csv")

#use knn to impute missing values
pre_value<-preProcess(loan_train, method=c("knnImpute", "center", "scale"))
library(RANN)
loan_train<-predict(pre_value, newdata=loan_train)
str(loan_train)

#converting categorical variables to numeric ones, and we don't need the ID
loan_train$Loan_Status<-ifelse(loan_train$Loan_Status=='Y', 1, 0)
loan_train<-loan_train[,-1]
hot_encode<-dummyVars(" ~.", data=loan_train, fullRank=T)
loan_train<-data.frame(predict(hot_encode, newdata=loan_train))
str(loan_train)

#split train/test
set.seed(113)
spl<-sample.split(loan_train$Loan_Status, SplitRatio = .7)
train<-subset(loan_train, spl==T)
test<-subset(loan_train, spl==F)

#logistic regression with all feature
logmodel<-glm(Loan_Status~., data=train, family="binomial")
summary(logmodel)
predlog<-predict(logmodel, newdata=test, type="response")
confusionMatrix(as.numeric(predlog>0.5), test$Loan_Status, positive="1")

#randomforest with all features
rfmodel<-ranger(Loan_Status~., data=train, importance="permutation")
predrf<-predict(rfmodel, data=test)
confusionMatrix(as.numeric(predictions(predrf)>0.5), test$Loan_Status, positive="1")

#2 ways of finding the important features
sort(-ranger::importance(rfmodel))
control<-rfeControl(functions=rfFuncs, 
                    method="repeatedcv", 
                    repeats=5, verbose=F)
dep<-"Loan_Status"
indep<-names(train)[!names(train)%in%dep]
important_features<-rfe(train[, indep], train[, dep], rfeControl=control)
important_features

#now we take the top 5 important variables
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term",
              "ApplicantIncome", "CoapplicantIncome")

#train with important features using h2o
library(h2o)
localH2O<-h2o.init(nthreads = -1)#using all CPUs
train.h2o<-as.h2o(train)
test.h2o<-as.h2o(test)
colnames(train.h2o)
dep.h2o<-19
indep.h2o<-c(12:16)

#logistic regression again with cv
logmodel.h2o<-h2o.glm(y=dep.h2o, x=indep.h2o, training_frame=train.h2o,
                      nfolds=10, family="binomial", seed=113)
predlog.h2o<-as.data.frame(h2o.predict(logmodel.h2o, test.h2o))                     
confusionMatrix(predlog.h2o$predict, test$Loan_Status, positive="1")

#randomforest again with cv
rfmodel.h2o<-h2o.randomForest(y=dep.h2o, x=indep.h2o, training_frame=train.h2o,
                              nfolds=10, ntrees = 500, mtries = 3, max_depth = 4, seed = 113)
predrf.h2o<-as.data.frame(h2o.predict(rfmodel.h2o, test.h2o))
confusionMatrix(as.numeric(predrf.h2o$predict>0.5), test$Loan_Status, positive="1")
h2o.shutdown()

#xgboost model 
library(xgboost)
xgbmodel<-xgboost(data=data.matrix(train[,indep]), label=train$Loan_Status,
                  eta=0.3, max_depth=15, nround=40, objective="binary:logistic",
                  eval_metric = "auc", seed=113)
predxgb<-predict(xgbmodel, newdata=data.matrix(test))
confusionMatrix(as.numeric(predxgb>0.5), test$Loan_Status, positive="1")

#nnet
fitControl<-trainControl(method="repeatedcv", number=10, repeats=5)
nnetmodel<-train(y=train[,dep], x=train[,indep],
                 method="nnet", trControl=fitControl)
prednnet<-predict(nnetmodel, newdata=test)
confusionMatrix(as.numeric(prednnet>0.5), test$Loan_Status, positive="1")

#convert testset 
Loan_ID<-loan_test[,1]
loan_test<-loan_test[,-1]
hot_encode<-dummyVars(" ~.", data=loan_test, fullRank=T)
loan_test<-data.frame(predict(hot_encode, newdata=loan_test))
str(loan_test)
colnames(loan_test)
clean<-c(11:15)
predxgb_test<-predict(xgbmodel, data.matrix(loan_test[,clean]))
Loan_Status<-as.numeric(predxgb_test>0.5)
First_result<-data.frame(Loan_ID, Loan_Status)
