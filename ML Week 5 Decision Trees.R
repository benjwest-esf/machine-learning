
#For decision trees
library(rpart)
library(rpart.plot)

#Model assessment
library(caret)
library(pROC)

#Attrition data
library(modeldata)
data(attrition)

#Create training and testing data####
set.seed(123)
row_idx <- sample(seq_len(nrow(attrition)), nrow(attrition))
training <- attrition[row_idx < nrow(attrition) * 0.8, ]
testing <- attrition[row_idx >= nrow(attrition) * 0.8, ]

#Decision tree model####
decision_tree <- rpart(Attrition ~ ., data = training)

#Confusion matrix
confusionMatrix(predict(decision_tree, training, type = "class"), 
                training$Attrition, 
                positive = "Yes")

#Plot decision tree
rpart(Attrition ~ ., data = training) |> 
  rpart.plot(type = 4, cex = 0.7)

#Create decision tree with only one split

rpart(Attrition ~ TotalWorkingYears, data = training) |> 
  rpart.plot(type = 4)

#Modify hyperparameters

#Reduce number of splits using maxdepth argument

rpart(Attrition ~ ., data = training, control = list(maxdepth = 2)) |> 
  rpart.plot(type = 4)

#Make sure each leaf has 100 observations using th minbucket argument (why?)

rpart(Attrition ~ ., data = training, control = list(minbucket = 100)) |> 
  rpart.plot(type = 4)

