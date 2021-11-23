library(modeldata)
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)

#Setup unmodified data####
data(attrition)

#Data cleaning
attrition_cleaned <- 
  attrition |> 
  #"where" function is super handy; stash that away
  mutate(across(where(is.factor), as.character)) |>
  mutate(Attrition = recode(Attrition, "Yes" = 1, "No" = 0))

#Create training and testing sets
set.seed(123) #this sets the start of a sequence of seeds; if you do another
              #random draw it will be seed 124
row_idx <- sample(seq_len(nrow(attrition_cleaned)), nrow(attrition_cleaned))
training <- attrition_cleaned[row_idx < nrow(attrition_cleaned) * 0.8, ]
testing <- attrition_cleaned[row_idx >= nrow(attrition_cleaned) * 0.8, ]

#Basic  model with all predictors####
attrition_model <- glm(Attrition ~ ., 
                       training, 
                       family = "binomial")

#Use probability threshold of 0.5 for classification
#(Q: where does this 0.5 come from?)
testing$prediction <- predict(attrition_model, 
                              testing, 
                              type = "response")

#Rounds up to 1 if >=0.5, down to 0 if <0.5
testing$prediction <- round(testing$prediction)

#Confusion matrix
attrition_confusion <- confusionMatrix(
  factor(testing$prediction),
  factor(testing$Attrition),
  positive = "1"
)
attrition_confusion

#ROC curve
attrition_roc <- roc(
  testing$Attrition,
  predict(attrition_model, testing, type = "response")
)
#Area under curve is 0.896

plot(attrition_roc)

#Model still mostly ignores "positive cases," i.e., quitters

#Imbalance is due to relatively low number of quitters
#This is an imbalanced classification problem
table(training$Attrition)

#Attempt to fix imbalance by weighting observations####
#so that "yes" observations are treated the same as "no's"

training_weights <- ifelse(training$Attrition, 5, 1)
training |> 
  mutate(weight = training_weights) |> 
  select(Age, Attrition, weight) |> 
  head()

#Apply weights to model
weighted_model <- glm(
  Attrition ~ ., 
  training,
  weights = training_weights, 
  family = "binomial")

testing$prediction <- predict(weighted_model, testing, type = "response")
testing$prediction <- round(testing$prediction)

#Confusion matrix for weighted model
confusionMatrix(factor(testing$prediction), 
                factor(testing$Attrition), 
                positive = "1")
#(accuracy not as good)

#ROC curve for weighted model
weighted_roc <- roc(
  testing$Attrition,
  predict(weighted_model, testing, type = "response")
)
#Area under curve is 0.857

#Weighted model has much higher sensitivity, lower specificity and AUC

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Sidebar on sampling####
#New example: Theoretical heights for every man in the country
#(not the sick "e" hack)
heights <- rnorm(1e5, mean = 68, sd = 4)
qplot(heights)

#Usually can't get data for an entire population; have to rely on samples,
#which may not resemble the population, especially with small population
sampled_heights <- sample(heights, 10)
qplot(sampled_heights)

#Upping sample size increases the resemblance of the sample to the population
sampled_heights <- sample(heights, 1e4)
qplot(sampled_heights)

#"Super sample" using sampling with replacement to create a sample
#larger than the population. Distribution is almost identical to
#that of the population
sampled_heights <- sample(heights, 1e6, replace = TRUE)
qplot(sampled_heights)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Back to attrition

#Do resampling of attrition####

#Separate classes into two data sets
positive_training <- training[training$Attrition == 1, ]
negative_training <- training[training$Attrition == 0, ]

#Resample the positives in the training data set
n_pos <- nrow(positive_training)
resampled_positives <- sample(1:n_pos, 
                              size = 5 * n_pos, 
                              replace = TRUE)
resampled_positives <- positive_training[resampled_positives, ]

#Combine resampled positives with negatives
resampled_training <- rbind(
  negative_training,
  resampled_positives)
  
table(resampled_training$Attrition)

#Fit model using resampled data
resampled_model <- glm(
  Attrition ~ ., 
  resampled_training,
  family = "binomial")

#Make predictions with probability threshold of 0.5
testing$prediction <- predict(resampled_model, 
                              testing, 
                              type = "response")
testing$prediction <- round(testing$prediction)

#Confusion matrix
confusionMatrix(factor(testing$prediction), 
                factor(testing$Attrition), 
                positive = "1")

#ROC and AUC
resampled_roc <- roc(
  testing$Attrition,
  predict(resampled_model, testing, type = "response")
)

plot(resampled_roc)

#Area under the curve: 0.8498; oops, worst one yet

#Weighte and resampled models are similar


