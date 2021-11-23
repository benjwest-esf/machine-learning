library(modeldata)
library(dplyr)
library(ggplot2)
library(caret) #for confusion matrix and related metrics
library(pROC) #for ROC curves


#Load data
data(attrition)

#Trying to predict attrition for IBM employees

#Trying lm; causes warnings related to factors
lm(Attrition ~ Age, attrition)

#Try converting to factors and running again
attrition_cleaned <- attrition |>
  mutate(across(where(is.factor), as.character))

try(attrition_model <- lm(Attrition ~ Age, attrition_cleaned))

#Converting Attrition to binary numeric variable and creating linear model

attrition_cleaned <- attrition_cleaned |> 
  mutate(Attrition = dplyr::recode(Attrition, "Yes" = 1, "No" = 0))


attrition_model <- lm(Attrition ~ Age, attrition_cleaned)

summary(attrition_model)

#Plot linear model (doesn't look good)

#Top is employees who quit at a given age
#Older employees seem less likely to quit

ggplot(attrition_cleaned, aes(Age, Attrition)) + 
  geom_jitter(height = 0) + 
  geom_smooth(method = "lm", formula = "y ~ x")

#Linear model is bad because it doesn't predict probability

#Example: for 75 year old has a predicted negative value

ggplot(attrition_cleaned, aes(Age, Attrition)) + 
  scale_x_continuous(limits = c(10, 90)) + 
  geom_jitter(height = 0) + 
  stat_smooth(method = "lm", formula = "y ~ x", fullrange = TRUE)

#Better option: logistic model####

#Create training data set
set.seed(123)
#Randomizes row numbers
row_idx <- sample(seq_len(nrow(attrition_cleaned)), nrow(attrition_cleaned))
training <- attrition_cleaned[row_idx < nrow(attrition_cleaned) * 0.8, ]
testing <- attrition_cleaned[row_idx >= nrow(attrition_cleaned) * 0.8, ]

#fit model
attrition_model <- glm(Attrition ~ Age, training, family = "binomial")

summary(attrition_model)

#Prediction plot for logistic model isn't meaningful with defaults

qplot(predict(attrition_model,training))

#Need to set type to "response"
qplot(predict(attrition_model, training, type = "response"))

#Assess model accuracy
testing$prediction <- predict(attrition_model, testing, type = "response")|>
  round()

#All the predictions are 0 because all probabilities are <0.5
sum(testing$prediction == testing$Attrition) / length(testing$Attrition)
sum(0 == testing$Attrition) / length(testing$Attrition)

#Other metrics are better

#Confusion matrix and associated metrics####

attrition_confusion <- confusionMatrix(
  # Predictions go first, "true" values second:
  data = factor(testing$prediction, levels = 0:1),
  reference = factor(testing$Attrition, levels = 0:1),
  # Specify what level is your "hit" or "positive" value
  #(these are the rarer values, in this case quitting the job)
  positive = "1"
)

#This is a confusion matrix
attrition_confusion$table
#262 true negatives
#33 false negatives
#0 false positives
#0 true positives

#Other accuracy metrics
round(attrition_confusion$overall, 4)
#AccuracyNull is the accuracy if we just guess the most common class
#(same as overall accuracy in this case)

#very useful part of confusionMatrix output
round(attrition_confusion$byClass, 3)

#Has predictive values, sensitivity, and specificity (see notes in Word)

#Model has worst sensitivity, best specificity

#ROC curves###


attrition_roc <- roc(
  testing$Attrition,
  predict(attrition_model, testing, type = "response")
)


#plots trade offs between sensitivity and specificity

plot(attrition_roc)

#Find best probability threshold to get closest to perfect accuracy (upper
#left corner of plot)

coords(attrition_roc, "best")

#^Normally done with validation set, not test set^

#Use threshold value from ROC curve 

testing$prediction <- predict(attrition_model, testing, type = "response")
testing$prediction <- as.numeric(testing$prediction > 0.20499)
confusionMatrix(factor(testing$prediction), 
                factor(testing$Attrition), 
                positive = "1")

#Overall accuracy is lower, but we now do a better job of predicting positive
#values

#Table of sensitiviy and specificity
coords(attrition_roc)

#Plot area under curve
plot(attrition_roc, auc.polygon = TRUE)

#Get area under curve
auc(attrition_roc)


