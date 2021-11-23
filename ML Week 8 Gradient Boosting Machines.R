#Get data set ready

set.seed(123)
ames <- AmesHousing::make_ames()
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#Baseline, default decision tree
library(rpart)
first_tree <- rpart(Sale_Price ~ ., training)

training$first_prediction <- predict(first_tree, training)

#RMSE for training set (not great)
sqrt(mean((training$first_prediction - training$Sale_Price)^2))

#Get residuals
training$residual <- training$Sale_Price - training$first_prediction

#Boosting####

#New training set
library(dplyr)
new_training <- select(training, -Sale_Price, -first_prediction)

#Create a new tree to try to predict residuals 
second_tree <- rpart(residual ~ ., new_training)

#Add predictions together 
training$second_prediction <- predict(second_tree, training)
training$adjusted_prediction <- 
  training$first_prediction + training$second_prediction

#RMSE for base model for training set 
sqrt(mean((training$first_prediction - training$Sale_Price)^2))

#RMSE for boosted model for training set
sqrt(mean((training$adjusted_prediction - training$Sale_Price)^2))

#RMSEs for test sets

testing$first_prediction <- predict(first_tree, testing)
testing$second_prediction <- predict(second_tree, testing)
testing$adjusted_prediction <- 
  testing$first_prediction + testing$second_prediction

#Base model
sqrt(mean((testing$first_prediction - testing$Sale_Price)^2))

#Boosted model (RMSE not great but better than base)
sqrt(mean((testing$adjusted_prediction - testing$Sale_Price)^2))

#Use second model in boosted set as a weak indicator of how far off the base
#prediction was

#Halve the correction factor (aka "learning rate")
testing$adjusted_prediction <- 
  testing$first_prediction + (testing$second_prediction * 0.5)

#Orig RMSE
sqrt(mean((testing$first_prediction - testing$Sale_Price)^2))

#RMSE with halved correction factor (better than first boosted model but still
#meh)
sqrt(mean((testing$adjusted_prediction - testing$Sale_Price)^2))

#Third tree, fit on the residuals of the second tree, continue using a learning
#rate of 0.5
training$adjusted_prediction <- 
  training$first_prediction + (training$second_prediction * 0.5)
training$residual <- training$Sale_Price - training$adjusted_prediction


new_training <- select(training, 
                       -Sale_Price, 
                       -first_prediction, 
                       -second_prediction, 
                       -adjusted_prediction)
third_tree <- rpart(residual ~ ., new_training)


#Compare RMSE
testing$third_prediction <- predict(third_tree, testing)
testing$final_prediction <- 
  testing$adjusted_prediction + (testing$third_prediction * 0.5)

#Orig
sqrt(mean((testing$first_prediction - testing$Sale_Price)^2))

#One level of boosting
sqrt(mean((testing$adjusted_prediction - testing$Sale_Price)^2))

#Two levels of boosting (three trees)
sqrt(mean((testing$final_prediction - testing$Sale_Price)^2))

#Gradient boosting machines (GBM)

library(gbm)

#Reset the training data set
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#GBM with default settings, untuned
first_gbm <- gbm(Sale_Price ~ ., data = training)

#RMSE for untuned, default GBM (better than regression, 
#worse than random forest)
sqrt(mean((predict(first_gbm, testing) - testing$Sale_Price)^2))

#Main hyperparameters for tuning:

#shrinkage: The learning rate for the model.
#
#n.trees: How many trees to grow. Can overfit if too many trees are used
#
#interaction.depth: The maximum depth of each tree (how many splits). 
#                   Tends to be from 3-8, though older papers often use 
#                   1 to make decision stumps.
#
#n.minobsinnode: The minimum number of observations per leaf node 
#                in each tree (like min.node.size in ranger).


#Use a grid to search learning rate; record time to try to balance time spent
#and 

tuning_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  rmse = NA,
  trees = NA,
  time = NA
)

for(i in seq_len(nrow(tuning_grid))) {
  train_time <- system.time({
  m <- gbm(
    formula = Sale_Price ~ .,
    # Optional -- silences a warning:
    distribution = "gaussian",
    data = training,
    n.trees = 6000, 
    shrinkage = tuning_grid$learning_rate[i], 
    interaction.depth = 3, 
    n.minobsinnode = 10,
    cv.folds = 5 
  )
})
tuning_grid$rmse[i]  <- sqrt(min(m$cv.error))
#which.min gets the position of the minimum value
tuning_grid$trees[i] <- which.min(m$cv.error)
tuning_grid$time[i]  <- train_time[["elapsed"]]
}

#Look at RMSE and time
#(on Mike's computer the timings are different)
arrange(tuning_grid, rmse)

#Going to use 0.01 as the learning rate

#New grid testing two hyperparameters using selected learning rate
tuning_grid <- expand.grid(
  learning_rate = 0.01,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15),
  trees = NA,
  rmse = NA
)

for(i in seq_len(nrow(tuning_grid))) {
  m <- gbm(
    formula = Sale_Price ~ .,
    distribution = "gaussian",
    data = training,
    n.trees = 6000, 
    shrinkage = tuning_grid$learning_rate[i], 
    interaction.depth = tuning_grid$interaction.depth[i], 
    n.minobsinnode = tuning_grid$n.minobsinnode[i],
    cv.folds = 5 
  )
  #which.min gets the position of the minimum value
  tuning_grid$trees[i] <- which.min(m$cv.error)
  tuning_grid$rmse[i]  <- sqrt(min(m$cv.error))
}

#look at RMSE
head(arrange(tuning_grid, rmse))

#Use best GBM from previous grid as final GBM
final_gbm <- gbm(
  Sale_Price ~ .,
  data = training,
  n.trees = 6000,
  shrinkage = 0.01,
  interaction.depth = 5,
  n.minobsinnode = 5
)

#RMSE of final GBM (much better than fully tuned random forest)
sqrt(mean((predict(final_gbm, testing) - testing$Sale_Price)^2))
