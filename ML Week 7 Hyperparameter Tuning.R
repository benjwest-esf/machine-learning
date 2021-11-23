
#k-fold cross validation (5-fold CV)####
set.seed(123)
ames <- AmesHousing::make_ames()
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#Random forest package

library(ranger)

#Function for RMSE
calc_rmse <- function(rf_model, data) {
  rf_predictions <- predictions(predict(rf_model, data))
  sqrt(mean((rf_predictions - data$Sale_Price)^2))
}

#Default random forest
first_rf <- ranger(Sale_Price ~ ., training)

#RSME for training set
calc_rmse(first_rf, training)

#RSME for test set (not great; double other RMSE)
calc_rmse(first_rf, testing)

#Create folds
per_fold <- floor(nrow(training) / 5)

fold_order <- sample(seq_len(nrow(training)), 
                     size = per_fold * 5)

fold_rows <- split(
  fold_order,
  rep(1:5, each = per_fold)
)

str(fold_rows)

#Run model using first fold as a test set
first_fold_test <- training[fold_rows[[1]], ]
first_fold_train <- training[-fold_rows[[1]], ]

first_fold_rf <- ranger(Sale_Price ~ ., first_fold_train)
calc_rmse(first_fold_rf, first_fold_test)

#Run full cross-validation
cv_rmse <- vapply(
  fold_rows,
  \(fold_idx) {
    fold_test <- training[fold_idx, ]
    fold_train <- training[-fold_idx, ]
    fold_rf <- ranger(Sale_Price ~ ., fold_train)
    calc_rmse(fold_rf, fold_test)
  },
  numeric(1)
) 
cv_rmse

#Get mean cross-validated RMSE
mean(cv_rmse)

calc_rmse(first_rf, training)

#Cross-validation mean RMSE is much closer to the test set RMSE
calc_rmse(first_rf, testing)

#k-fold CV function that returns mean CV RMSE

#The "..." can be used to pass different hyperparameter levels to ranger
k_fold_cv <- function(data, k, ...) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      fold_rf <- ranger(Sale_Price ~ ., fold_train, ...)
      calc_rmse(fold_rf, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Easy 10-fold CV
k_fold_cv(training, 10)

#Pass different hyperparameter values to ranger
k_fold_cv(training, 10, num.trees = 10, min.node.size = 100)


#See Week 6 for different RF hyperparameters that can be tuned

#Hyperparameter tuning for regression using grid search####

#Make grid
tuning_grid <- expand.grid(
  mtry = floor(ncol(training) * c(0.3, 0.6, 0.9)),
  min.node.size = c(1, 3, 5), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(0.5, 0.63, 0.8),
  #Empty column to store RMSEs
  rmse = NA                                               
)
head(tuning_grid)

#Run grid search (takes a 2.5 mins)
{
  start_time <- Sys.time()
  
  for (i in seq_len(nrow(tuning_grid))) {
    tuning_grid$rmse[i] <- k_fold_cv(
      training, 
      k = 5,
      mtry = tuning_grid$mtry[i],
      min.node.size = tuning_grid$min.node.size[i],
      replace = tuning_grid$replace[i],
      sample.fraction = tuning_grid$sample.fraction[i]
    )
  }
  head(tuning_grid[order(tuning_grid$rmse), ])
  
  end_time <- Sys.time()
}

#(Curious about run time)
end_time - start_time

#Refine tuning grid 
tuning_grid_refined <- expand.grid(
  #Intermediate mtry values
  mtry = c(20, 25, 30, 35, 40, 45, 50),
  min.node.size = c(1, 3, 5),
  #Only replace = F
  replace = FALSE,
  #Shift sample fraction up
  sample.fraction = c(0.6, 0.8, 1),                       
  rmse = NA                                               
)

for (i in seq_len(nrow(tuning_grid_refined))) {
  tuning_grid_refined$rmse[i] <- k_fold_cv(
    training, 
    k = 5,
    mtry = tuning_grid_refined$mtry[i],
    min.node.size = tuning_grid_refined$min.node.size[i],
    replace = tuning_grid_refined$replace[i],
    sample.fraction = tuning_grid_refined$sample.fraction[i]
  )
}
head(tuning_grid_refined[order(tuning_grid_refined$rmse), ])

#Stop tuning here and use best model
grid_rf <- ranger(
  Sale_Price ~ .,
  training,
  num.trees = 800,
  mtry = 20,
  min.node.size = 5,
  replace = FALSE,
  sample.fraction = 1
)

#Tuned random forest has $1000 lower RMSE when used on test set
calc_rmse(grid_rf, testing)

calc_rmse(first_rf, testing)

#Variations on grid search

#Can use every single value of a hyperparameter (OK if computing power and/or
#time are available)

tuning_grid_full <- expand.grid(
  mtry = 1:80,
  min.node.size = 1:10, 
  replace = c(TRUE, FALSE),                               
  sample.fraction = seq(0.01, 1, 0.01),                       
  rmse = NA                                               
)

#Random grid search####
#Subsample the full tuning grid

which_trials <- sample(1:nrow(tuning_grid_full), 300)

tuning_grid_rand <- tuning_grid_full[which_trials, ]

for (i in seq_len(nrow(tuning_grid_rand))) {
  tuning_grid_rand$rmse[i] <- k_fold_cv(
    training, 
    k = 5,
    mtry = tuning_grid_rand$mtry[i],
    min.node.size = tuning_grid_rand$min.node.size[i],
    replace = tuning_grid_rand$replace[i],
    sample.fraction = tuning_grid_rand$sample.fraction[i]
  )
}
head(tuning_grid_rand[order(tuning_grid_rand$rmse), ])

#RMSE of best grid search model
random_rf <- ranger(
  Sale_Price ~ .,
  training,
  num.trees = 800,
  mtry = 30,
  min.node.size = 5,
  replace = FALSE,
  sample.fraction = 1
)

#Random grid seach random forest has slightly lower
calc_rmse(random_rf, testing)

calc_rmse(first_rf, testing)

#Hyperparameter tuning for classification####
#Using cross-entropy loss

#Example: attrition data
library(modeldata)
data(attrition)
row_idx <- sample(seq_len(nrow(attrition)), nrow(attrition))
training <- attrition[row_idx < nrow(attrition) * 0.8, ]
testing <- attrition[row_idx >= nrow(attrition) * 0.8, ]

#Untuned random forest
#Need probabilities to do cross-entropy loss
first_rf <- ranger(Attrition ~ ., training, probability = TRUE)

#Prediction outputs a matrix 
predict(first_rf, training) |> 
  predictions() |> 
  head(2)

#Add predictions to data frame
library(dplyr)
predict(first_rf, training) |> 
  predictions() |> 
  cbind(training) |> 
  head(2) |> 
  select(1:6)

#Get column that is probability of correct prediction
predict(first_rf, training) |> 
  predictions() |> 
  cbind(training) |> 
  mutate(prediction = ifelse(Attrition == "Yes", Yes, No)) |> 
  select(Attrition, No, Yes, prediction) |> 
  head(2)

#Take negative log of probability of correct prediction
predict(first_rf, training) |> 
  predictions() |> 
  cbind(training) |> 
  mutate(prediction = ifelse(Attrition == "Yes", Yes, No),
         loss = -log(prediction)) |> 
  select(Attrition, No, Yes, prediction, loss) |> 
  head(2)

#Make a cross entropy function
calc_cross_entropy <- function(rf_model, data) {
  data <- predict(rf_model, data) |> 
    predictions() |> 
    cbind(data) |> 
    mutate(prediction = ifelse(Attrition == "Yes", Yes, No),
           # Force prediction to not be exactly 0 or 1
           prediction = max(1e-15, min(1 - 1e-15, prediction)),
           loss = -log(prediction))
  sum(-log(data$prediction))
}
calc_cross_entropy(first_rf, testing)

#Alter k_fold_cv function to accomodate cross-entropy

k_fold_cv <- function(data, k, ...) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      fold_rf <- ranger(Attrition ~ ., 
                        fold_train, 
                        #Need probabilities from ranger
                        probability = TRUE, 
                        ...)
      #Do cross-entropy instead of RMSE
      calc_cross_entropy(fold_rf, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Intitial tuning grid

tuning_grid <- expand.grid(
  mtry = floor(ncol(training) * c(0.3, 0.6, 0.9)),
  min.node.size = c(1, 3, 5), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(0.5, 0.63, 0.8),                       
  loss = NA                                               
)
head(tuning_grid)

#Run models through tuning grid

for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv(
    training, 
    k = 5,
    mtry = tuning_grid$mtry[i],
    min.node.size = tuning_grid$min.node.size[i],
    replace = tuning_grid$replace[i],
    sample.fraction = tuning_grid$sample.fraction[i]
  )
}
head(tuning_grid[order(tuning_grid$loss), ])

#Confusion matrix
grid_rf <- ranger(
  Attrition ~ .,
  training,
  num.trees = 310,
  mtry = 9,
  min.node.size = 1,
  replace = TRUE,
  sample.fraction = 0.63
)


caret::confusionMatrix(
  predictions(predict(grid_rf, testing)),
  testing$Attrition,
  positive = "Yes"
)
#Sensitivity still isn't great

#Sidebar: thing to watch with expand.grid####

#expand.grid turns strings into factors by default, 
#need to set stringsAsFactors = F


expand.grid(a = c(1,3,5),
            b = c(2,4,6),
            c = c("a","b","c"),
            stringsAsFactors = F)

#can also use tidyr::crossing; doesn't do factor conversion


