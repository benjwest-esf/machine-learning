install.packages(
  "lightgbm", 
  repos = "https://cran.microsoft.com/snapshot/2021-10-24/")

library(lightgbm)


ames <- AmesHousing::make_ames()

#Package that makes it easier to dummy encode
#(lightgbm doesn't do straight categorical variables)
install.packages("recipes")
library(recipes)

ames <- AmesHousing::make_ames()
ames <- recipe(Sale_Price ~ ., data = ames) |> 
  step_dummy(where(is.factor)) |> 
  prep() |> 
  bake(ames)
head(ames[36:309])

#Subset data
set.seed(123)
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#First lightgbm model####

#lightgbm takes a matrix of predictors and a vector for outcomes
xtrain <- as.matrix(training[setdiff(names(training), "Sale_Price")])
ytrain <- training[["Sale_Price"]]

#Fit a lightgbm model
first_lgb <- lightgbm(
  data = xtrain,
  label = ytrain,
  #"quiet" mode
  verbose = -1L,
  #specify regression
  obj = "regression",
)

#Convert test data to matrix
xtest <- as.matrix(testing[setdiff(names(testing), "Sale_Price")])

#Predictons from lightgbm
lgb_predictions <- predict(first_lgb, xtest)

#RMSE
sqrt(mean((lgb_predictions - testing$Sale_Price)^2))
#This RMSE sucks


#RMSE calculation function
calc_rmse <- function(model, data) {
  xtest <- as.matrix(data[setdiff(names(data), "Sale_Price")])
  lgb_predictions <- predict(model, xtest)
  sqrt(mean((lgb_predictions - data$Sale_Price)^2))
}

#Function to to do k-fold cross validation for lightgbm
k_fold_cv <- function(data, k, nrounds = 10L, ...) {
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
      xtrain <- as.matrix(fold_train[setdiff(names(fold_train), 
                                             "Sale_Price")])
      ytrain <- fold_train[["Sale_Price"]]
      fold_lgb <- lightgbm(
        data = xtrain,
        label = ytrain,
        verbose = -1L,
        obj = "regression",
        nrounds = nrounds,
        params = ...
      )
      calc_rmse(fold_lgb, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Tune standard GBM hyperparameters####
tuning_grid <- expand.grid(
  learning_rate = 0.1,
  #Number of trees in a model
  nrounds = c(10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000),
  rmse = NA
)

for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$rmse[i] <- k_fold_cv(
    training, 
    k = 5,
    learning_rate = tuning_grid$learning_rate[i],
    nrounds = tuning_grid$nrounds[i]
  )
}
head(arrange(tuning_grid, rmse), 2)

#Unique lightgbm syntax for standard GBM hyperparameters
#max_depth: maximum depth of trees (-1 is a special values that allows
#          unlimited depth)
#min_data_in_bin: minimum observations in a leaf

#Tune max_depth including the -1 argument
tuning_grid <- expand.grid(
  learning_rate = 0.1,
  nrounds = 1000,
  max_depth = c(-1, 2, 8, 32, 63),
  min_data_in_bin = c(3, 8, 13, 18),
  rmse = NA
)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$rmse[i] <- k_fold_cv(
    training, 
    k = 5,
    learning_rate = tuning_grid$learning_rate[i],
    nrounds = tuning_grid$nrounds[i],
    max_depth = tuning_grid$max_depth[i],
    min_data_in_bin = tuning_grid$min_data_in_bin[i]
  )
}
head(arrange(tuning_grid, rmse), 2)


#Tuning of stochastic hyperparameters####

#bagging_fraction: what percent of observations should be sampled for each tree,
#                  just like sampling.fraction back in random forests in ranger.

#bagging_freq: how often bootstrapping should be resampled. 
#              If you set it to 1, then each tree is fit to a different 
#              bootstrap sample; if you set it to 5, then the model will fit 
#              five trees to the same sample before resampling. 
#              If you set it to 0, the model won't resample ever.

#feature_fraction: the percentage of variables that should be available to 
#                  each tree. For instance, if you set it to 0.8, each tree 
#                  would be fit using a random 80% of predictors.

#Tuning grid for stochastic hyperparameters
tuning_grid <- expand.grid(
  learning_rate = 0.1,
  nrounds = 1000,
  max_depth = 2,
  min_data_in_bin = 18,
  bagging_freq = c(0, 1, 5, 10), 
  bagging_fraction = seq(0.3, 1.0, 0.1),
  feature_fraction = seq(0.3, 1.0, 0.1),
  rmse = NA
)
#Run grids
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$rmse[i] <- k_fold_cv(
    training, 
    k = 5,
    learning_rate = tuning_grid$learning_rate[i],
    nrounds = tuning_grid$nrounds[i],
    max_depth = tuning_grid$max_depth[i],
    min_data_in_bin = tuning_grid$min_data_in_bin[i],
    bagging_freq = tuning_grid$bagging_freq[i],
    bagging_fraction = tuning_grid$bagging_fraction[i],
    feature_fraction = tuning_grid$feature_fraction[i]
  )
}
head(arrange(tuning_grid, rmse), 2) |> 
  select(bagging_freq, bagging_fraction, feature_fraction, rmse)

#The seven hyperparameters in the above tuning grid are the most important
#There are 86 total that can be tuned

#Final lightgbm after tuning####
xtrain <- as.matrix(training[setdiff(names(training), "Sale_Price")])
ytrain <- training[["Sale_Price"]]
final_lgb <- lightgbm(
  data = xtrain,
  label = ytrain,
  verbose = -1L,
  obj = "regression",
  nrounds = 1000,
  params = list(
    learning_rate = 0.1,
    max_depth = 2,
    min_data_in_bin = 18,
    bagging_freq = 5,
    bagging_fraction = 0.9,
    feature_fraction = 0.6
  )
)
calc_rmse(final_lgb, testing)

#Stacked ensemble using averaged predictions####
#Combo multiple good models to get an even better one

#Tuned random forest
library(ranger)
final_ranger <- ranger(
  Sale_Price ~ .,
  training,
  num.trees = 800,
  mtry = 20,
  min.node.size = 5,
  replace = FALSE,
  sample.fraction = 1
)

#Tuned vanilla GBM
library(gbm)
final_gbm <- gbm(
  Sale_Price ~ .,
  data = training,
  n.trees = 6000,
  shrinkage = 0.01,
  interaction.depth = 5,
  n.minobsinnode = 5
)

#Copy test set to new object
stacked_test <- testing

#Add prediction columns to test set
xtest <- as.matrix(stacked_test[setdiff(names(stacked_test), 
                                        "Sale_Price")])
stacked_test$lgb <- predict(final_lgb, xtest)
stacked_test$rf <- predictions(predict(final_ranger, stacked_test))
stacked_test$gbm <- predict(final_gbm, stacked_test)

#Simplest stacking method: assign equal weights to all models
stacked_test$avg <- (stacked_test$lgb + 
                       stacked_test$rf + 
                       stacked_test$gbm) / 3

stacked_test |> 
  summarise(
    across(c(lgb, rf, gbm, avg), 
           \(x) sqrt(mean((x - Sale_Price)^2)))
  )

#Now going to weight models based on performance, using RMSE from k-fold CV

#Function to fit ranger model and return RMSE
fit_ranger <- function(training, testing) {
  cv_model <- ranger(
    Sale_Price ~ .,
    training,
    num.trees = 800,
    mtry = 20,
    min.node.size = 5,
    replace = FALSE,
    sample.fraction = 1
  )
  preds <- predictions(predict(cv_model, testing))
  sqrt(mean((preds - testing$Sale_Price)^2))
}

#Similar function as the one above, but for GBM
fit_gbm <- function(training, testing) {
  cv_model <- gbm(
    Sale_Price ~ .,
    data = training,
    n.trees = 6000,
    shrinkage = 0.01,
    interaction.depth = 5,
    n.minobsinnode = 5
  )
  preds <- predict(cv_model, testing)
  sqrt(mean((preds - testing$Sale_Price)^2))
}

#Another fit then RMSE function, this one for lightgbm
fit_lgb <- function(training, testing) {
  xtrain <- as.matrix(training[setdiff(names(training), "Sale_Price")])
  ytrain <- training[["Sale_Price"]]
  xtest <- as.matrix(testing[setdiff(names(testing), "Sale_Price")])
  cv_model <- lightgbm(
    data = xtrain,
    label = ytrain,
    verbose = -1L,
    obj = "regression",
    nrounds = 1000,
    params = list(
      learning_rate = 0.1,
      max_depth = 2,
      min_data_in_bin = 18,
      bagging_freq = 5,
      bagging_fraction = 0.9,
      feature_fraction = 0.6
    )
  )
  preds <- predict(cv_model, xtest)
  sqrt(mean((preds - testing$Sale_Price)^2))
}

#Now fit models to folds to do 5-fold CV

per_fold <- floor(nrow(training) / 5)
fold_order <- sample(seq_len(nrow(training)), 
                     size = per_fold * 5)
fold_rows <- split(
  fold_order,
  rep(1:5, each = per_fold)
)
model_rmse <- data.frame(
  rf = rep(NA, 5),
  gbm = rep(NA, 5),
  lgb = rep(NA, 5)
)
for (i in seq_along(fold_rows)) {
  fold_test <- training[fold_rows[[i]], ]
  fold_train <- training[-fold_rows[[i]], ]
  model_rmse$rf[i] <- fit_ranger(fold_test, fold_train)
  model_rmse$gbm[i] <- suppressWarnings(fit_gbm(fold_test, fold_train))
  model_rmse$lgb[i] <- fit_lgb(fold_test, fold_train)
}

model_rmse

model_rmse1 <- model_rmse

#Get mean RMSE for each model type
#NOTE: Putting parentheses around an assignment autoprints. Sick
(model_rmse <- apply(model_rmse1, 2, mean))

#Calculate weights

#Inverse RMSE (intermediate step for weights)
inverse_rmse <- (1 / model_rmse)

#Calculate weights from inverses
(rmse_weights <- inverse_rmse / sum(inverse_rmse))

#Apply weights to stacked ensemble prediction
stacked_test$weighted <- 
  stacked_test$lgb * rmse_weights[["lgb"]] + 
  stacked_test$rf * rmse_weights[["rf"]] + 
  stacked_test$gbm * rmse_weights[["gbm"]]
stacked_test |> 
  summarise(
    across(c(lgb, rf, gbm, avg, weighted), 
           \(x) sqrt(mean((x - Sale_Price)^2)))
  )

#More complex stacked ensemble using a model to aggregate predictions####

#Breaking off a validation set
training <- ames[row_idx < nrow(ames) * 0.6, ]
validation <- ames[row_idx >= nrow(ames) * 0.6 & 
                     row_idx < nrow(ames) * 0.8, ]

#Random forest
validation_rf <- ranger(
  Sale_Price ~ .,
  training,
  num.trees = 800,
  mtry = 20,
  min.node.size = 5,
  replace = FALSE,
  sample.fraction = 1
)

#GBM
validation_gbm <- gbm(
  Sale_Price ~ .,
  data = training,
  n.trees = 6000,
  shrinkage = 0.01,
  interaction.depth = 7,
  n.minobsinnode = 10
)

#lightgbm
xtrain <- as.matrix(training[setdiff(names(training), "Sale_Price")])
ytrain <- training[["Sale_Price"]]
validation_lgb <- lightgbm(
  data = xtrain,
  label = ytrain,
  verbose = -1L,
  obj = "regression",
  nrounds = 1000,
  params = list(
    learning_rate = 0.1,
    max_depth = 2,
    min_data_in_bin = 18,
    bagging_freq = 5,
    bagging_fraction = 0.9,
    feature_fraction = 0.6
  )
)

#Predict validation data
xtest <- as.matrix(validation[setdiff(names(validation), "Sale_Price")])
validation$rf <- predictions(predict(validation_rf, validation))
validation$gbm <- predict(validation_gbm, validation)
validation$lgb <- predict(validation_lgb, xtest)

#Fit model to predict sale price based on predictions
#(Using lm here; could use machine learning methods, too)
ensemble_model <- lm(Sale_Price ~ rf * gbm * lgb, data = validation)


#Look at predictive accuracy (RMSE) of stacked ensemble model
stacked_test$model_weight <- predict(ensemble_model, stacked_test)
stacked_test |> 
  summarise(
    across(c(lgb, rf, gbm, avg, weighted, model_weight), 
           \(x) sqrt(mean((x - Sale_Price)^2)))
  )

#Modeled aggregate didn't perform as well as just averaging


