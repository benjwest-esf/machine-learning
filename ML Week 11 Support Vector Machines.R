install.packages("kernlab")

#NOTE: kernlab center scales automatically
library(kernlab)

#SVM for classification###
library(modeldata)
data(attrition)
set.seed(123)

row_idx <- sample(seq_len(nrow(attrition)), nrow(attrition))
training <- attrition[row_idx < nrow(attrition) * 0.8, ]
testing <- attrition[row_idx >= nrow(attrition) * 0.8, ]

first_svm <- ksvm(Attrition ~ ., training)
first_svm

#C is the number of values allowed within the margin
#Gaussian radial basis kernel function (RBF) is used to dimension cast

#Sigma is a kernel hyperparameter (does not need to be tuned if using RBF)

#Tuning C by powers of 2
tuning_grid <- data.frame(
  C = 2^(-8:8),
  loss = NA
)

#Balance classes so we can use overall accuracy as a loss function
positive_training <- training[training$Attrition == "Yes", ]
negative_training <- training[training$Attrition == "No", ]
n_pos <- nrow(positive_training)
resampled_positives <- sample(1:n_pos, 
                              size = 5 * n_pos, 
                              replace = TRUE)
resampled_positives <- positive_training[resampled_positives, ]
resampled_training <- rbind(
  negative_training,
  resampled_positives
)

#Accuracy function
calc_accuracy <- function(model, data) {
  matching <- predict(model, data) == data$Attrition
  sum(matching) / length(matching)
}


#k-fold CV function so we can run our tuning grid

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
      fold_svm <- ksvm(Attrition ~ ., fold_train, ...)
      calc_accuracy(fold_svm, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Now run the tuning grid in a loop
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv(
    resampled_training, 
    5,
    C = tuning_grid$C[i]
  )
}

#Accuracy vs. C
ggplot(tuning_grid, aes(C, loss)) + 
  geom_point() + 
  geom_line()

#Fit final model to entire training set
final_svm <- ksvm(
  Attrition ~ .,
  resampled_training,
  C = 64
)

calc_accuracy(final_svm, testing)

caret::confusionMatrix(
  predict(final_svm, testing),
  testing$Attrition,
  positive = "Yes"
)

#SVM for regression####

#Load data
ames <- AmesHousing::make_ames()
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#RMSE loss function
calc_rmse <- function(model, data) {
  predictions <- predict(model, data)
  sqrt(mean((predictions - data$Sale_Price)^2))
}

#Repurposed k-fold CV function
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
      fold_svm <- ksvm(Sale_Price ~ ., fold_train, ...)
      calc_rmse(fold_svm, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Tuning grid, now including epsilon 
#(generally a very  small positive number; controls variance around hyperplane)
tuning_grid <- expand.grid(
  C = 2^(-5:5),
  epsilon = 2^(-8:0),
  rmse = NA
)

#Run k-fold CV with tuning grid
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$rmse[i] <- k_fold_cv(
    training, 
    5,
    C = tuning_grid$C[i],
    epsilon = tuning_grid$epsilon[i]
  )
}
head(arrange(tuning_grid, rmse), 2)

#Final model
final_svr <- ksvm(
  Sale_Price ~ .,
  training,
  C = 2,
  epsilon = 2^-4
)

#RMSE of final model 
calc_rmse(final_svr, testing)
