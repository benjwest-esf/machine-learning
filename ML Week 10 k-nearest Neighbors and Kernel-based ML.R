#KNN for classification####

#Load data

library(modeldata)
data(attrition)
set.seed(123)
row_idx <- sample(seq_len(nrow(attrition)), nrow(attrition))
training <- attrition[row_idx < nrow(attrition) * 0.8, ]
testing <- attrition[row_idx >= nrow(attrition) * 0.8, ]

#Plot observations where colors are classes and axes are two useful predictors
library(ggplot2)
training |> 
  ggplot(aes(DailyRate, Age, color = Attrition)) + 
  geom_point() + 
  scale_color_brewer(palette = "Dark2")

#k-nearest neighbors (KNN) by default plots all axes with the same scale 
#(plot below is a visual representation)
training |> 
  ggplot(aes(DailyRate, Age, color = Attrition)) + 
  geom_point() + 
  scale_color_brewer(palette = "Dark2") + 
  scale_y_continuous(limits = c(0, max(training$DailyRate))) + 
  scale_x_continuous(limits = c(min(training$Age), max(training$DailyRate)))

#Center scale data to make the model work

#This recipe will be applicable to the test set, still using
#the mean and SD of the training set
library(recipes)
scaling_recipe <- recipe(Attrition ~ ., data = training) |> 
  step_center(where(is.numeric)) |> 
  step_scale(where(is.numeric)) |> 
  prep()

#Center scale data based on training dataset
training <- bake(scaling_recipe, training)
testing <- bake(scaling_recipe, testing)

#Do KNN model

library(caret)

(first_knn <- knn3(Attrition ~ ., training))

confusionMatrix(
  predict(first_knn, testing, type = "class"),
  testing$Attrition,
  positive = "Yes"
)
#Lol sensitivity is ass

#AUC of ROC
library(pROC)
auc(
  roc(
    testing$Attrition,
    predict(first_knn, testing)[, 2],
    plot = TRUE
  )
)
#Also sucks

#Resampling to balance classes

training <- attrition[row_idx < nrow(attrition) * 0.8, ]
testing <- attrition[row_idx >= nrow(attrition) * 0.8, ]
positive_training <- training[training$Attrition == "Yes", ]
negative_training <- training[training$Attrition == "No", ]
n_pos <- nrow(positive_training)
resampled_positives <- sample(1:n_pos, 
                              size = 5 * n_pos, 
                              replace = TRUE)
resampled_positives <- positive_training[resampled_positives, ]
training <- rbind(
  negative_training,
  resampled_positives
)

#Matrix of "probabilities" (actually proportions)
predict(first_knn, testing) |> head(1)

#AUC function (for the purpose of tuning)
calc_auc <- function(model, data) {
  roc(data$Attrition, predict(model, data)[, 2]) |> 
    auc() |> 
    suppressMessages()
}

#k-fold CV function, repurposed for KNN
k_fold_cv <- function(data, k, n) {
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
      scaling_recipe <- recipe(Attrition ~ ., data = fold_train) |> 
        step_center(where(is.numeric)) |> 
        step_scale(where(is.numeric)) |> 
        prep()
      fold_train <- bake(scaling_recipe, fold_train)
      fold_test <- bake(scaling_recipe, fold_test)
      fold_knn <- knn3(Attrition ~ ., fold_train, k = n)
      calc_auc(fold_knn, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Tuning grid where k increases evenly between 1 and 401
tuning_grid <- expand.grid(
  n = floor(seq(1, 401, length.out = 20)),
  auc = NA
)

#Add one to all even k values to prevent ties
tuning_grid$n <- ifelse(
  tuning_grid$n %% 2 == 0,
  tuning_grid$n + 1,
  tuning_grid$n
)

#Iterate over grid
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$auc[i] <- k_fold_cv(
    attrition,
    5,
    n = tuning_grid$n[i]
  )
}

head(arrange(tuning_grid, -auc))

#Plot AUC vs k
ggplot(tuning_grid, aes(n, auc)) + 
  geom_line() + 
  geom_point() + 
  labs(x = "k")

#Rescale both training and testing set to evaluate accuracy
scaling_recipe <- recipe(Attrition ~ ., data = training) |> 
  step_center(where(is.numeric)) |> 
  step_scale(where(is.numeric)) |> 
  prep()
training <- bake(scaling_recipe, training)
testing <- bake(scaling_recipe, testing)

#Accuracy evaluation
final_knn <- knn3(Attrition ~ ., training, k = 85)
calc_auc(final_knn, testing)

confusionMatrix(
  predict(final_knn, testing, type = "class"),
  testing$Attrition,
  positive = "Yes"
)

#KNN for regression####

#Load data
ames <- AmesHousing::make_ames()
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

#RMSE function
calc_rmse <- function(model, data) {
  sqrt(mean((predict(model, data) - data$Sale_Price)^2))
}

#Repurposed k-fold CV function
k_fold_cv <- k_fold_cv <- function(data, k, n) {
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
      
      #Need to scale each set of folds separately
      scaling_recipe <- recipe(Sale_Price ~ ., data = fold_train) |>
        #"- all_outcomes" avoids center scaling of sale price
        step_center(where(is.numeric), -all_outcomes()) |> 
        step_scale(where(is.numeric),  -all_outcomes()) |> 
        prep()
      fold_train <- bake(scaling_recipe, fold_train)
      fold_test <- bake(scaling_recipe, fold_test)
      #Using knnreg instead of knn3 with a different formula
      fold_knn <- knnreg(Sale_Price ~ ., fold_train, k = n)
      #Using calc_rmse as loss function
      calc_rmse(fold_knn, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#New tuning grid
tuning_grid <- expand.grid(
  n = floor(seq(1, 401, length.out = 20)),
  rmse = NA
)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$rmse[i] <- k_fold_cv(
    training,
    5,
    n = tuning_grid$n[i]
  )
}
head(arrange(tuning_grid, rmse))


ggplot(tuning_grid, aes(n, rmse)) + 
  geom_line() + 
  geom_point() + 
  labs(x = "k")

#Best k between 1 and 40; using 22 for now

#Q: Why do we rescale the data at the end?
#A: Each set of folds needs its own scaling

#Rescale both datasets
scaling_recipe <- recipe(Sale_Price ~ ., data = training) |> 
  step_center(where(is.numeric), -all_outcomes()) |> 
  step_scale(where(is.numeric), - all_outcomes()) |> 
  prep()
training <- bake(scaling_recipe, training)
testing <- bake(scaling_recipe, testing)

#Looks at RMSE of final KNN model
final_knn <- knnreg(Sale_Price ~ ., training, k = 22)
calc_rmse(final_knn, testing)
