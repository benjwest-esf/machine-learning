
library(renv)
renv::restore()

#Other packages####

#Data handling and processing
library(dplyr)
library(readr)
library(tidyr)
library(tibble)

#Model assessment
library(caret)
library(pROC)
library(e1071) #needed for confusion matrix

#For decision trees
library(rpart)
library(rpart.plot) #Not essential; I just wanted to see the plot

#For random forest
library(ranger)

#For lightgbm
library(lightgbm)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data stuff####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Read in data####
bird_data <- read_csv("ML Project/Max Count of Birds by Point.csv")
hab_data <- read_csv("ML Project/Final Habitat Bird Points.csv")

#Need to do some light processing; only want 2019 Acadian Flycatcher detections/
#no detection and 2019 habitat
ACFL_data <-
bird_data %>%
  filter(yr == 2019)%>%
  filter(species == 'ACFL')%>%
  unite(point_id, yr, col = "point_yr", sep = "_")%>%
  left_join(.,hab_data)%>%
  select(-year)%>%
  mutate(ACFL_detect = case_when(
    max_count == 0 ~ "No",
    TRUE ~ "Yes"
  ))%>%
  mutate(across(ACFL_detect, as.factor))%>%
  dplyr::select(ACFL_detect, everything())%>%
  dplyr::select(-c(max_count,point_yr,species,bca))

#Setting up test and training data###
set.seed(123)
row_idx <- sample(seq_len(nrow(ACFL_data)), nrow(ACFL_data))
training <- ACFL_data[row_idx < nrow(ACFL_data) * 0.8, ]
testing <- ACFL_data[row_idx >= nrow(ACFL_data) * 0.8, ]


#Classes are imbalanced; going to resample####

#Separate classes into two data sets
positive_training <- training[training$ACFL_detect == "Yes", ]
negative_training <- training[training$ACFL_detect == "No", ]

#Resample the positives in the training data set
n_pos <- nrow(positive_training)

set.seed(96)
resampled_positives <- sample(1:n_pos, 
                              #Same number of positives as negatives
                              size = nrow(negative_training), 
                              replace = TRUE)
resampled_positives <- positive_training[resampled_positives, ]

#Combine resampled positives with negatives
resampled_training <- rbind(
  negative_training,
  resampled_positives)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Decision tree model (including tuning)####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Cross-entropy loss function for decision tree

calc_cross_entropy_tree <- function(tree_model, data, ctrl) {
  data <- predict(tree_model, data, type = "prob") %>%
    tibble(prediction = .)%>%
    cbind(data) |> 
    mutate(# Force prediction to not be exactly 0 or 1
           prediction = max(1e-15, min(1 - 1e-15, prediction)),
           loss = -log(prediction))
  sum(-log(data$prediction))
}

#k_fold_cv function for decision tree that accommodates cross-entropy

k_fold_cv_tree <- function(data, k, minbucket) {
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
      fold_tree <- rpart(ACFL_detect ~ ., 
                        fold_train, 
                        control = list(minbucket = minbucket)
                        )
      #Do cross-entropy instead of RMSE
      calc_cross_entropy_tree(fold_tree, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

#Initial tuning grid

tuning_grid <- tibble(minbucket = 1:40,
                      loss = NA)

#setting seed to avoid random fluctations
set.seed(96)
for (i in seq_len(nrow(tuning_grid))) {
  
  tuning_grid$loss[i] <- k_fold_cv_tree(
    resampled_training, 
    k = 5,
    minbucket = tuning_grid$minbucket[i]
  )
}

tuning_grid%>%
  arrange(loss)

decision_tree <- rpart(ACFL_detect ~ ., data = resampled_training,
                       control = list(minbucket = 31))

rpart.plot(decision_tree, type = 4)
#Confusion matrix, decision tree, testing data

(dtree_cmat <-
confusionMatrix(predict(decision_tree, testing, type = "class"), 
                testing$ACFL_detect, 
                positive = "Yes"))

(dtree_roc <-
roc(
  testing$ACFL_detect,
  predict(decision_tree, testing, type = "prob")[,2]%>%
    as.vector()
  
)%>%
  auc())


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Random forest model (including tuning)####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Cross-entropy loss function for random forest
calc_cross_entropy <- function(rf_model, data) {
  data <- predict(rf_model, data) |> 
    predictions() |> 
    cbind(data) |> 
    mutate(prediction = ifelse(ACFL_detect == "Yes", Yes, No),
           # Force prediction to not be exactly 0 or 1
           prediction = max(1e-15, min(1 - 1e-15, prediction)),
           loss = -log(prediction))
  sum(-log(data$prediction))
}

#k_fold_cv function for random forests that accommodates cross-entropy

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
      fold_rf <- ranger(ACFL_detect ~ ., 
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

#Tuning grid

tuning_grid <- expand.grid(
  mtry = floor(ncol(resampled_training) * c(0.3, 0.6, 0.9)),
  min.node.size = c(1, 3, 5), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(0.5, 0.63, 0.8),                       
  loss = NA                                               
)

#Run models through tuning grid

set.seed(96)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv(
    resampled_training, 
    k = 5,
    mtry = tuning_grid$mtry[i],
    min.node.size = tuning_grid$min.node.size[i],
    replace = tuning_grid$replace[i],
    sample.fraction = tuning_grid$sample.fraction[i]
  )
}
head(tuning_grid[order(tuning_grid$loss), ])

#Tuned random forest

#non-probability for confusion matrix
{set.seed(96)
grid_rf <- ranger(
  ACFL_detect ~ .,
  resampled_training,
  num.trees = 800,
  mtry = 5,
  min.node.size = 3,
  replace = T,
  sample.fraction = 0.63
)}

#Probability for ROC curve
{set.seed(96)
grid_rf_roc <- ranger(
  ACFL_detect ~ .,
  resampled_training,
  num.trees = 800,
  mtry = 5,
  min.node.size = 3,
  replace = T,
  sample.fraction = 0.63,
  probability = T
)}

#Confusion matrix for training data (making sure I can't further refine tuning)
caret::confusionMatrix(
  predictions(predict(grid_rf, resampled_training)),
  resampled_training$ACFL_detect,
  positive = "Yes"
)
#Fits training set perfectly!

#Confusion matrix for testing data
(rf_cmat <- caret::confusionMatrix(
  predictions(predict(grid_rf, testing)),
  testing$ACFL_detect,
  positive = "Yes"
))


(rf_roc <-
    roc(
      testing$ACFL_detect,
      predictions(predict(grid_rf_roc, testing))[,2]
      
    )%>%
    auc())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#lightgbm model (including tuning)####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Cross-entropy loss function for lightgbm
calc_cross_entropy_lgbm <- function(lgbm_model, data) {
  data <- predict(lgbm_model, data) %>%
    tibble(prediction = .)%>%
    cbind(data) |> 
    mutate(
           # Force prediction to not be exactly 0 or 1
           prediction = max(1e-15, min(1 - 1e-15, prediction)),
           loss = -log(prediction))
  sum(-log(data$prediction))
}


#Convert ACFL_detect testing and training data to binary instead of factor
testing_lgbm <- testing%>%
  mutate(ACFL_bin = case_when(
    ACFL_detect == "Yes" ~ 1,
    T ~ 0
  ))%>%
  select(-ACFL_detect)

training_lgbm <- resampled_training%>%
  mutate(ACFL_bin = case_when(
    ACFL_detect == "Yes" ~ 1,
    T ~ 0
  ))%>%
  select(-ACFL_detect)


#Function to convert binary 0 or 1 to Yes/No factor so the confusion matrix
#will take it
bin_to_fact <- \(x){
    ifelse(x == 0, "No","Yes")%>%
    factor(levels = c("No","Yes"))
}

#lightgbm takes a matrix of predictors and a vector for outcomes
xtrain <- as.matrix(training_lgbm[setdiff(
  names(training_lgbm), "ACFL_bin")])

#Need to make things binary 0 and 1 instead of factors
ytrain <- training_lgbm$ACFL_bin

#k_fold_cv function for lightgbm that accommodates cross-entropy

k_fold_cv_lgbm <- function(data, k, nrounds = 10L, ...) {
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
                                             "ACFL_bin")])
      ytrain <- fold_train[["ACFL_bin"]]
      fold_lgb <- lightgbm(
        data = xtrain,
        label = ytrain,
        verbose = -1L,
        obj = "binary",
        nrounds = nrounds,
        params = ...
      )
      calc_cross_entropy_lgbm(fold_lgb, xtrain)
    },
    numeric(1)
  ) |> 
    mean()
}

#Tune nrounds and learning rate
tuning_grid <- expand.grid(
  learning_rate = c(0.01, 0.05, 0.1),
  #Number of trees in a model
  nrounds = c(10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000),
  loss = NA
)

{set.seed(96)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv_lgbm(
    training_lgbm, 
    k = 5,
    learning_rate = tuning_grid$learning_rate[i],
    nrounds = tuning_grid$nrounds[i]
  )
}}
head(arrange(tuning_grid, loss), 5)


#Tune max_depth including the -1 argument
#Also tuning min_data_in_bin
tuning_grid <- expand.grid(
  learning_rate = 0.01, #from previous tuning grid iteration
  nrounds = 10, #from previous tuning grid iteration
  max_depth = c(-1, 2, 8, 32, 63), 
  min_data_in_bin = c(3, 8, 13, 18), 
  loss = NA
)

{set.seed(96)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv_lgbm(
    training_lgbm, 
    k = 5,
    learning_rate = tuning_grid$learning_rate[i],
    nrounds = tuning_grid$nrounds[i],
    max_depth = tuning_grid$max_depth[i],
    min_data_in_bin = tuning_grid$min_data_in_bin[i]
  )
}
}
head(arrange(tuning_grid, loss), 5)

#Tuning grid for stochastic hyperparameters
tuning_grid <- expand.grid(
  #Previously tuned, standard GBM parameters
  learning_rate = 0.01,
  nrounds = 10,
  max_depth = 2,
  min_data_in_bin = 8,
  #Stochastic
  bagging_freq = c(0, 1, 5, 10), 
  bagging_fraction = seq(0.3, 1.0, 0.1),
  feature_fraction = seq(0.3, 1.0, 0.1),
  loss = NA
)

#Run grid
{set.seed(96)
for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$loss[i] <- k_fold_cv_lgbm(
    training_lgbm, 
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
}
head(arrange(tuning_grid, loss), 5)

tuned_lgb <- lightgbm(
  data = xtrain,
  label = ytrain,
  #"quiet" mode
  verbose = -1L,
  #specify binary classification
  obj = "binary",
  learning_rate = 0.01,
  nrounds = 10,
  max_depth = 2,
  min_data_in_bin = 8,
  bagging_freq = 1,
  bagging_fraction = 0.3,
  feature_fraction = 0.3
  
)

#Testing data from lightgbm
xtest <- as.matrix(testing[setdiff(names(testing), "ACFL_detect")])


#Predictions from tuned lightgbm
lgb_predictions <- predict(tuned_lgb, xtest)%>%
  #default rounding (OK because I rebalanced classes)
  round(0)%>%
  bin_to_fact()

#Confusion matrix, untuned LGBM, testing data
(lgb_cmat <- confusionMatrix(lgb_predictions, 
                bin_to_fact(testing_lgbm$ACFL_bin), 
                positive = "Yes"))

(lgb_roc <-
    roc(
      testing$ACFL_detect,
      predict(tuned_lgb, xtest)
    )%>%
    auc())

########################

#Convert test data to matrix
xtest <- as.matrix(testing[setdiff(names(testing), "ACFL_detect")])

#Results table ####
results_table <-
list(list(dtree_cmat,"Decision Tree"),
      list(rf_cmat,"Random Forest"),
     list(lgb_cmat,"LightGBM"))%>%
lapply(\(x){
  tibble(
   `Model Type` = x[[2]],
    Accuracy =  x[[1]]$overall["Accuracy"],
    Sensitivity = x[[1]]$byClass["Sensitivity"],
    Specificity = x[[1]]$byClass["Specificity"]
  )
})%>%
  bind_rows()%>%
  mutate(AUC = c(as.numeric(dtree_roc),
                 as.numeric(rf_roc),
                 as.numeric(lgb_roc)
                   ))%>%
  mutate(across(where(is.numeric), round, 3))%>%
  arrange(desc(Accuracy))#%>%
  #dplyr::select(`Model Type`, AUC, everything())
  
write_csv(results_table,"ACFL_ML_results.csv")


