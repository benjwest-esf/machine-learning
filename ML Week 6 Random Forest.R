
#Import Ames housing data and separate out testing and training 
set.seed(123)
ames <- AmesHousing::make_ames()
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]


#Single bootstrap resample
single_bootstrap <- training[sample(1:nrow(training), 
                                    size = nrow(training), 
                                    replace = TRUE), ]

#bootstrap aggregation, or bagging of multiple decision trees####

#100 bootstrap samples
ames_bootstraps <- lapply(
  1:100,
  \(x) training[sample(1:nrow(training), 
                       size = nrow(training), 
                       replace = TRUE), ]
)

#Multiple trees using bootstrapped data
library(rpart)
ames_trees <- lapply(
  ames_bootstraps,
  \(x) rpart(Sale_Price ~ ., x)
)

#Plot single decision tree from bootstrap aggregate
rpart.plot::rpart.plot(ames_trees[[1]])

#Add 100 new columns to testing dataframe, each with prediction form a different
#tree
tree_testing <- testing
for (i in seq_along(ames_trees)) {
  tree_testing[[paste0("tree_", i)]] <- predict(
    ames_trees[[i]],
    testing
  )
}

#Take mean of each prediction from the columns created above
library(dplyr)
tree_rmse <- tree_testing |> 
  summarize(
    across(
      starts_with("tree_"),
      ~ sqrt(mean((.x - tree_testing$Sale_Price)^2))
    )
  )

#Convert to long format dataset
library(tidyr)
tree_rmse <- tree_rmse |> 
  pivot_longer(everything(), values_to = "RMSE") |> 
  arrange(RMSE)

tree_rmse

#Compared bagged model to single tree####

single_tree <- rpart(Sale_Price ~ ., training)
sqrt(mean((predict(single_tree, training) - training$Sale_Price)^2))

head(tree_rmse, 1)

#Look at predictive power of individual trees in bagged model
tail(tree_rmse)

#Get hyperparameters from bagged model
new_params <- ranger(Sale_Price ~ ., 
                     training, 
                     num.trees = 800, 
                     mtry = 20, 
                     min.node.size = 1, 
                     replace = FALSE, 
                     sample.fraction = 1)
rf_predictions <- predictions(predict(new_params, testing))
sqrt(mean((rf_predictions - tree_testing$Sale_Price)^2))

#Fit each tree to a random 40% of predictors

ames_names <- setdiff(names(ames), "Sale_Price")

random_variable_trees <- lapply(
  ames_bootstraps,
  \(x) {
    # Select a random 40% of variables (and Sale_Price)
    x <- x[
      c(sample(ames_names, size = length(ames_names) * 0.4), 
        "Sale_Price")
    ]
    rpart(Sale_Price ~ ., x)
  }
)

tree_testing <- testing
for (i in seq_along(random_variable_trees)) {
  tree_testing[[paste0("tree_", i)]] <- predict(
    random_variable_trees[[i]],
    testing
  )
}
tree_testing <- tree_testing |> 
  mutate(tree_aggregate = rowMeans(across(starts_with("tree_"))))
tree_rmse <- tree_testing |> 
  summarize(
    across(
      starts_with("tree_"),
      ~ sqrt(mean((.x - tree_testing$Sale_Price)^2))
    )
  )
tree_rmse <- tree_rmse |> 
  pivot_longer(everything(), values_to = "RMSE") |> 
  arrange(RMSE)



#Random forest####

#Chooses different variables at each split of a decision tree
install.packages("ranger")
library(ranger)

#Random forest (same syntax as lm)
ames_rf <- ranger(Sale_Price ~ ., training)

#Predictions for random forest (need ranger::predictions to get things in
#right format)
rf_predictions <- predictions(predict(ames_rf, testing))

#Calculate RMSE
sqrt(mean((rf_predictions - tree_testing$Sale_Price)^2))

#Hyperparameters for random forest####

#min.node.size: controls depth of tree; lower = deeper tree
#Min value is one, practically tops out at around 20
#Set min node size to 1 (increase tree depth)
min_node <- ranger(Sale_Price ~ ., 
                   training, 
                   min.node.size = 1)
rf_predictions <- predictions(predict(min_node, testing))
#RMSE is a little better
sqrt(mean((rf_predictions - tree_testing$Sale_Price)^2))


#Alter more hyperparameters
# mtry: How many variables should be considered at each split?
# num.trees: How many decision trees should be aggregated?
# sample.fraction: What percent of observations should be sampled for each tree?
#                  If set to 1 and replace is set to TRUE, the random forest 
#                  will use bagged trees like we walked through earlier, but 
#                  it's often a good idea to mess with the sampling.
# replace: Sample observations with replacement?
new_params <- ranger(Sale_Price ~ ., 
                         training, 
                         num.trees = 800, 
                         mtry = 20, 
                         min.node.size = 1, 
                         replace = FALSE, 
                         sample.fraction = 1)
rf_predictions <- predictions(predict(new_params, testing))
sqrt(mean((rf_predictions - tree_testing$Sale_Price)^2))
