
#Regression using Ames housing data

library(AmesHousing)
library(ggplot2)
library(dplyr)
library(tidyr)

#Cleaned data
ames <- AmesHousing::make_ames()

#Plot data with ordinary least squares linear regression line
ggplot(ames, aes(Year_Built, Sale_Price)) + 
  geom_point()+ 
  geom_smooth(method = "lm", color = "red")

#Linear model with multiple predictors

ames_lm <- lm(Sale_Price ~ Year_Built + Gr_Liv_Area, ames)


summary(ames_lm)

#Predicted vs. observed values
ames_copy <- ames%>%
  mutate(predicted = predict(ames_lm,ames))
  
ggplot(ames_copy, aes(predicted, Sale_Price))+
  geom_point(alpha = 0.4)

#Get RMSE to assess accuracy
#(has same units as predicted variable)
sqrt(mean((ames_copy$predicted - ames_copy$Sale_Price)^2))

#Get mean absolute error (MAE)
mean(abs((ames_copy$predicted - ames_copy$Sale_Price)))

#Assess model performance using RMSE
(model_performance <- ames %>%
    mutate(
      predictions_full = predict(
        lm(Sale_Price ~ Year_Built + Gr_Liv_Area, ames),
        .),
      predictions_year = predict(
        lm(Sale_Price ~ Year_Built, ames), 
        .),
      predictions_area = predict(
        lm(Sale_Price ~ Gr_Liv_Area, ames), 
        .)
    ) %>% 
    summarise(
      # This calculates RMSE "across" columns 
      # whose names start with "predictions"
      across(
        starts_with("predictions"), 
        ~ sqrt(mean((. - Sale_Price)^2)))
    ))

#Splitting data into training and test sets
set.seed(123)
row_idx <- sample(seq_len(nrow(ames)), nrow(ames))
training <- ames[row_idx < nrow(ames) * 0.8, ]
testing <- ames[row_idx >= nrow(ames) * 0.8, ]

nrow(ames)
nrow(training)

#RMSE on testing data set after using training data set
testing %>% 
  mutate(
    predictions_full = predict(
      lm(Sale_Price ~ Year_Built + Gr_Liv_Area, training), 
      .),
    predictions_year = predict(
      lm(Sale_Price ~ Year_Built, training), 
      .),
    predictions_area = predict(
      lm(Sale_Price ~ Gr_Liv_Area, training), 
      .)
  ) %>% 
  summarise(
    across(starts_with("predictions"), ~ sqrt(mean((. - Sale_Price)^2)))
  )

#Adding a categorical predictor (foundation type)
ames_2 <- lm(Sale_Price ~ Year_Built + Gr_Liv_Area + Foundation, training)

testing %>% 
  mutate(predictions_full = predict(ames_2, .)) %>% 
  summarise(predictions_full = sqrt(mean((predictions_full - Sale_Price)^2)))
#Slightly improves model fit

#R automatically converts categorical variables to booleans when
#performing "lm()"

#Manual boolean conversion aka "one-hot encoding"

one_hot_training <- training %>% 
  mutate(dummy_value = 1) %>% # Create a dummy value column
  pivot_wider(
    # "Turn each value of Foundation into a variable"
    names_from = Foundation,
    # Optional -- add the column name prefix lm uses
    names_prefix = "Foundation", 
    # "Fill the new variables with our dummy value"
    values_from = dummy_value, 
    # Will fill in all the other Foundation fields with 0s
    values_fill = 0 
  ) %>% 
  # Optional -- only keep the columns we care about
  select(starts_with("Foundation"), Sale_Price, Year_Built, Gr_Liv_Area)

head(one_hot_training, n = 2)

#Exclude brick tile foundations and run model on training data, This
#makes brick tile a dummy variable

one_hot_training %>% 
  select(-FoundationBrkTil) %>% 
  lm(Sale_Price ~., data = .) %>% 
  summary()

#NOTE: R automatically sets one dummy variable when doing its
#automatic one-hot encoding, because linear models don't deal well with
#one-hot encoded data. 









