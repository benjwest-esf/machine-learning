
#Quick notes on new-ish R functions####

#will use native pipes (|>)

#There's a new shorthand for function; do \(x){} instead of function(x){}

#dplyr-based (thank goodness)

library(dplyr)
library(ggplot2)

#Get summary stats across all columns (sick)
iris %>%
  group_by(Species)%?%
  summarize(across(everything(),mean))

#Example data and linear model####

#Generate data

set.seed(123)
plants <- data.frame(sunlight = rep(1:24, 5), 
                     growth = rep(1:24, 5) + rnorm(24 * 5, 0, 4),
                     #Generates 120 samples
                     noise = runif(24 * 5))


#Run simple linear model

plants_model <- lm(growth ~ sunlight + noise, data = plants)
plants_model

summary(plants_model)


#Plot model

ggplot(plants, aes(sunlight, growth)) + 
  geom_smooth(method = "lm", formula = y ~ x) + 
  theme(axis.text = element_blank())

#Plot model with data

ggplot(plants, aes(sunlight, growth)) + 
  geom_smooth(method = "lm", formula = y ~ x) + 
  geom_point() + 
  theme(axis.text = element_blank())

#Use linear model for predictions

plants_pred <-
plants%>%
  mutate(pred = predict(plants_model, plants))


ggplot(plants_pred, aes(sunlight, growth)) + 
  geom_point() + 
  geom_point(aes(y = pred), color = "red") + 
  theme(axis.text = element_blank())

#RMSE of predictions
sqrt(mean((plants_pred$pred - plants_pred$growth)^2))

