# Multiple Linear Regression

# Import Dataset
dataset <- read.csv("50_Startups.csv")

# Encoding Categorical Variable 
dataset$State <- factor(dataset$State,levels = c("New York","California","Florida"),labels = c(1,2,3)) 

# Splitting the dataset in test and train
library(caTools)
set.seed(123)
split <- sample.split(dataset$Profit,SplitRatio = 0.8)
training_set <- subset(dataset,split == TRUE)
test_set <- subset(dataset,split == FALSE)

# Linear regression 
#regressor <- lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State,
#                 data=dataset)
regressor <- lm(formula = Profit ~ .,
                 data=dataset)
summary(regressor)
