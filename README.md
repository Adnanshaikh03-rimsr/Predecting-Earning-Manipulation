# Load necessary libraries
library(caret)
library(randomForest)
library(class)
library(ggplot2)

# Load the data
data <- read.csv("Earnings Manipulation 220.csv")
data$MANIPULATOR <- as.factor(data$MANIPULATOR)
data$MANIPULATOR <- factor(data$MANIPULATOR, levels = c("0", "1"))

# Summary and structure of the dataset
str(data)
summary(data)

# Splitting data into train and test sets (65% - 35%)
set.seed(123)
trainIndex <- createDataPartition(data$MANIPULATOR, p = 0.65, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Scaling numeric features
numeric_vars <- sapply(train_data, is.numeric)
train_scaled <- scale(train_data[, numeric_vars])
test_scaled <- scale(test_data[, numeric_vars])

train_scaled <- as.data.frame(train_scaled)
train_scaled$MANIPULATOR <- train_data$MANIPULATOR

test_scaled <- as.data.frame(test_scaled)
test_scaled$MANIPULATOR <- test_data$MANIPULATOR

# Implementing KNN
set.seed(123)
knn_pred <- knn(
  train = train_scaled[, numeric_vars],
  test = test_scaled[, numeric_vars],
  cl = train_scaled$MANIPULATOR,
  k = 5
)

# Confusion Matrix for KNN
confusionMatrix(knn_pred, test_scaled$MANIPULATOR)

# Building Random Forest model
set.seed(123)
rf_model <- randomForest(MANIPULATOR ~ ., data = train_scaled, ntree = 300, mtry = 5)

# Predictions
rf_pred <- predict(rf_model, newdata = test_scaled)

# Confusion Matrix for Random Forest
confusionMatrix(rf_pred, test_scaled$MANIPULATOR)

# Feature Importance plot
importance_df <- as.data.frame(rf_model$importance)
importance_df$Feature <- rownames(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Features", y = "Importance Score") +
  theme_minimal()
  
