 #####################################################
## CA2 - Heart Attack Analysis - predictive modeling
#####################################################

# import the heart data frame
# Store file into working directory
# and then read into a data frame. 
heart_data <- read.csv("heart.csv")

# view top records
head(heart_data)

##############################
## Data Prep
##############################
# rename trtbps (4th item in dataframe) to RestingBloodPressure
names(heart_data)[4] <- "RestingBloodPressure"
# rename chol (5th item in dataframe) to Cholestoral
names(heart_data)[5] <- "Cholestoral"
# rename thalachh  (8th item in dataframe) to MaxHeartRate
names(heart_data)[8] <- "MaxHeartRate"
# rename output (14th item in dataframe) to HeartAttackRisk
# names(heart_data)[14] <- "HeartAttackRisk"

# remove age as it is not relevant to the model
#heart_data <- subset(heart_data, select = -c(age))
# remove sex as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(sex))
# remove cp as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(cp))
# remove fbs as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(fbs))
# remove restecg as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(restecg))
# remove exng as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(exng))
# remove slp as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(slp))
# remove caa as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(caa))

# remove sex as it is not relevant to the model
heart_data <- subset(heart_data, select = -c(output))

# Remove Coloumns not in assignment handout - thall and oldpeak 
heart_data$thall <- NULL
heart_data$oldpeak <- NULL

# Test Results
head(heart_data)

# View the summary of the data
summary(heart_data)

######################################
# create training and testing datasets
######################################
set.seed(1) #reproduce same samples as original dataset
no_rows_data <- nrow(heart_data)
sample <- sample(1:no_rows_data, size=round(0.7 * no_rows_data), replace = FALSE) # 70% of data from training, 30% for testing

training_data <- heart_data[sample,]
testing_data <- heart_data[-sample,]

#####################
# build the MLR model
#####################
# build regression model and save to data 'fit'
# research question: does age, Resting blood pressure and MaxHeartRate affect level of cholesterol level?
fit <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)

# MLR model evaluation
summary(fit)
# resting blood pressure coefficient p <0.02
# age  coefficient p <0.003
# maxheartrate  coefficient p <0.12
# All of the predictor variables account for 0.6% of the variance in cholesterol rates level. This is indicated by the Multiple R-squared value.
# predicter variables do not interact

# model summary
confint(fit)

# analyse model - qqPlot
library(car)
install.packages("qqPlot")
qqPlot(fit, labels=row.names(heart_data), id.method="identify", simulate=TRUE, main="Q-Q Plot")

#qqPlot generates the probability plot and shows that records 86 showing as outlier
# with exceptions of the outliers are close to the line
training_data[86,] # shows cholesterol is 304

fitted(fit)[86] 
# These results show that the cholesterol rate for one patient is 304 however the predicted model predicts 278 for the same patient

# histogram
student_fit <- rstudent(fit)
hist(student_fit,
     breaks=10,
     freq=FALSE,
     xlab="Studentized Residuals",
     main="Distribution of Errors")

rug(jitter(student_fit), col="brown")

curve(dnorm(x, mean=mean(student_fit), sd=sd(student_fit)), add=TRUE, col="blue", lwd=2)

lines(density(student_fit)$x, density(student_fit)$y, col="red", lwd=2, lty=2)

legend("topright", legend = c("Normal Curve", "Kernel Density Curve"), lty=1:2, col=c("blue", "red"), cex=.7)
# Errors follow a normal distribution with the exception of 1 outlier less than 0 boundary.
# There is no outlier with standardized results that are larger or less than 2/-2 therefore no further attention is needed

#outlierTest
outlierTest(fit)
# record 86 is again showing as an outlier using the Bonferroni p test

# remove 86 records from heart_data
# Drop rows using slice() function in R
library(dplyr)
heart_data <- heart_data %>% slice(-c(86))

# split the data into training and testing again
set.seed(1)
no_rows_data <- nrow(heart_data)
sample <- sample(1:no_rows_data, size = round(0.7 * no_rows_data), replace = FALSE)

training_data <- heart_data[sample,]
testing_data <- heart_data[-sample,]

training_data

# rebuild MLR again
fit <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)

# re-run outlier test  
outlierTest(fit)

# histogram
student_fit <- rstudent(fit)
hist(student_fit,
     breaks=10,
     freq=FALSE,
     xlab="Studentized Residuals",
     main="Distribution of Errors")

rug(jitter(student_fit), col="brown")

curve(dnorm(x, mean=mean(student_fit), sd=sd(student_fit)), add=TRUE, col="blue", lwd=2)

lines(density(student_fit)$x, density(student_fit)$y, col="red", lwd=2, lty=2)

legend("topright", legend = c("Normal Curve", "Kernel Density Curve"), lty=1:2, col=c("blue", "red"), cex=.7)
# kernel density curve is now more similar to the normal curve after removing the outlier

# investigating nonlinearity in the relationship between the dependent variable and the independent variables:
crPlots(fit)
# all variables are linear, no sharp deviation from the linear line is found

# investigating influential observations:
cutoff <- 4/(nrow(training_data) - length(fit$coefficients) - 2)
plot(fit, which = 4, cook.levels = cutoff)
abline(h = cutoff, lty = 2, col = "red")
# patients with cholesterol level 29, 220 and 151 are indicated above the red line and are influencial observation

# added variable plot:
avPlots(fit, ask=FALSE)

# influent plot:
library(car)
influencePlot(fit, main="Influence Plot",
              sub="Circle size is proportional to Cook's distance")
# patients with cholesterol levels of 246, 29 and 151 are beyond -2 to 2 boundary and could be outliers

# homoscedasticity - data to have variance in our residuals
ncvTest(fit)
# significant result, (p=0.020907), suggesting that we have not met the constant variance assumption. 
# If a P value is significant (p < 0.05), the error variance changes with the level of fitted values

# create a scatter plot of the absolute standardized residuals versus the fittted values
spreadLevelPlot(fit)

# Global validation of linear model assumption
install.packages("gvlma")
library(gvlma)
gvmodel <- gvlma(fit)
summary(gvmodel)
# from the Global validation of linear model assumption reults, we have not met Global Stat, Skewness and Kurtosis

library(car)
vif(fit)

# We can check whether any of the variables indicate a multicollinearity problem
# if the value > 2
sqrt(vif(fit)) > 2
# variance inflation factor (VIF) all returned back false
library(car)
summary(powerTransform(training_data$Cholestoral))

# Transform Murder variable as indicated by spreadLevelPlot() function
sqrt_transform_chol <- sqrt(training_data$Cholestoral)
training_data$Cholestoral_sqrt <- sqrt_transform_chol
fit_model1 <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)
fit_model2 <- lm(Cholestoral_sqrt ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)
AIC(fit_model1,fit_model2)

spreadLevelPlot(fit_model2)

# STEPWISE REGRESSION - model that includes all predictor variables, and then remove one variable at a time
library(MASS)
fit_test <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)
stepAIC(fit_test, direction="backward")
# AIC decreasing from 1625.9 to 1628.0

#  subsets regression
install.packages("leaps")
library(leaps)
leaps <-regsubsets(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data, nbest=4)
plot(leaps, scale="adjr2")
# model with Resting Blood Presure and age only is the best

#  subsets regression with AIC
library(MASS)
fit_test <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)
stepAIC(fit_test, direction="backward")

#library(leaps)
leaps <-regsubsets(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data, nbest=4)
plot(leaps, scale="adjr2")

# Predicted accuracy:
fit_model <- lm(Cholestoral ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)
fit_model_sqrt <- lm(Cholestoral_sqrt ~ RestingBloodPressure + age + MaxHeartRate, data = training_data)

predicted_Cholestoral <- predict(fit_model, testing_data)
predicted_Cholestoral_sqrt <- predict(fit_model_sqrt, testing_data)
converted_Cholestoral_sqrt <- predicted_Cholestoral_sqrt ^2

# make actuals_predicted dataframe.
actuals_predictions <- data.frame(cbind(actuals = testing_data$Cholestoral,predicted = predicted_Cholestoral))
head(actuals_predictions)

# make actuals_predicted dataframe for sqrt(Murder)
actuals_predictions_sqrt <- data.frame(cbind(actuals = testing_data$Cholestoral,predicted = converted_Cholestoral_sqrt))
head(actuals_predictions_sqrt)

correlation_accuracy <- cor(actuals_predictions)
correlation_accuracy
# this shows us that the model has 22% accuracy

correlation_accuracy <- cor(actuals_predictions_sqrt)
correlation_accuracy
# this shows us that the model has 25% accuracy
# more accurate predicted results using sqrt

# Min - max accuracy
min_max_accuracy <- mean(apply(actuals_predictions, 1, min) / apply(actuals_predictions, 1, max))
min_max_accuracy
# 0.8617738

# Min - max accuracy
min_max_accuracy <- mean(apply(actuals_predictions_sqrt, 1, min) /apply(actuals_predictions_sqrt, 1, max))
min_max_accuracy
# 0.8622211

sigma(fit_model)/ mean(testing_data$Cholestoral)
# 0.1908885

sigma(fit_model_sqrt)/ mean(testing_data$Cholestoral)
# 0.006437358

summary(heart_data)
