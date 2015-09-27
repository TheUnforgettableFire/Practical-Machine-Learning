# Practical Machine Learning Project: Predicting manner in which exercise was done
Rohan Jagdish Ashar  
September 27, 2015  

The report contains the following sections:

* Synopsis and Background
* Model Building
* Cross Validation
* Sample Error
* Prediction

Synopsis and Background
=======================

The **objective** of this report is to predict the manner in which six participants did the exercise. 

The data for this assignment come in the form of a comma-separated-value file. You can download the file from the course web site:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>.
The dataset includes data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Model Building
==============

The dataset contains many columns that have accelerometer data. We want to remove the columns with missing values and the columns that contain factors.

### Loading and preprocessing the data


```r
# Loading training and test dataset
traindata <- read.csv("pml-training.csv", na.strings = c("", "NA"))
testdata <- read.csv("pml-testing.csv", na.strings = c("", "NA"))

# Remove columns where there are NA or missing data
traindata <- traindata[, apply(traindata, 2, function(x) !any(is.na(x)))]

# Remove columns that are recording user information
traindata <- traindata[,-c(1:7)]
```

Before running any machine learning algorithm we want to remove the covariates that are correlated to each other.

### Covariate creation


```r
# Load the caret and Random Forest libraries
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
# Check for correlations between covariates
correlationMatrix <- cor(na.omit(traindata[sapply(traindata, is.numeric)]))

# identify covariates that show greater than 90% correlation
varCorrelated <- findCorrelation(correlationMatrix, cutoff = .90, verbose = TRUE)
```

```
## Compare row 10  and column  1 with corr  0.992 
##   Means:  0.27 vs 0.168 so flagging column 10 
## Compare row 1  and column  9 with corr  0.925 
##   Means:  0.25 vs 0.164 so flagging column 1 
## Compare row 9  and column  4 with corr  0.928 
##   Means:  0.233 vs 0.161 so flagging column 9 
## Compare row 8  and column  2 with corr  0.966 
##   Means:  0.245 vs 0.157 so flagging column 8 
## Compare row 19  and column  18 with corr  0.918 
##   Means:  0.091 vs 0.158 so flagging column 18 
## Compare row 46  and column  31 with corr  0.914 
##   Means:  0.101 vs 0.161 so flagging column 31 
## Compare row 46  and column  33 with corr  0.933 
##   Means:  0.083 vs 0.164 so flagging column 33 
## All correlations <= 0.9
```

```r
# Remove columns where there are NA or missing data
traindata <- traindata[,-varCorrelated]
```

Split the training data into training and testing data. We can test the accuracy of the model on the test dataset before applying to the 20 test observations.

### Create Training and Test dataset from the Training Data 


```r
# Split the data training = 70% and testing = 30% 
inTrain = createDataPartition(y = traindata$classe, p = 0.7, list = FALSE)

# Assign training and test data to separate data frames
smallTrain = traindata[inTrain, ]
smallTest = traindata[-inTrain, ]

# Check dimensions of the train and test datasets
dim(smallTrain); dim(smallTest)
```

```
## [1] 13737    46
```

```
## [1] 5885   46
```

I am going to use Random Forest algorithm to predict the classe (exercise) based on the accelerometer variables.

### Run Prediction Algorithm 


```r
# Run Random Forest Algorithm on Training Dataset
modFit <- randomForest(classe ~ ., data = smallTrain, mtry = 7)
```

Cross Validation
================

Cross validation will allow us to check for the accuracy of the prediction algorithm.


```r
# Check the prediction on the hold out test sample
pred <- predict(modFit,smallTest)

# Cross Validation Matrix to identify accuracy
smallTest$PredRight <- pred==smallTest$classe
predMatrix <- table(pred,smallTest$classe)
```

Sample Error
============


```r
# Calculate sample error on test dataset
sum(diag(predMatrix))/sum(as.vector(predMatrix))
```

```
## [1] 0.9954121
```

The accuracy of the model on the test sample is 99% which suggests that the Random Forest algorithm is doing a good job at predicting the class. Also, the final test error on the 20 observations may be less than 99%.

Now we can predict the 20 test observations using the model that gave us 99% error on the test data.

Also, the following prediction code creates the data to upload for the project assignment.

Prediction
==========


```r
# Calculate sample error on test dataset
predTest <- predict(modFit,testdata)

# Output the classe prediction for test dataset
predTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predTest)
```

