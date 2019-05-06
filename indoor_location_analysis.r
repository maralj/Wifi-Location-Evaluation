library(caret)
library(dplyr)
library(C50)
library(ggplot2)
library(doParallel)


# Find how many cores are on my machine
detectCores() # Result = Typically 4 to 6

# Create Cluster
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 
########## Understanding the dataset ########## 

summary(trainingData_2)
head(trainingData_2$LONGITUDE, 2)

# Mapping lat and long to visualize building shape and users' locations. 
ggplot(trainingData_2, aes(LONGITUDE,  LATITUDE)) + geom_hex()  +  ggtitle("Visualizing user's locations")
  geom_hex()

ncol(trainingData_2)
head(trainingData_2$USERID, 5)
head(trainingData_2$PHONEID, 5)
str(trainingData_2)

########## Feature engineering => preprocessing and dropping the unnecessary columns ########## 

# Aliasing the dataset 
df <- trainingData_2
str(df)

# Removing zero var features
?nearZeroVar()
zeroVarData<-nearZeroVar(trainingData_2, saveMetrics = TRUE)
zeroVarData
zeroVarData1<-which(zeroVarData_2$zeroVar == T)
zeroVarData1

df <- trainingData_2[,-(zeroVarData1)]

# Removing unsued variables 
df$LATITUDE <- NULL
df$LONGITUDE <- NULL
# df$USERID <- NULL
# df$PHONEID<-NULL
# df$TIMESTAMP<-NULL

# set data types
df$FLOOR<-as.factor(df$FLOOR)
df$BUILDINGID<-as.factor(df$BUILDINGID)
df$SPACEID<-as.factor(df$SPACEID)
df$RELATIVEPOSITION<-as.factor(df$RELATIVEPOSITION)

# Testing
str(df$SPACEID)

########## subsetting the dataset to improve computational efficiency ##########  
building_0 <- subset(df, df$BUILDINGID == 0)
building_1 <- subset(df, df$BUILDINGID == 1)
building_2 <- subset(df, df$BUILDINGID == 2)

#unite cols#
library(tidyr)

bldg0_loc <- unite(building_0, "LOCATION", c("FLOOR","BUILDINGID","SPACEID","RELATIVEPOSITION"))
bldg0_loc$LOCATION<-as.factor(bldg0_loc$LOCATION)
bldg0_loc$LOCATION = as.factor(bldg0_loc$LOCATION)

bldg1_loc <- unite(building_1, "LOCATION", c("FLOOR","BUILDINGID","SPACEID","RELATIVEPOSITION"))
bldg1_loc$LOCATION<-as.factor(bldg1_loc$LOCATION)
str(bldg1_loc$LOCATION)

bldg2_loc <- unite(building_2, "LOCATION", c("FLOOR","BUILDINGID","SPACEID","RELATIVEPOSITION"))
bldg2_loc$LOCATION<-as.factor(bldg2_loc$LOCATION)
str(bldg2_loc$LOCATION)

# moving location to the first column 
ncol(df)
nrow(df)
which(colnames(bldg0_loc) == "LOCATION")
which(colnames(bldg1_loc) == "LOCATION")
which(colnames(bldg2_loc) == "LOCATION")

# bldg0_loc <- bldg0_loc[c(468, 1: 468)]
# bldg1_loc <- bldg1_loc[c(468, 1: 468)]
# bldg2_loc <- bldg2_loc[c(468, 1: 468)]

#Model Buidling
library(caret)
set.seed(123)

############################################ Data Modeling ############################################ 

nrow(building_0)
str(bldg1_loc)

# # #create a 20% sample of the data
# build0_trainSize <- round(nrow(bldg0_loc) * 0.2) # 20%
# build1_trainSize <- round(nrow(bldg1_loc) * 0.2) # 20%
# build2_trainSize <- round(nrow(bldg2_loc) * 0.2) # 20%
# 
# trainSample_0 <- bldg0_loc[sample(1:nrow(bldg0_loc), build0_trainSize,replace=FALSE),]
# trainSample_1 <- bldg1_loc[sample(1:nrow(bldg1_loc), build1_trainSize,replace=FALSE),]
# trainSample_2 <- bldg2_loc[sample(1:nrow(bldg2_loc), build2_trainSize,replace=FALSE),]

bldg0_inTraining <- createDataPartition(bldg0_loc$LOCATION, p = .75, list = FALSE)
bldg0_training <- bldg0_loc[bldg0_inTraining,]
bldg0_testing <- bldg0_loc[-bldg0_inTraining,]

bldg1_inTraining <- createDataPartition(bldg1_loc$LOCATION, p = .75, list = FALSE)
bldg1_training <- bldg1_loc[bldg1_inTraining,]
bldg1_testing <- bldg1_loc[-bldg1_inTraining,]

bldg2_inTraining <- createDataPartition(bldg2_loc$LOCATION, p = .75, list = FALSE)
bldg2_training <- bldg2_loc[bldg2_inTraining,]
bldg2_testing <- bldg2_loc[-bldg2_inTraining,]

# cross validation 
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

############################################ C5.0 ############################################ 

DT_0 <- system.time(dt_c50_0   <- train(LOCATION~., data = bldg0_training, method = 'C5.0', trControl=fitControl)) # X 
DT_1 <- system.time(dt_c50_1 <- train(LOCATION~., data = bldg1_training, method = 'C5.0', trControl=fitControl)) # X 
DT_2 <- system.time(dt_c50_2 <- train(LOCATION~., data = bldg2_training, method = 'C5.0', trControl=fitControl)) # X 

str(bldg0_training$location)
summary(dt_c50_0)
summary(dt_c50_1)

prediction0 <- predict(dt_c50_0, bldg0_testing)
prediction1 <- predict(dt_c50_1, bldg1_testing)
prediction2 <- predict(dt_c50_2, bldg2_testing)

############################################################################################ 

mat1 = confusionMatrix(prediction0, bldg0_testing$LOCATION)
postResample(prediction0, bldg0_testing$LOCATION)
postResample(prediction1, bldg1_testing$LOCATION)
postResample(prediction2, bldg2_testing$LOCATION)

############################################ SVM ############################################ 
svm0 <- system.time(SVMFit0 <- train(LOCATION~., data = bldg0_training, method = "svmRadialWeights", trControl=fitControl))
svm1 <- system.time(SVMFit1 <- train(LOCATION~., data = bldg0_training, method = "svmRadialWeights", trControl=fitControl))
svm2 <- system.time(SVMFit2 <- train(LOCATION~., data = bldg0_training, method = "svmRadialWeights", trControl=fitControl))

############################################ rf ############################################ 

rf0 <- system.time(rf_0 <- train(LOCATION~., data = bldg0_training, method = 'rf', trControl=fitControl)) # X 

rf1 <- system.time(rf_1 <- train(LOCATION~., data = bldg1_training, method = 'rf', trControl=fitControl)) # x
trainSample_2$location <- droplevels(trainSample_2$location)
rf2 <- system.time(rf_2 <- train(LOCATION~., data = bldg2_training, method = 'rf', trControl=fitControl)) # x

rf_prediction0 <- predict(rf_0, bldg0_testing)
rf_prediction1 <- predict(rf_1, bldg1_testing)
rf_prediction2 <- predict(rf_2, bldg2_testing)

postResample(rf_prediction0, bldg0_testing$LOCATION)
postResample(rf_prediction1, bldg1_testing$LOCATION)
postResample(rf_prediction2, bldg2_testing$LOCATION)
head(bldg1_training,1)
############################################################################################ 

