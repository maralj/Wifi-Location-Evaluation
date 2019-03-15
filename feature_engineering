library(caret)
library(dplyr)
library(C50)

########## Understanding the dataset ########## 
summary(trainingData)
ncol(trainingData)
head(trainingData$USERID, 5)
head(trainingData$PHONEID, 5)
str(trainingData)

########## Feature engineering => preprocessing and dropping the unnecessary columns ########## 

# Aliasing the dataset 
df <- trainingData
str(df)

# removing zero var fetures
?nearZeroVar()
zeroVarData<-nearZeroVar(trainingData, saveMetrics = TRUE)
zeroVarData
zeroVarData1<-which(zeroVarData$zeroVar == T)
zeroVarData1

df <- trainingData[,-(zeroVarData1)]

# Removing unsued variables 
df$LATITUDE <- NULL
df$LONGITUDE <- NULL
df$USERID <- NULL
df$PHONEID<-NULL
df$TIMESTAMP<-NULL

# set datatypes
df$FLOOR<-as.factor(df$FLOOR)
df$BUILDINGID<-as.factor(df$BUILDINGID)
df$SPACEID<-as.factor(df$SPACEID)
df$RELATIVEPOSITION<-as.factor(df$RELATIVEPOSITION)
str(df)

########## subsetting the dataset by building ID to improve computational efficiency ##########  
building_0 <- subset(df, df$BUILDINGID == 0)
building_1 <- subset(df, df$BUILDINGID == 1)
building_2 <- subset(df, df$BUILDINGID == 2)

# unite cols
library(tidyr)

bldg0_loc <- unite(building_0, "LOCATION", c("FLOOR","BUILDINGID","SPACEID","RELATIVEPOSITION"))
bldg0_loc$LOCATION<-as.factor(bldg0_loc$LOCATION)
str(bldg0_loc$LOCATION)

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

#bldg0_loc <- bldg0_loc[c(466, 1: 466)]
#bldg1_loc <- bldg1_loc[c(466, 1: 466)]
#bldg2_loc <- bldg2_loc[c(466, 1: 466)]

#Model Buidling
library(caret)

############################################ Data Modeling ############################################ 

set.seed(123)
nrow(building_0)
str(bldg1_loc)

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

DT_0 <- system.time(dt_c50 <- train(LOCATION~., data = bldg0_training, method = 'C5.0', trControl=fitControl)) # X 
DT_1 <- system.time(dt_c50_1 <- train(LOCATION~., data = bldg0_training, method = 'C5.0', trControl=fitControl)) # X 
DT_2 <- system.time(dt_c50_2 <- train(LOCATION~., data = bldg0_training, method = 'C5.0', trControl=fitControl)) # X 

# Reviewing Models 
dt_c50
dt_c50_1
dt_c50_2

