## import the data
train = read.csv("W23P1_train.csv")
test = read.csv("W23P1_test.csv")


##### 6 predictors
sapply(train, class)
" fare_amount is target variable
  pickup_datetime is char
  pickup_longitude is num
  pickup_latitude is num
  dropoff_longitude is num
  dropoff_latitude is num
  passenger_count is int
"

## load required libraries
library(reshape2)
library(ggplot2)
library(tidyverse)
library(randomForest)
library(lubridate)
library(geosphere)
library(caret)
library(glmnet)
library(gbm)
library(xgboost)
library(janitor)
library(ranger)
library(mlr)
library(tuneRanger)
library(MASS)
library(dplyr)

## confirming that all data is January 2013
month(train$pickup_datetime, label=TRUE)
month(test$pickup_datetime, label=TRUE)
year(train$pickup_datetime, label=TRUE)
year(test$pickup_datetime, label=TRUE)

##################################################### data transformation ######

## Treat date/time var into separate columns
train$daynum = mday(train$pickup_datetime)
train$day = wday(train$pickup_datetime, label=TRUE)
train$hour = hour(train$pickup_datetime)
train$min = minute(train$pickup_datetime)
train$sec = second(train$pickup_datetime)

test$day = wday(test$pickup_datetime, label=TRUE)
test$daynum = mday(test$pickup_datetime)
test$hour = hour(test$pickup_datetime)
test$min = minute(test$pickup_datetime)
test$sec = second(test$pickup_datetime)

## create a new column,  ## 1 if weekend, 0 if weekday
train$weekend = ifelse(train$day == "Sat"| train$day == "Sun", 1, 0)
test$weekend = ifelse(test$day == "Sat"| test$day == "Sun", 1, 0)

ggplot(train, aes(x=rush, y=fare_amount )) +
  geom_point()

## create a new column separating times -- categorical variables
## rush hour is between 4-9 and 15 - 19 

train$rush=ifelse((train$hour>=4 & train$hour<= 9) | 
                    (train$hour>= 15 & train$hour<= 19), 1, 0)
test$rush=ifelse((test$hour>=4 & test$hour<= 9) | 
                    (test$hour>= 15 & test$hour<= 19), 1, 0)

## calculate distance between to points using haversine formula
dist = rep(0,35000)
for (i in 1:35000) {
  dist[i] = distm(c(train$pickup_longitude[i], train$pickup_latitude[i]), 
                  c(train$dropoff_longitude[i], train$dropoff_latitude[i]), 
                  fun = distHaversine)
}
train$dist = dist

dist1 = rep(0,35000)
for (i in 1:35000) {
  dist1[i] = distm(c(test$pickup_longitude[i], test$pickup_latitude[i]), 
                  c(test$dropoff_longitude[i], test$dropoff_latitude[i]), 
                  fun = distHaversine)
}
test$dist = dist1

ggplot(train, aes(x=dist, y=fare_amount )) +
  geom_point() ## strong correlation!!

## calculate bearing between two points -- north, south east or west?
bear = rep(0,35000)
for (i in 1:35000) {
  bear[i] = bearing(c(train$pickup_longitude[i], train$pickup_latitude[i]), 
                  c(train$dropoff_longitude[i], train$dropoff_latitude[i]))
}
train$bear = bear
ggplot(train, aes(x=bear, y=fare_amount )) +
  geom_point()

bear1 = rep(0,35000)
for (i in 1:35000) {
  bear1[i] = bearing(c(test$pickup_longitude[i], test$pickup_latitude[i]), 
                   c(test$dropoff_longitude[i], test$dropoff_latitude[i]))
}
test$bear = bear1

ggplot(train, aes(x=bear, y=fare_amount )) +
  geom_point() ## strong correlation!!

#### create new test/train files with all new variables so do not have to rerun all the code again ##
write.csv(train, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\train1.csv",
          row.names=TRUE)
write.csv(test, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\test1.csv",
          row.names=TRUE)

train = read.csv("train1.csv")
test = read.csv("test1.csv")

#### based on pickup location, create new variables if pickup is penn station or one of the airports ##
## let's create a flag variable and see if anything useful

gpspenn = c(-73.99142, 40.75016)
gpsgua = c(-73.5230, 40.4630)
gpsjfk = c(-73.4644, 40.3823)

penndist = rep(0,35000)
for (i in 1:35000) {
  penndist[i] = distHaversine(c(train$pickup_longitude[i], train$pickup_latitude[i]), 
                   gpspenn)
}

lgadist = rep(0,35000)
for (i in 1:35000) {
  lgadist[i] = distHaversine(c(train$pickup_longitude[i], train$pickup_latitude[i]), 
                              gpsgua)
}

jfkdist = rep(0,35000)
for (i in 1:35000) {
  jfkdist[i] = distHaversine(c(train$pickup_longitude[i], train$pickup_latitude[i]), 
                             gpsjfk)
}

train$pennflag = ifelse(penndist < 200, 1, 0)
train$airport = ifelse(lgadist < 200 | jfkdist < 200, 1, 0)

train %>%
  ggplot(aes(x = pennflag, y = fare_amount)) +
  geom_point()

#### airports for test
penndist1 = rep(0,35000)
for (i in 1:35000) {
  penndist1[i] = distHaversine(c(test$pickup_longitude[i], test$pickup_latitude[i]), 
                              gpspenn)
}

lgadist1 = rep(0,35000)
for (i in 1:35000) {
  lgadist1[i] = distHaversine(c(test$pickup_longitude[i], test$pickup_latitude[i]), 
                             gpsgua)
}

jfkdist1 = rep(0,35000)
for (i in 1:35000) {
  jfkdist1[i] = distHaversine(c(test$pickup_longitude[i], test$pickup_latitude[i]), 
                             gpsjfk)
}

test$pennflag = ifelse(penndist1 < 200, 1, 0)
test$airport = ifelse(lgadist1 < 200 | jfkdist1 < 200, 1, 0)


###################################################### data visualization ######

train %>%
  ggplot(aes(fare_amount)) +
  geom_histogram(fill = "blue", binwidth = 2) 

b= boxplot(train$fare_amount, horizontal = TRUE)


ggplot(train, aes(x=hour, y=fare_amount )) +
  geom_point()

# rides per date
train %>%
  ggplot(aes(x=day)) +
  geom_bar()

ggplot(train, aes(x=dropoff_longitude, y=dropoff_latitude )) +
  geom_point()

ggplot(train, aes(x=pickup_longitude, y = pickup_latitude )) +
  geom_point()

###########################################  exploratory data analysis ########
## which hour has the highest average trip fare?
mean_by_hour = train %>%
  group_by(hour) %>%
  summarize(averaged.fare = mean(fare_amount))
plot(mean_by_hour, type="h")

ggplot(train, aes(x=hour, y=fare_amount )) +
  geom_point()

## which weekdays has the highest average trip fare
mean_by_day = train %>%
  group_by(day) %>%
  summarize(averaged.fare = mean(fare_amount))
plot(mean_by_day)

## passenger effect on fare
ggplot(train, aes(x=passenger_count, y=fare_amount )) +
  geom_point()
mean_by_pass = train %>%
  group_by(passenger_count) %>%
  summarize(averaged.fare = mean(fare_amount))
plot(mean_by_pass, type="h")

## passenger effect on fare lin reg
slr = lm(fare_amount~passenger_count, data = train)
summary(slr)

## most common pickup location?
xy = select(train, pickup_longitude, pickup_latitude)
res <- table(do.call(paste, xy))
res[which.max(res)]
xy[which.max(ave(seq(res), res, FUN = length)), ]


###################################### run a simple linear regression ###########

## best result slr
model1=lm(fare_amount~pickup_longitude + pickup_latitude + dropoff_longitude
          + dropoff_latitude + passenger_count + day + daynum + hour + min + sec
          + weekend + rush + dist + bear + pennflag + airport, data=train)

model11 = step(model1, direction="both", k=2, trace = 0)
test11 = predict(model11, newdata = test)

preds = cbind(test$uid, test11)

write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub11.csv",
          row.names=TRUE)

## score 4.19758

#####################################################  random forest ###########

##  make simple random forest
model3=randomForest(fare_amount~., data = train, importance=TRUE, ntree=50)
rf.pred=predict(model3, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub3.csv",
          row.names=TRUE)
## better
model7=randomForest(fare_amount~pickup_longitude + pickup_latitude + dropoff_longitude
                    + dropoff_latitude + passenger_count + day + daynum + hour + min + sec
                    + weekend + rush + dist + bear
                    , data = train, importance=TRUE, ntree=100)
rf.pred=predict(model7, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub7.csv",
          row.names=TRUE)

##  only use cov from the step reg  ##### 

model8=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                    + dropoff_latitude + passenger_count + day + daynum + hour + sec
                     + rush + dist + bear
                    , data = train, importance=TRUE, ntree=100)
rf.pred=predict(model8, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub8.csv",
          row.names=TRUE)


## try with only vars up to pickup_longitude ############# best result so far
model10=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                    + dropoff_latitude + hour 
                     + dist + bear
                    , data = train, importance=TRUE, ntree=100)

rf.pred=predict(model10, test)


preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub10.csv",
          row.names=TRUE)

### try more trees 
model11=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                     + dropoff_latitude + hour 
                     + dist + bear
                     , data = train, importance=TRUE, ntree=200)

rf.pred=predict(model11, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub30.csv",
          row.names=TRUE)

### try even more trees ## 500 most optimal ## 2nd best so far
model12=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                     + dropoff_latitude + hour 
                     + dist + bear
                     , data = train, importance=TRUE, ntree=500)

rf.pred=predict(model12, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub31.csv",
          row.names=TRUE)

#########################
## try with new airports variables -- not better

model1=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                     + dropoff_latitude + hour 
                     + dist + bear 
                     , data = train, importance=TRUE, ntree=500)

rf.pred=predict(model14, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub33.csv",
          row.names=TRUE)

## redid stepwise with new flag variables -- try new stepwise results
model41=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                    + dropoff_latitude + hour 
                    + dist + bear + sec + daynum + pennflag + rush 
                    , data = train, importance=TRUE, ntree=500)

fare_amount=predict(model41, test)
preds = cbind(test$uid, fare_amount)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub41.csv",
          row.names=TRUE)

##### best rf so far -- add day forgot to add it  ##################
model42=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                     + dropoff_latitude + hour 
                     + dist + bear + sec + daynum + pennflag + rush + day
                     , data = train, importance=TRUE, ntree=500)

fare_amount=predict(model42, test)

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub42.csv",
          row.names=TRUE)

########################## let's tune this one #####################

importance(model42)
varImpPlot(model42)

ctrl = trainControl(method = "cv", number = 10)
rf.Grid = expand.grid(mtry = 3)

rf.cv.model=caret::train(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude
                  + dropoff_latitude + hour 
                  + dist + bear + sec + daynum + pennflag + rush + day, data = train, 
                  method="rf", 
                  trControl = ctrl,
                  tuneGrid = rf.Grid)

rf.cv.model ## optimal mtry is 5 already

############### try even more trees ## not better, probably overfitting
model13=randomForest(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                     + dropoff_latitude + hour 
                     + dist + bear
                     , data = train, importance=TRUE, ntree=1000)

###### want to try tuning random forest
## result is worse, tuning not more effective ######################################
train_nouid = train[-1]

randomss = runif(1000,min=1, max = 35000)
train_x1 = as.data.frame(train_x1)
train_x1$fare_amount = train$fare_amount
tune.task = makeRegrTask(data = train_x1[randomss,], target = "fare_amount")
estimateTimeTuneRanger(tune.task)

res=tuneRanger(tune.task)

model13=ranger(fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
               + dropoff_latitude + hour 
               + dist + bear
               , data = train, min.node.size = 2, mtry = 1)
rf.pred=predict(model13, test)$predictions

preds = cbind(test$uid, rf.pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub13.csv",
          row.names=TRUE)

######################################### try trees ########################
## need day to be numerical
train$dayfac = train$day
train$dayfac<-c(Mon=1,Tue=2, Wed=3, Thu=4, Fri=5,Sat=6,Sun=7)[train$dayfac]
test$dayfac = test$day
test$dayfac<-c(Mon=1,Tue=2, Wed=3, Thu=4, Fri=5,Sat=6,Sun=7)[test$dayfac]

boost.tree = gbm(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude
                 + dropoff_latitude + hour 
                 + dist + bear + sec + daynum + pennflag + rush + dayfac
                 , data = train, n.trees = 500, interaction.depth=3, cv.folds = 5)

cv.num = gbm.perf(boost.tree) ## 255

## result is not better than random forests
yhat.boost.cv=predict(boost.tree,newdata=test,n.trees=cv.num)
preds = cbind(test$uid, yhat.boost.cv)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub52t.csv",
          row.names=TRUE)

## var ipmortance
summary(boost.tree, cBars = 10,
        method = relative.influence,
        las = 2)

#################################################### to tune, try xgboost #####
## split into test and train
set.seed(123)
train_id = sample(1:nrow(train), size = floor(0.75 * nrow(train)), replace = FALSE)

ntrain = train[train_id,]
ntest = train[-train_id,]

basetrain = dplyr::select(ntrain, -uid, -fare_amount)
ntest1 = dplyr::select(ntest, -uid, -fare_amount)

test_nouid = dplyr::select(test, -uid)

xgtrain = xgb.DMatrix(data = data.matrix(basetrain), label = ntrain$fare_amount)
xgtest = xgb.DMatrix(data = data.matrix(test_nouid))
xgvalid = xgb.DMatrix(data = data.matrix(ntest1), label = ntest$fare_amount) 
watchlist = list(traindata=xgtrain, val=xgvalid)

params = list(booster = "gbtree", eta = 0.3, gamma = 0, max_depth = 6, min_child_weight = 1, subsample = 0.7,
              colsample_bytree = 0.6, eval_metrix = "rmsle", seed = 123)

xgbcv = xgb.cv ( params = params, data = xgtrain, nrounds = 200, nfold =5, showsd = T, stratified = T,
                 print_every_n=10, early_stopping_rounds = 20, maximize = F)

xgb = xgb.train(params = params, nrounds = 30, data = xgtrain,  
                print_every_n = 10, watchlist = watchlist)
              
pred = predict(xgb, xgtest)
preds = cbind(test$uid, pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\sub1xgt.csv",
          row.names=TRUE)

### let's try to tune this one
## takes too long to successfully train
train_control = trainControl(method="cv", number = 5, search = "grid")

gbmGrid = expand.grid(max_depth = c(3,5,7),
                      nrounds = (1:5),
                      eta = 0.3,
                      gamma = 0,
                      subsample=1,
                      min_child_weight = 1,
                      colsample_bytree = 0.6
                      )

modelxgb10 = caret::train(fare_amount ~ ., data = train[train_id,], method = "xgbTree", trControl = train_control) 
pred_y = predict(modelxgb10, test)

preds = cbind(test$uid, pred)
write.csv(preds, "C:\\Users\\cream\\OneDrive - Queen's University\\Desktop\\STAT457\\Project 1\\1partxg.csv",
          row.names=TRUE)

## takes way too long to train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5)

gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.1, 0.05),
                       n.minobsinnode = 10)

ntrainmat = as.matrix(ntrain)
gbm.cv.model <- caret::train(fare_amount~fare_amount~ pickup_longitude + pickup_latitude + dropoff_longitude
                             + dropoff_latitude + hour 
                             + dist + bear + sec + daynum + pennflag + rush + day, data = ntrainmat,
                      method = "gbm",
                      trControl = ctrl,
                      tuneGrid = gbm.Grid,
                      verbose = FALSE)



