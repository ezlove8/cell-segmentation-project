library(AppliedPredictiveModeling)
options(stringsAsFactors = F)
data("segmentationOriginal") ##loading the data from package
data = segmentationOriginal

uniq.len = apply(data,2,function(x) length(unique(x)))
which(uniq.len == 1)
data = data[,-which(uniq.len == 1)]

#split into categorical and continuous
cat.data = data[,c(1,2,3,grep("Status",colnames(data)),grep("Centroid",colnames(data)))]#categorical features+sexChr
cont.data = data[,-grep("Status",colnames(data))]#continous features+sexChr


##data can be categorical/continuous/both
data = cont.data
data.train = data[which(data$Case=="Train"),-c(1,2)]
data.test = data[which(data$Case=="Test"),-c(1,2)]

feature.pval = c()
for (i in 2:ncol(data.train)) {
  feature.pval[i-1] = t.test(data.train[which(data.train$Class=="WS"),i],data.train[which(data.train$Class=="PS"),i])$p.value
}
names(feature.pval) = colnames(data.train)[2:ncol(data.train)]
sort.feature = names(feature.pval)[order(feature.pval,decreasing = F)]

##Cross Validation determine feature numbers
library(MASS)
library(caret)
library(e1071)
library(randomForest)
flds <- createFolds(2:nrow(data.train), k = 5, list = TRUE, returnTrain = FALSE)
top.feature = sort.feature
res.lda = res.qda = res.svm.linear = res.svm.radial = res.rf = list()
for (feature.num in 1:length(top.feature)) {
  res.lda[[as.character(feature.num)]]=res.qda[[as.character(feature.num)]]=res.svm.linear[[as.character(feature.num)]]=res.svm.radial[[as.character(feature.num)]]=res.rf[[as.character(feature.num)]]=list()
}

set.seed(15213)
for(i in 1:5){
  index.test<-flds[[i]]
  index.train<-unlist(flds)[-index.test]
  sub.train<-data.train[index.train,]
  sub.test<-data.train[index.test,]
  for (j in 1:(ncol(data.train)-1)) {
    print(paste("top feature",j))
    sub.train.feature = sub.train[,c('Class',top.feature[1:j])]
    sub.test.feature = sub.test[,c('Class',top.feature[1:j])]
    
    #LDA
    lda.fit=lda(Class~.,data=sub.train.feature)
    lda.pred=predict(lda.fit, sub.test.feature)$`class`
    cm = confusionMatrix(factor(as.character(lda.pred)),factor(as.character(sub.test.feature$Class)))
    res.lda[[as.character(j)]][["ACC"]] = c(res.lda[[as.character(j)]][["ACC"]],cm$overall[['Accuracy']])
    res.lda[[as.character(j)]][["Youden"]] = c(res.lda[[as.character(j)]][["Youden"]],cm$byClass[['Specificity']]+cm$byClass[['Sensitivity']]-1)
     
      
    #SVM:linear kernel
    svm.fit.linear=svm(Class~.,sub.train.feature,kernel = "linear")
    svm.pred.linear=predict(svm.fit.linear,newdata=sub.test.feature)
    cm = confusionMatrix(factor(as.character(svm.pred.linear)),factor(as.character(sub.test.feature$Class)))
    res.svm.linear[[as.character(j)]][["ACC"]] = c(res.svm.linear[[as.character(j)]][["ACC"]],cm$overall[['Accuracy']])
    res.svm.linear[[as.character(j)]][["Youden"]] = c(res.svm.linear[[as.character(j)]][["Youden"]],cm$byClass[['Specificity']]+cm$byClass[['Sensitivity']]-1)
    
    #SVM:radial kernel
    svm.fit.radial=svm(Class~.,sub.train.feature,kernel = "radial")
    svm.pred.radial=predict(svm.fit.radial, sub.test.feature)
    cm = confusionMatrix(factor(as.character(svm.pred.radial)),factor(as.character(sub.test.feature$Class)))
    res.svm.radial[[as.character(j)]][["ACC"]] = c(res.svm.radial[[as.character(j)]][["ACC"]],cm$overall[['Accuracy']])
    res.svm.radial[[as.character(j)]][["Youden"]] = c(res.svm.radial[[as.character(j)]][["Youden"]],cm$byClass[['Specificity']]+cm$byClass[['Sensitivity']]-1)
    
    #rf.fit=randomForest(Class~.,data=sub.train.feature,importance=TRUE)
    #rf.pred = predict(rf.fit,newdata=sub.test.feature)
    #cm = confusionMatrix(factor(as.character(rf.pred)),factor(as.character(sub.test.feature$Class)))
    #res.rf[[as.character(j)]][["ACC"]] = c(res.rf[[as.character(j)]][["ACC"]],cm$overall[['Accuracy']])
    #res.rf[[as.character(j)]][["Youden"]] = c(res.rf[[as.character(j)]][["Youden"]],cm$byClass[['Specificity']]+cm$byClass[['Sensitivity']]-1)
    
  }
}

rf.fit=randomForest(Class~.,data=data.train,importance=TRUE)
important.features = as.data.frame(rf.fit$importance[,c("MeanDecreaseAccuracy", "MeanDecreaseGini")])
sort.important.features = important.features[order(important.features$MeanDecreaseGini,decreasing = T),]
rf.pred = predict(rf.fit,newdata=data.test)
cm = confusionMatrix(factor(as.character(rf.pred)),factor(as.character(data.test$Class)))
cm$byClass[["Sensitivity"]]+cm$byClass[["Specificity"]]-1

#ACC results collect
lda.mean = sapply(res.lda,function(x) mean(x[[1]]))
lda.sd = sapply(res.lda,function(x) sd(x[[1]]))
thres = lda.mean[which.max(lda.mean)]-lda.sd[which.max(lda.mean)]
lda.mean[which(lda.mean>thres)[1]]
lda.mean[which.max(lda.mean)]


lda.df = data.frame(meanACC = lda.mean, sdACC = lda.sd, feature.num = as.numeric(names(lda.mean)))
ggplot(lda.df, aes(x=feature.num, y=meanACC)) + 
  geom_errorbar(aes(ymin=meanACC-sdACC, ymax=meanACC+sdACC), width=.1) +
  geom_line() +
  geom_point() +
  geom_point(data=lda.df, aes(x=59, y=lda.df$meanACC[59]), colour="green",size=3)+
  geom_segment(aes(x = 59, y = lda.df$meanACC[59]-lda.df$sdACC[59], xend = 59, yend = lda.df$meanACC[59]+lda.df$sdACC[59]),colour = "green")+
  geom_point(data=lda.df, aes(x=93, y=lda.df$meanACC[93]), colour="red", size=3)+
  geom_segment(aes(x = 93, y = lda.df$meanACC[93]-lda.df$sdACC[93], xend = 93, yend = lda.df$meanACC[93]+lda.df$sdACC[93]),colour = "red")+
  ggtitle("LDA")



svm.linear.mean = sapply(res.svm.linear,function(x) mean(x[[1]]))
svm.linear.sd = sapply(res.svm.linear,function(x) sd(x[[1]]))
thres = svm.linear.mean[which.max(svm.linear.mean)]-svm.linear.sd[which.max(svm.linear.mean)]
svm.linear.mean[which(svm.linear.mean>thres)[1]]
svm.linear.mean[which.max(svm.linear.mean)]


svm.linear.df = data.frame(meanACC = svm.linear.mean, sdACC = svm.linear.sd, feature.num = as.numeric(names(svm.linear.mean)))
ggplot(svm.linear.df, aes(x=feature.num, y=meanACC)) + 
  geom_errorbar(aes(ymin=meanACC-sdACC, ymax=meanACC+sdACC), width=.1) +
  geom_line() +
  geom_point() +
  geom_point(data=svm.linear.df, aes(x=73, y=svm.linear.df$meanACC[73]), colour="green",size=3)+
  geom_segment(aes(x = 73, y = svm.linear.df$meanACC[73]-svm.linear.df$sdACC[73], xend = 73, yend = svm.linear.df$meanACC[73]+svm.linear.df$sdACC[73]),colour = "green")+
  geom_point(data=svm.linear.df, aes(x=107, y=svm.linear.df$meanACC[107]), colour="red", size=3)+
  geom_segment(aes(x = 107, y = svm.linear.df$meanACC[107]-svm.linear.df$sdACC[107], xend = 107, yend = svm.linear.df$meanACC[107]+svm.linear.df$sdACC[107]),colour = "red")+
  ggtitle("svm.linear")


svm.radial.mean = sapply(res.svm.radial,function(x) mean(x[[1]]))
svm.radial.sd = sapply(res.svm.radial,function(x) sd(x[[1]]))
thres = svm.radial.mean[which.max(svm.radial.mean)]-svm.radial.sd[which.max(svm.radial.mean)]
svm.radial.mean[which(svm.radial.mean>thres)[1]]
svm.radial.mean[which.max(svm.radial.mean)]

svm.radial.df = data.frame(meanACC = svm.radial.mean, sdACC = svm.radial.sd, feature.num = as.numeric(names(svm.radial.mean)))
ggplot(svm.radial.df, aes(x=feature.num, y=meanACC)) + 
  geom_errorbar(aes(ymin=meanACC-sdACC, ymax=meanACC+sdACC), width=.1) +
  geom_line() +
  geom_point() +
  geom_point(data=svm.radial.df, aes(x=61, y=svm.radial.df$meanACC[61]), colour="green",size=3)+
  geom_segment(aes(x = 61, y = svm.radial.df$meanACC[61]-svm.radial.df$sdACC[61], xend = 61, yend = svm.radial.df$meanACC[61]+svm.radial.df$sdACC[61]),colour = "green")+
  geom_point(data=svm.radial.df, aes(x=82, y=svm.radial.df$meanACC[82]), colour="red", size=3)+
  geom_segment(aes(x = 82, y = svm.radial.df$meanACC[82]-svm.radial.df$sdACC[82], xend = 82, yend = svm.radial.df$meanACC[82]+svm.radial.df$sdACC[82]),colour = "red")+
  ggtitle("svm.radial")

#Youden results collect
lda.mean = sapply(res.lda,function(x) mean(x[[2]]))
lda.sd = sapply(res.lda,function(x) sd(x[[2]]))
thres = lda.mean[which.max(lda.mean)]-lda.sd[which.max(lda.mean)]
lda.mean[which(lda.mean>thres)[1]]
lda.mean[which.max(lda.mean)]


svm.linear.mean = sapply(res.svm.linear,function(x) mean(x[[2]]))
svm.linear.sd = sapply(res.svm.linear,function(x) sd(x[[2]]))
thres = svm.linear.mean[which.max(svm.linear.mean)]-svm.linear.sd[which.max(svm.linear.mean)]
svm.linear.mean[which(svm.linear.mean>thres)[1]]
svm.linear.mean[which.max(svm.linear.mean)]


svm.radial.mean = sapply(res.svm.radial,function(x) mean(x[[2]]))
svm.radial.sd = sapply(res.svm.radial,function(x) sd(x[[2]]))
thres = svm.radial.mean[which.max(svm.radial.mean)]-svm.radial.sd[which.max(svm.radial.mean)]
svm.radial.mean[which(svm.radial.mean>thres)[1]]
svm.radial.mean[which.max(svm.radial.mean)]

#Testing
#LDA
train = data.train[,c('Class',top.feature[1:66])]#k from CV
test = data.test[,c('Class',top.feature[1:66])]
lda.fit=lda(Class~.,data=train)
lda.pred=predict(lda.fit, test)
cm = confusionMatrix(lda.pred$`class`,test$Class)
cm[["byClass"]][["Specificity"]]+cm[["byClass"]][["Sensitivity"]]-1
auc = auc(as.factor(test$Class), as.numeric(lda.pred$x))#????????????

#SVM.linear
train = data.train[,c('Class',top.feature[1:74])]#k from CV
test = data.test[,c('Class',top.feature[1:74])]
svm.fit.linear=svm(Class~.,train,kernel = "linear")
svm.pred.linear=predict(svm.fit.linear, test)
cm = confusionMatrix(svm.pred.linear,test$Class)
cm[["byClass"]][["Specificity"]]+cm[["byClass"]][["Sensitivity"]]-1

#SVM.radial
train = data.train[,c('Class',top.feature[1:71])]#k from CV
test = data.test[,c('Class',top.feature[1:71])]
svm.fit.radial=svm(Class~.,train,kernel = "radial")
svm.pred.radial=predict(svm.fit.radial, test)
cm = confusionMatrix(svm.pred.radial,test$Class)
cm[["byClass"]][["Specificity"]]+cm[["byClass"]][["Sensitivity"]]-1

#RandomForest
train = data.train[,c('Class',top.feature[1:4])]#k from CV
test = data.test[,c('Class',top.feature[1:4])]
rf.fit=randomForest(Class~.,data=train,importance=TRUE)
rf.pred = predict(rf.fit,newdata=test)
cm = confusionMatrix(rf.pred,test$Class)
youden = cm[["byClass"]][["Specificity"]]+cm[["byClass"]][["Sensitivity"]]-1
