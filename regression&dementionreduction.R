##Load data
library('AppliedPredictiveModeling')
data("segmentationOriginal")
##Load Libraries
library(lasso2)####package which has prostate data
library(glmnet)#####package for lasso, ridge and glmnet
library(leaps)###Cp, BIC and adjusted Rsquare
library(gglasso)##group lasso
library(gplots)###draw heatmap
library("genlasso")####a generalized lasso package which can do fused lasso
library(SGL)####sparse group lasso

##Dimension Reduction##
## Equally divide the data into one training dataset and one testing dataset.
set.seed(101) 
sample <- sample.int(n = nrow(segmentationOriginal), size = floor(.5*nrow(segmentationOriginal)), replace = F)
train <- subset(segmentationOriginal, Case == "Train")
test  <- subset(segmentationOriginal, Case == "Test")

##Use ridge regression. Find the optimal lambda which will return the smallest cross validation error using the training data
#ridge regression
set.seed(101)
x <- train[,4:119]
y <- as.numeric(train[,3])
ridge_cv <- cv.glmnet(as.matrix(x), y, alpha = 0, nfolds = 10)
ridge_cv$lambda
#note that glmnet package will choose lambda sequence automatically, there is no need to manually
plot(ridge_cv)
#we use the minimum criterion for ridge
lambda_cv <- ridge_cv$lambda.min
lambda_cv
#optimal lambda is 0.071198
#building model ussing lambda
model_ridge <- glmnet(as.matrix(x), y, alpha = 0, lambda = lambda_cv, standardize = TRUE)
ridge_pred = predict(model_ridge, newx = as.matrix(test[,4:119]))
mean((ridge_pred - as.numeric(test[,3]))^2)
## MSE = 0.13889
sqrt(mean((ridge_pred - as.numeric(test[,3]))^2))
## RMSE = 0.0.3727 
############
###Lasso####
############
#(Part 1) Finding optimal lambda
lasso_cv <- cv.glmnet(as.matrix(x), y, alpha = 1, nfolds = 10)
plot(lasso_cv)
lasso_lambda_cv <- lasso_cv$lambda.min
lasso_lambda_cv #optimal lambda = 0.003889
#(Part 2) Build lasso model and test error in terms of RMSE
model_lasso <- glmnet(as.matrix(x), y, alpha = 1, lambda = lasso_lambda_cv, standardize = TRUE)
lasso_pred = predict(model_lasso, newx = as.matrix(test[,4:119]))
mean((lasso_pred - as.numeric(test[,3]))^2)
## MSE = 0.1394
sqrt(mean((lasso_pred - as.numeric(test[,3]))^2))
## RMSE = 0.3734
##############################################
####################
##Cluster Analysis##
####################
#dimension reduction
########### PCA
which(apply(x, 2, var)==0) #have zero variance: MemberAvgAvgIntenStatusCh2 MemberAvgTotalIntenStatusCh2 
mydata <- segmentationOriginal[,-c(76,77)]
x <- mydata[,4:117]
label<-mydata$Class
musum(is.na(x)) #0
dim(x) #2019 x 114
res<-prcomp(x, center = TRUE,scale.  = TRUE)
pca1<-res$x[,1]
pca2<-res$x[,2]
label <- as.factor(label)
colors = rainbow(length(unique(label)))
colors <- c("#00AFBB", "#FC4E07")
colors <- colors[as.numeric(iris$Species)]
names(colors) = unique(label)
par(mgp=c(2.5,1,0))
plot(pca1,pca2)
plot(pca1,pca2, main="PCA ", xlab="PCA dimension 1", ylab="PCA dimension 2", "cex.main"=2, "cex.lab"=1.5, col=colors)
legend("bottomleft", pch=c(1,1),legend = c('PS','WS'), col = c("red","turquoise"), bty = "o")
text(pca1,pca2, labels=label, col=colors[label],cex=2.5)

##############
#    MDS     #  ask about
##############

##############
#   tSNE     #
##############
###########TSNE
library(readr)
library(Rtsne)
tsne <- Rtsne(x, dims = 2, perplexity=10, verbose=TRUE, max_iter = 500)

label <- as.factor(label)
colors = rainbow(length(unique(label)))
names(colors) = unique(label)
par(mgp=c(2.5,1,0))
plot(tsne$Y, main="tSNE", xlab="tSNE dimension 1", ylab="tSNE dimension 2", "cex.main"=2, "cex.lab"=1.5, col =colors)
legend("bottomleft", pch=c(1,1),legend = c('PS','WS'), col = c("red","turquoise"), bty = "o")
text(tsne$Y, labels=label, col=colors[label],cex=2.5)


