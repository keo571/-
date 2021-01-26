rm(list = ls())
library(kernlab)
set.seed(0)

# load the data
myData = read.csv("final_x_train_with_y.csv", header = T)
testData = read.csv("final_x_test.csv", header = T)
x = myData[,1:7]
y = myData[,8]

# see the first few rows
head(myData)

svp = ksvm(as.matrix(x),as.factor(y),type='C-svc',kernel='vanilladot',C=100,
         scaled=T)
ypred = predict(svp, x)
table(y, ypred)
sum(ypred==y)/length(y)

# rbfdot
svp2 = ksvm(as.matrix(x), as.factor(y), type = "C-svc",
                kernel = "rbfdot", C = 100, scaled = T)
ypred = predict(svp2, x)
table(y, ypred)
sum(ypred==y)/length(y)

# result for testData
test_ypred = predict(svp2, testData)
write.csv(test_ypred,"test_pred.csv", row.names=FALSE)


