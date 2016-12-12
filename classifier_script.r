## load ibraries

library(caret)
library(caTools)
library(randomForest)
library(e1071)
library(glmnet)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(wordcloud)
library(tm)
library(SnowballC)
library(ROCR)
library(pROC)
library(RColorBrewer)


## input

sms <- read.csv("spam.csv", stringsAsFactors = F)
str(sms)
sms$X <- NULL
sms$X.1 <- NULL
sms$X.2 <- NULL

names(sms) <- c("label","message")
levels(as.factor(sms$label))
sms$label[sms$label == "ham"] <- "non-spam"
sms$label[sms$label == "spam"] <- "spam"
sms$label <- factor(sms$label)

# cleaning text for analysis

bag <- Corpus(VectorSource(sms$message))

bag <- tm_map(bag, tolower)

bag <- tm_map(bag, PlainTextDocument)

bag <- tm_map(bag, removePunctuation)

bag <- tm_map(bag, removeWords, c(stopwords("english")))

bag <- tm_map(bag, stripWhitespace)

bag <- tm_map(bag, stemDocument)

# converting to data frame

frequencies <- DocumentTermMatrix(bag)

findFreqTerms(frequencies, lowfreq = 200) # to words that appear atleast 20 times

sparseWords <- removeSparseTerms(frequencies, 0.999) # words that appear in 99.5 percent tweets

sparseWords <- as.data.frame(as.matrix(sparseWords)) # converting the matrix of sparse words to data frame

colnames(sparseWords) <- make.names(colnames(sparseWords)) # rename column names to proper format in order to be used by R.

str(sparseWords)

sparseWords$label <- sms$label

# predicting whether message is spam/non-spam

## splitting data into train and test in 75:25 ratio.

set.seed(987)
split <- sample.split(sparseWords$label, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)
test <- subset(sparseWords, split == F)

## baseline model

table(test$label)
print(paste("Predicting all messages as non-spam gives an accuracy of: ",
            100*round(table(test$label)[1]/nrow(test), 4), "%"))

## logistic regression

glm.model <- glm(label ~ ., data = train, family = "binomial")
glm.predict <- predict(glm.model, test, type = "response")

### ROC curve

glm.ROCR <- prediction(glm.predict, test$label)
glm.AUC <- as.numeric(performance(glm.ROCR,"auc")@y.values)
glm.AUC
glm.prediction <- prediction(abs(glm.predict), test$label)
glm.performance <- performance(glm.prediction,"tpr","fpr")
plot(glm.performance)

### selecting threshold = 0.9 for spam filtering
table(test$label, glm.predict > 0.9)
glm.accuracy.table <- as.data.frame(table(test$label, glm.predict > 0.9))
print(paste("logistic model accuracy:",
            100*round(((glm.accuracy.table$Freq[1]+glm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

# svm

svm.model <- svm(label ~ ., data = train, kernel = "linear", cost = 0.1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
table(test$label, svm.predict)
svm.accuracy.table <- as.data.frame(table(test$label, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

## rpart model

tree.model <- rpart(label ~ ., data = train, method = "class", minbucket = 35)
prp(tree.model) 
tree.predict <- predict(tree.model, test, type = "class")
table(test$label, tree.predict)
rpart.accuracy.table <- as.data.frame(table(test$label, tree.predict))
print(paste("rpart (decision tree) accuracy:",
            100*round(((rpart.accuracy.table$Freq[1]+rpart.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

## glmnet

predictors.names <- names(train)[names(train) != "label"]
trainX <- as.matrix(train[, predictors.names])
trainY <- train$label
testX <- test[, predictors.names]

glmnet.model <- cv.glmnet(trainX, trainY, family = "binomial", alpha = 0)
glmnet.model
glmnet.model <- glmnet(trainX, trainY, family = "binomial", alpha = 0, lambda = 0.0152346)

glmnet.prediction <- predict(glmnet.model, as.matrix(testX), s = 0.0152346)

table(test$label, glmnet.prediction > 0.9)
glmnet.accuracy.table <- as.data.frame(table(test$label, glmnet.prediction > 0.5))
print(paste("glmnet accuracy:",
            100*round(((glmnet.accuracy.table$Freq[1]+glmnet.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))


# wordcloud

bag <- TermDocumentMatrix(bag)
bag <- as.matrix(bag)
bag <- sort(rowSums(bag), decreasing = T)
bag.df <- data.frame(word = names(bag), freq = bag)

set.seed(154)
str(bag)
wordcloud(words = bag.df$word, freq = bag.df$freq, min.freq = 100,
          max.words=1500, random.order=FALSE, rot.per=0.25,
          colors=brewer.pal(8, "Dark2"),
          scale = c(0.5,3))




