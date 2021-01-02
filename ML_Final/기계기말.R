############# 기계 기말 프로젝트 ##############

### 데이터 불러오기 ###
who = read.csv("LifeExpectancy.csv", header=T, na.strings = "?")

### 데이터 탐색 ###
head(who) # 1행부터 6행까지의 데이터 확인 
dim(who) # 2938개의 관측값, 22개의 변수
names(who) # 22개의 변수명 확인
str(who) # 데이터의 구조 확인; Country, Status만 chr 
summary(who) # 데이터의 요약 통계량 확인

### 데이터 전처리 ###

# 1. Country; character -> 분석에서 제외
who = who[,-1] # 변수 제거
names(who) # 확인

# 2. Status; character -> factor

attach(who)
Status = as.factor(Status)
class(Status)

# 3. 결측값 처리

sum(is.na(who)) # 결측치 개수 확인
mean(is.na(who)) # 결측치가 꽤 많은 비중; 2938*22칸 중 2563개의 칸이 비어있음 
mean(!complete.cases(who)) # 적어도 한 개 이상의 결측값을 가지는 행; 거의 반 정도 결측치 포함 

# 따라서 결측값을 제거하기엔 데이터 손실이 너무 크므로 결측값을 대체하자
# 누락 데이터가 5% 이상인 관측이나 행은 버리고 나머지는 대체 by knn
# 종속변수는 예측해서 채워넣으면 예측력의 현실성이 떨어지므로 제외

library(DMwR)
who = knnImputation(who[, !names(who)%in% c("Life.expendency","Status")], k=10)
sum(is.na(who))

# 4. 데이터 스케일링 해야될까??
# 이 데이터의 경우 데이터를 스케일링하면 직관적인 해석이 어려울 듯 하여 굳이 표준화는 하지않기로 한다.

### 데이터 분석 ###


# 모수추정값, y 예측값, 모수 해석, 신뢰구간 해석
# 모형의 적합도 by RSE, R2

# 2. 다중선형회귀 by 변수선택

fit = lm(Life.expectancy~., data=who)
summary(fit)


### 부분집합선택; 최적부분집합선택법 ### 

set.seed(0401)
library(leaps) 
library(MASS)

fit.best = regsubsets(Life.expectancy~.,nvmax=20,data=who) 
best.summary = summary(fit.best)
summary(fit.best)

# 그래프 그리기

par(mfrow=c(2,2)) 

#1
plot(best.summary$rss,xlab="Number of Variables",ylab="RSS",type="l") #RSS; Residual Sum of Squares
#2
plot(best.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l") #Adjusted R^2
which.max(best.summary$adjr2) #Adjr2의 최대값; 변수가 8개일 때
points(16,best.summary$adjr2[16], col="red",cex=2,pch=20) #최대값 빨간점으로 찍기
#3
plot(best.summary$cp,xlab="Number of Variables",ylab="Cp",type='l') #Cp
which.min(best.summary$cp) #Cp의 최소값
points(16,best.summary$cp[16],col="red",cex=2,pch=20) #최소값 빨간점으로 찍기
#4
plot(best.summary$bic,xlab="Number of Variables",ylab="BIC",type='l') #BIC
which.min(best.summary$bic) #BIC의 최소값
points(13,best.summary$bic[13],col="red",cex=2,pch=20) #최소값 빨간점으로 찍기

#그래프 확인 결과, BIC가 다른 방법들보다 적은 변수 개수를 선택

#fitting된 결과로 계수 plot을 그려봄
par(mfrow=c(1,1))
plot(fit.best,scale="r2")
plot(fit.best,scale="adjr2")
plot(fit.best,scale="Cp")
plot(fit.best,scale="bic")

# 결론; 변수가 13개일 때 최소인 bic가 선택한 변수 확인
coef(fit.best,13) 
best.summary$rsq[13]

# 최적부분집합 선택법은 p가 클수록 계산이 어려워짐
# 따라서 전진, 후진, 계단선택법 등을 이용 가능 
# 특히 계단선택법을 살펴보자

### 부분집합선택; 계단선택법 ###

fit = lm(Life.expectancy~., data=who)
fit.con = lm(Life.expectancy~1, data=who)
fit.step = step(fit.con,scope=list(lower=fit.con,upper=fit),direction="both")
summary(fit.step)

### 예측을 위해 train, test data 반반씩 나누기 ###

# 축소추정법 들어가기 전 먼저 훈련과 시험데이터 반반씩 나누기
set.seed(1)
train.size = dim(who)[1]/2
train = sample(1:dim(who)[1], train.size)
test = -train
who.train = who[train, ]
who.test = who[test, ]

### 축소추정법; Linear ###

lm.fit = lm(Life.expectancy~., data=who.train)
lm.pred = predict(lm.fit, who.test)
mean((lm.pred-who.test$Life.expectancy)^2) 

### 축소추정법; Ridge ###

#fitting by ridge regression model
library(glmnet)

train.X = model.matrix(Life.expectancy~., data=who.train)
test.X = model.matrix(Life.expectancy~., data=who.test)

grid = 10^seq(4,-2,length=100)
ridge.fit = glmnet(train.X, who.train$Life.expectancy, alpha=0, lambda=grid, thresh=1e-12)
ridge.cv = cv.glmnet(train.X, who.train$Life.expectancy, alpha=0, lambda=grid, thresh=1e-12)

bestlambda.R = ridge.cv$lambda.min
bestlambda.R

#test error
ridge.pred = predict(ridge.fit, s=bestlambda.R, newx=test.X)
mean((ridge.pred - who.test$Life.expectancy)^2)

### 축소추정법; Lasso ###

#fitting by lasso model
lasso.fit = glmnet(train.X, who.train$Life.expectancy, alpha=1, lambda=grid, thresh=1e-12)
lasso.cv = cv.glmnet(train.X, who.train$Life.expectancy, alpha=1, lambda=grid, thresh=1e-12)

bestlambda.L = lasso.cv$lambda.min
bestlambda.L

#test error
lasso.pred = predict(lasso.fit, s=bestlambda.L, newx=test.X)
mean((lasso.pred - who.test$Life.expectancy)^2)

#non-zero coefficient estimates
predict(lasso.fit, s=bestlambda.L, type="coefficients")

### 차원축소; PCR ###
set.seed(1)
#fitting by PCR model
library(pls)
pcr.fit = pcr(Life.expectancy~., data=who.train, scale=TRUE, validation="CV")
validationplot(pcr.fit, val.type="MSEP")

#test error
pcr.pred = predict(pcr.fit, who.test, ncomp = 10)
mean((pcr.pred - who.test$Life.expectancy)^2)

### 차원축소; PLS ###

#fitting by PLS model
pls.fit = plsr(Life.expectancy~., data=who.train, scale=TRUE, validation="CV")
validationplot(pls.fit, val.type="MSEP")

#test error
pls.pred = predict(pls.fit, who.test, ncomp = 11)
mean((pls.pred - who.test$Life.expectancy)^2)

# R^2값 비교
test.avg = mean(who.test$Life.expectancy)
lm.r2 = 1-mean((lm.pred - who.test$Life.expectancy)^2)/mean((test.avg - who.test$Life.expectancy)^2)
ridge.r2 = 1-mean((ridge.pred - who.test$Life.expectancy)^2) / mean((test.avg - who.test$Life.expectancy)^2)
lasso.r2 = 1-mean((lasso.pred - who.test$Life.expectancy)^2) / mean((test.avg - who.test$Life.expectancy)^2)
pcr.r2 = 1-mean((pcr.pred - who.test$Life.expectancy)^2) / mean((test.avg - who.test$Life.expectancy)^2)
pls.r2 = 1-mean((pls.pred - who.test$Life.expectancy)^2) / mean((test.avg - who.test$Life.expectancy)^2)

all.r2 = c(lm.r2, ridge.r2, lasso.r2, pcr.r2, pls.r2)
all.r2

# 3. 의사결정나무

# 다시 한번 나눈 데이터를 상기하자
set.seed(1)
train.size = dim(who)[1]/2
train = sample(1:dim(who)[1], train.size)
test = -train
who.train = who[train, ]
who.test = who[test, ]


### 회귀나무 ###

# 회귀나무 훈련
library(tree)
tree.who = tree(Life.expectancy~., data=who)

summary(tree.who)
plot(tree.who)
text(tree.who, pretty=0)

cv.who=cv.tree(tree.who)
plot(cv.who$size,cv.who$dev,type='b') 

prune.who=prune.tree(tree.who,best=5) #5개에서 pruning
plot(prune.who)
text(prune.who, pretty=0)

# 회귀나무 예측
yhat=predict(tree.who, newdata=who.test) #예측
who.test=who[-train,"Life.expectancy"]

plot(yhat,who.test) #점 찍기
abline(0,1) #추세선 
mean((yhat-who.test)^2) #mse

### 랜덤포레스트 ###
library(randomForest)
rf.who=randomForest(Life.expectancy~.,data=who, mtry=5,importance=TRUE) 
summary(rf.who)

yhat.rf = predict(rf.who,newdata=who.test) #예측
plot(yhat.rf, who.test) #점찍기

importance(rf.who)
varImpPlot(rf.who)

# 변수중요도 알아보기 
library(randomForest)
fit.rf = randomForest(Life.expectancy~., data=who, importance=TRUE)
varImpPlot(fit.rf)

### 부스팅 ### 

library(gbm)
set.seed(1)

boost.who=gbm(Life.expectancy~.,data=who.train,distribution="gaussian"
              ,n.trees=5000,interaction.depth=4, shrinkage=0.2, verbose=F) 
summary(boost.who) #각 변수별 상대적 중요도; lstat, rm이 가장 중요함 

par(mfrow=c(1,2))
plot(boost.who,i="Income.composition.of.resources") #정비례
plot(boost.who,i="HIV.AIDS") #반비례

yhat.boost=predict(boost.who,newdata=who[-train,],n.trees=5000) #예측
mean((yhat.boost-who.test)^2) #mse; 조금 더 작아짐 

#예측에서 who[-train]이랑 who.test왜 결과가 다르지?


# 기계 분석과정 참고; https://m.blog.naver.com/leedk1110/220775742538

### 회귀진단 ###


# 계단선택법으로 얻은 모형으로 회귀진단 할거임 

fit = lm(Life.expectancy~Schooling+HIV.AIDS+Adult.Mortality+Income.composition.of.resources
         +Diphtheria+BMI+GDP+Polio+Measles+thinness..1.19.years+Year+Total.expenditure
         +Hepatitis.B+under.five.deaths+infant.deaths+Alcohol, data=who)

# 다중공선성이 있는 변수를 알아보자

library(car)
vif(fit)

# 변수중요도 알아보기 
library(randomForest)
fit.rf = randomForest(Life.expectancy~., data=who, importance=TRUE)
varImpPlot(fit.rf)

# 5 또는 10을 넘는 변수에 underfivedeath, infantdeath
# 특기 가장 큰 값인 under.five.deaths를 빼고 다시 적합

fit2 = lm(Life.expectancy~Schooling+HIV.AIDS+Adult.Mortality+Income.composition.of.resources
          +Diphtheria+BMI+GDP+Polio+Measles+thinness..1.19.years+Year+Total.expenditure
          +Hepatitis.B+infant.deaths+Alcohol, data=who)
vif(fit2)

# 이제 5를 넘는 변수가 없으므로 이 모형을 선택하고 
# 나머지 가정에 대해서도 회귀진단을 하겠다.

summary(fit2)

# 결정계수; 0.82
# 나머지 가정에 대해서도 회귀진단


#1; 선형성도 맞는 듯 (등분산성)
#2; 약간 애매하지만 정규성 가정 맞는 듯 ?
#3
#4

par(mfrow=c(2,2))
plot(fit2)

# 이상치 처리; 76,2308,2310

### 이상치 처리 ###

# 이상치 확인
boxplot(who$Alcohol) # 극단치 기준 확인
boxplot(who$Alcohol)$stats #통계량 확인

# 이상치 제거
who = who[who$Alcohol<17.31, ] # 기준을 벗어나는 값 제거
boxplot(who$Alcohol) # 다시 확인

lm.fit = lm(Life.expectancy ~ Alcohol) #기대수명~HIV바이러스
summary(lm.fit) #통계량 확인

plot(Life.expectancy,Alcohol,pch=20) # 그래프
abline(lm.fit,lwd=3,col="red") # 최소제곱추정선

par(mfrow=c(2,2))
plot(lm.fit) # For 잔차분석; Q-Q plot

fit3 = lm(Life.expectancy~Schooling+HIV.AIDS+Adult.Mortality+Income.composition.of.resources
          +Diphtheria+BMI+GDP+Polio+Measles+thinness..1.19.years+Year+Total.expenditure
          +Hepatitis.B+infant.deaths+Alcohol, data=who[-c(76,2308,2310),])
summary(fit3) 

# 예측구간
set.seed(1)
pred=predict(fit3, newdata=who, interval="predict")
pred=as.data.frame(pred)
head(pred) #예측구간  

pred = cbind(pred,who$Life.expectancy)
head(pred) #실제값과 비교

# 잘 예측되었는지 한눈에 확인하기 위해서 비율을 살펴보자

tf = NA
pred = cbind(pred,tf)
pred$tf[pred$'who$Life.expectancy'>=pred$lwr & pred$'who$Life.expectancy'<=pred$upr] = T
pred$tf[is.na(pred$tf)] = F
head(pred)
sum(pred$tf=="TRUE")/dim(pred)[1] #예측에 성공하는 비율이 약 93%로 성공적인 회귀분석 완료
