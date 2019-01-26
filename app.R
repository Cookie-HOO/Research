library(shiny)
library(DT)
library(shinythemes)
library(ggplot2)
library(reshape2)  # 数据整理
library(car)   # 模型诊断
library(leaps) # 全子集回归
library(psych) # 主成分作图 corr.test
library(MASS)  # 岭回归
library(rpart) # 经典决策树
library(rpart.plot)  # 经典决策树
library(party)  # 条件推断树
library(randomForest)  # 随机森林
library(e1071)  # 支持向量机//朴素贝叶斯
library(kknn)  # K最临近
library(adabag) # boosting//bagging分类
library(qcc)  # 泊松回归的过度离势检验
# 读取身高数据2.10
cp_height <- read.csv('2.10cp_height.csv')
male <- cp_height$male
female <- cp_height$female
fit2.10 <- lm(male~female)
unitable <- data.frame('ft'=30.48,'in'=2.54, 'm'=100, 'dm'=10)
corr <- cor(male,female)
covv <- cov(male,female)
# 读取报纸数据2.12
newspaper <- read.csv('2.12newspaper.csv')
ord <- newspaper$ordinary
sun <- newspaper$sunday
fit2.12 <- lm(sun~ord)
# 读取并清洗香烟数据3.15
cig <- read.csv('3.15cigarette.csv')
attach(cig)
for(i in colnames(cig[c(1,2,4,5,6)])){
  exp <- gsub('i',i,'i[i>99]<-i[i>99]/10')
  eval(parse(text=exp))
}
cigarette <- data.frame(sales,age,HS,income,bl_ratio,fe_ratio,price)
detach(cig)
fit3.15a <- lm(sales~.,data=cigarette)
fit3.15b <- lm(sales~.-fe_ratio-bl_ratio+I(fe_ratio+bl_ratio),data=cigarette)
fit3.15c <- lm(sales~-income,data=cigarette)
fit3.15d <- lm(sales~age+price+income,data=cigarette)
fit3.15e <- lm(sales~income,data=cigarette)
cigar <- data.frame(cigarette$sales[-c(2,29,30)],cigarette$income[-c(2,29,30)],cigarette$price[-c(2,29,30)],cigarette$age[-c(2,29,30)])
colnames(cigar) <- c('sales','income','price','age')
cigare <- data.frame(cigar$sales[-c(7,8,31)],cigar$income[-c(7,8,31)],cigar$price[-c(7,8,31)],cigar$age[-c(7,8,31)])
colnames(cigare) <- c('sales','income','price','age')
fit3.15op <- lm(sales~.,data=cigar)
fit3.15opp <- lm(sales~.,data=cigare)
fit3.15oppp <- lm(sales~.+I(price^2),data=cigare)
cig_n <- length(fitted(fit3.15d))
cig_p <- length(coefficients(fit3.15d))

# 读取模拟数据建模数据4.12-4.15
mod <- read.csv('4.12-4.14model.csv')
fit4all <- lm(Y~.,mod)
mod_n <- length(fitted(fit4all))
mod_p <- length(coefficients(fit4all))

# 读取多重共线性数据10.7
mll <- read.csv('10.7collinearity.csv',sep='\t',header = T)
fit10.7pc <- princomp(~X1+X2+X3+X4+X5+X6, data=mll, cor=TRUE,scores=TRUE)  #从相关矩阵求解，并列出主成分得分
pre<-predict(fit10.7pc)
main <- data.frame(mll$Y,pre[,1],pre[,2],pre[,3],pre[,4],pre[,5],pre[,6])
colnames(main) <- c('Y','main1','main2','main3','main4','main5','main6')
fit10.7 <- lm(Y~.,mll)
fit10.7op <- lm(Y~.,main)
data <- data.frame(scale(mll))
fit10.7ri <- lm.ridge(Y~.,data=data,lambda=seq(0.2,2,0.1))
fit10.7pre <- lm.ridge(Y~.,data=data,lambda=.3)
fit10.7main2 <- lm(main2~main1,data=main)

beta <- coef(fit10.7op)
A <- loadings(fit10.7pc)
coll_eigen <- as.data.frame(rbind(CP1 = A[,1], CP2 = A[,2]))
x.bar <- fit10.7pc$center
x.sd <- fit10.7pc$scale
coef <- (beta[2]*A[,1]+beta[3]*A[,2])/x.sd
beta0 <- beta[1]-sum(x.bar*coef)
coll_cps_conclusion <- c(beta0, coef)
coll_mean_sd <- as.data.frame(rbind(mean = sapply(mll,mean), sd = sapply(mll,sd)))

# 读取telephone的报告数据
telephone <- read.csv('telephone.csv')
colnames(telephone) <- c('prodiff','tele','lar_ratio','lit_ratio')
fit.rep <- lm(prodiff~.-lit_ratio,data=telephone)
fit.replog <- lm(log(prodiff+1)~.-lit_ratio,data=telephone)
tel_n <- length(fitted(fit.rep))
tel_p <- length(coefficients(fit.rep))
residplot <- function(fit,nbreaks=100)
{
  rstudent <- rstudent(fit)
  hist(rstudent,breaks=nbreaks, freq=F,ylim=c(0,.7))
  rug(jitter(rstudent))
  curve(dnorm(x,mean(rstudent),sd(rstudent)), add=T, lty=1, lwd=2.5, col='blue')
  lines(density(rstudent)$x, density(rstudent)$y, lty=2, lwd=2.5, col='red')
  legend("topright", legend=c("Normal", "Density"), lty=1:2, lwd=2.5, col=c('blue','red'))
}

# 读取credit的数据
credit <- read.table("credit.txt", header = T)
credit$Rating <- factor(credit$Rating, levels=c(0,1), labels=c("bad", "good"))  # 结果变量因子化
set.seed(1234)
train <- sample(nrow(credit), 0.7*nrow(credit))  # 70%数据用于训练
credit.train <- credit[train,]  # 挑出70%
credit.validate <- credit[-train,]  # 减掉70%剩下的用于验证
# regression
fit.logit <- glm(Rating~., data=credit.train, family=binomial())  # 用生成的70%的数据进行建模
fit.logit <- step(fit.logit)  # 逐步回归，将不好的变量踢掉，再进行上面的分析
fit.probit <- glm(Rating~., data=credit.train, family=binomial(link=probit))  # 用生成的70%的数据进行建模
fit.probit <- step(fit.probit)  # 逐步回归，将不好的变量踢掉，再进行上面的分析
prob.logit <- predict(fit.logit, credit.validate, type="response")  # 不写type默认恶性的对数概率
logit.pred <- factor(prob.logit>0.5, levels=c(FALSE, TRUE), labels=c("bad", "good")) # >.5恶性
logit.perf <- table(credit.validate$Rating,  # 验证集里的真实的结果 
                    logit.pred,         # 预测出的结果
                    dnn=c("Actual", "Predicted"))  # 标签名
prob.probit <- predict(fit.probit, credit.validate, type="response")  # 不写type默认恶性的对数概率
probit.pred <- factor(prob.probit>0.5, levels=c(FALSE, TRUE), labels=c("bad", "good")) # >.5恶性
probit.perf <- table(credit.validate$Rating,  # 验证集里的真实的结果 
                     probit.pred,         # 预测出的结果
                     dnn=c("Actual", "Predicted"))  # 标签名
# tree
set.seed(1234)
dtree <- rpart(Rating~., data=credit.train, method="class", parms=list(split="information"))
dtree.pruned<-prune(dtree, cp=0.01)
# 剪枝？？
dtree.pred <- predict(dtree.pruned, credit.validate, type="class")  # 预测
dtree.p <- predict(dtree.pruned, credit.validate)
dtree.perf <- table(credit.validate$Rating, dtree.pred, dnn=c("Actual", "Predicted"))  # 结果

fit.ctree <- ctree(Rating~., data=credit.train)
ctree.pred <- predict(fit.ctree, credit.validate, type="response")  # 预测
ctree.perf <- table(credit.validate$Rating, ctree.pred, dnn=c("Actual", "Predicted"))  #结果

# forest
set.seed(1234)
fit.forest <- randomForest(Rating~., data=credit.train, na.action=na.roughfix, importance=T)
forest.pred <- predict(fit.forest, credit.validate)  # 预测
forest.perf <- table(credit.validate$Rating, forest.pred, dnn=c("Actual", "Predicted"))  #结果

# svm
set.seed(1234)
fit.svm <- svm(Rating~., data=credit.train)
svm.pred <- predict(fit.svm, na.omit(credit.validate))  # 预测 不允许新数据集有空值给
svm.perf <- table(na.omit(credit.validate)$Rating, svm.pred, dnn=c("Actual", "Predicted"))  #结果
credit.name = c('Logit', 'Probit', 'Decision_Tree','Condetion_Tree', 'Forest', 'SVM', 'KNN', 'NaiveBayes','boosting','bag')
credit.Sensitivity = c(0.81,0.81,0.91,0.79,0.82,0.83,0.81,0.77,0.85,0.87)
credit.Specificity = c(0.77,0.77,0.64,0.8,0.74,0.75,0.74,0.82,0.73,0.69)
credit.Positive_Predictive_Value = c(0.85,0.85,0.8,0.86,0.84,0.84,0.83,0.87,0.83,0.82)
credit.Negative_Predictive_Value = c(0.72,0.72,0.82,0.7,0.73,0.73,0.71,0.69,0.75,0.77)
credit.Accuracy = c(0.8, 0.8, 0.81, 0.79,0.79,0.8,0.78,0.79,0.8,0.8)
evalute.table <- data.frame('name'=credit.name,'Sensitivity(TPR)'=credit.Sensitivity,'Specificity(1-FPR)'=credit.Specificity,'Positive_Predictive_Value'=credit.Positive_Predictive_Value,'Negative_Predictive_Value'=credit.Negative_Predictive_Value,'Accuracy'=credit.Accuracy)
# knn
credit.knn <- kknn(Rating~.,credit.train, credit.validate, distance = 5,kernel= "triangular")
fit.knn <- fitted(credit.knn)
knn.perf <- table(credit.validate$Rating, fit.knn,dnn=c("Actual", "Predicted"))
# naivebayes
credit.nb <- naiveBayes(Rating~.,credit.train)
fit.nb <- predict(credit.nb, credit.validate)
nb.perf <- table(credit.validate$Rating, fit.nb,dnn=c("Actual", "Predicted"))
# boosting
fit.boosting <- boosting(Rating~.,data=credit.train, mfinal=20)#建立adaboost分类模型
boosting.pred <- predict(fit.boosting,credit.validate)$class  
boosting.perf <-table(credit.validate$Rating, boosting.pred)#查看模型的预测结果
error.boosting <- errorevol(fit.boosting,credit.train) 
# bagging
fit.bagging <- bagging(Rating~.,data=credit.train, mfinal=25) #建立bagging分类模型
bagging.pred <- predict(fit.bagging,credit.validate)$class  
bagging.perf <-table(credit.validate$Rating, bagging.pred)#查看模型的预测结果
error.bagging<-errorevol(fit.bagging,credit.train)                       #计算全体的误差演变

# 整合数据
new.perfor <- melt(evalute.table, ID="name")
perform <- function(table, n=2){
  if(!all(dim(table) == c(2,2))){stop("Must be a 2*2 table")}
  tn = table[1,1]
  fp = table[1,2]
  fn = table[2,1]
  tp = table[2,2]
  sensitivity = tp/(tp+fn)  #TPR
  specificity = tn/(tn+fp)  #1-FPR
  ppp = tp/(tp+fp)
  npp = tn/(tn+fn)
  hitrate = (tp+tn)/(tp+tn+fn+fp)
  result <- paste("Sensitivity(TPR) = ", round(sensitivity,n),
                  "\nSpecificity(1-FPR) = ", round(specificity,n),
                  "\nPositive Predictive Value = ", round(ppp, n),
                  "\nNegative Predictive Value = ", round(npp,n),
                  "\nAccuracy = ", round(hitrate, n), "\n", sep=" ")
  cat(result)
}
# 作业9--order
sa.train <- read.table("sa-train.txt", header = T)
sa.test <- read.table("sa-test.txt", header = T)
# 全变量
fit.order <- polr(as.factor(score)~W1+W2+W3+W4+W5+W6+W7,data=sa.train, method="probit",Hess=T)
# anova
fit.order.anova <- polr(as.factor(score)~W2+W3+W6+W7,data=sa.train, method="probit",Hess=T)
# aic
fit.order.aic <- step(fit.order)
# anova后的变量预测
order.pred.anova <- predict(fit.order.anova,sa.test)
order.perf.anova <- table(sa.test$score, order.pred.anova,dnn=c("Actual", "Predicted"))
# aic后的变量预测
order.pred.aic <- predict(fit.order.aic,sa.test)
order.perf.aic <- table(sa.test$score, order.pred.aic,dnn=c("Actual", "Predicted"))

#作业10--possion-search
search <- read.csv("search.csv")
# 全变量
fit.search <- glm(click~.,data=search, family=poisson())
# anova后的变量
fit.search.anova <- glm(click~.-show,data=search, family=poisson())
# aic后的变量
fit.search.aic <- step(fit.search)

ui <- fluidPage(
  titlePanel('BJUT Regression by:Cookie 2018'),
  navbarPage(
    themeSelector(),
    tabPanel('HW1-cp_height2.10',
             sidebarPanel(
               conditionalPanel(
                 'input.cp_height === "Data"',
                 h4("2.10cp_height.csv"),
                 downloadButton("cp_height.csv", "Download")
                 ),
               conditionalPanel(
                 'input.cp_height === "Cor&Cov"',
                 h4("Cor&Cov:"),
                 tags$hr(),
                 selectInput('unit', 'unit:', choices=c('ft'='30.48','in'='2.54','m'='100','dm'='10','cm'='1')),
                 br(),
                 sliderInput('times1','Times(Both):',min=1,max=10,value=1),
                 br(),
                 sliderInput('times2','Times(One):',min=1,max=10,value=1),
                 br(),
                 sliderInput('plus1','Plus(Both):',min=0,max=10,value=0),
                 br(),
                 sliderInput('plus2','Plus(One):',min=0,max=10,value=0),
                 h5('1-cov(cp_height)'),
                 h5('2-cov(cp_height): cm->in'),
                 br(),
                 h5('3-cor(cp_height)'),
                 h5('4-cor(cp_height): cm->in'),
                 tags$hr(),
                 h5('5-cor(cp_height): male:male-5')
                 ),
               conditionalPanel(
                 'input.cp_height === "Regression"',
                 h4("Regression:"),
                 tags$hr(),
                 h5('6-Model-Y-Reasons'),
                 br(),
                 h5('7-Gradient==0?'),
                 h5('8-Intercept==0?')
                 ),
               conditionalPanel(
                 'input.cp_height === "Test"',
                 h4("Test:"),
                 tags$hr(),
                 checkboxGroupInput('ttest', 'Test:',
                                    c('General','Norm','Liner','Cov','Independence'), selected = c('General','Norm','Liner','Cov','Independence')))
             ),
             mainPanel(
               tabsetPanel(
                 id = 'cp_height',
                 tabPanel("Data", 
                          h2('CP_Height-Summary',align='center'),
                          verbatimTextOutput('summary'),
                          h2('CP_Height-Data',align='center'),
                          DT::dataTableOutput("table_height")
                          ),
                 tabPanel("Cor&Cov", h3(paste('Cor_Orginal:',corr)), 
                          h3(paste('Cov_Orginal:',covv)),
                          tags$hr(),
                          h3(textOutput("newcor")),
                          h3(textOutput("newcov")),
                          tags$hr(),
                          h3('Cov_New/Cov_Orginal:'),
                          h3(textOutput("di")),
                          tableOutput('unitable')),
                 tabPanel("Regression",
                          h1('CP_Height-Plot',align='center'),
                          plotOutput('pointt'),
                          h1('Regression-Summary',align='center'),
                          verbatimTextOutput('summfit')),
                 tabPanel('Test',
                          h2('General',align='center'),
                          plotOutput('point_gen'),
                          h2('Normal',align='center'),
                          plotOutput('point_gao'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin'),
                          h2('Cov',align='center'),
                          verbatimTextOutput('point_var'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den')),
                 tabPanel('Thinking',
                          plotOutput('point_all')),
                 tabPanel('Report',
                          plotOutput('cp_report')))
             )
    ),
    tabPanel('HW1-newspaper2.12', 
             sidebarPanel(
               conditionalPanel(
                 'input.newspaper === "Data"',
                 h4("2.12newspaper.csv"),
                 downloadButton("newspaper.csv", "Download")
               ),
               conditionalPanel(
                 'input.newspaper === "Regression"',
                 h4("Regression:"),
                 tags$hr(),
                 h5('1-sun~ord-Plot,Liner?reasonable?'),
                 h5('2-sun~ord-Regression'),
                 br(),
                 h5('3-confident-interval95%'),
                 h5('4-Liner?Test/Conclusion'),
                 h5('5-explain/adj.R')
                 ),
               conditionalPanel(
                 'input.newspaper === "Predict"',
                 h4('Predict:'),
                 tags$hr(),
                 checkboxGroupInput('ttest', 'questions:',
                                    c('6-500,000-Mean-Confidence-Interval-95%','7-500,000-Interval-95%','8-2,000,000-Interval-95%'), selected = c('6-500,000-Mean-Confidence-Interval-95%','7-500,000-Interval-95%','8-2,000,000-Interval-95%')),
                 h5('Note:'),
                 h5('Unit&Scope')),
               conditionalPanel(
                 'input.newspaper === "Test"',
                 h4('Test:'),
                 tags$hr(),
                 checkboxGroupInput('ttest', 'Types:',
                                    c('General','Norm','Liner','Cov','Independence'), selected = c('General','Norm','Liner','Cov','Independence')))
             ),
             mainPanel(
               tabsetPanel(
                 id = 'newspaper',
                 tabPanel("Data", 
                          h2('Newspaper-Summary',align='center'),
                          verbatimTextOutput('summary_new'),
                          h2('Newspaper-Data',align='center'),
                          DT::dataTableOutput("table_new")
                          ),
                 tabPanel("Regression",
                          h1('Newspaper-Plot',align='center'),
                          plotOutput('pointt_new'),
                          h1('Regression-Summary',align='center'),
                          verbatimTextOutput('summfit_new'),
                          h1('Confident-Interval-95%',align='center'),
                          verbatimTextOutput('confint')),
                 tabPanel("Predict",
                          verbatimTextOutput('predict_one'),
                          verbatimTextOutput('predict_two'),
                          verbatimTextOutput('predict_three'),
                          h4('predict(fit2.12, data.frame(ord=500), interval="confidence")'),
                          br(),
                          h4('predict(fit2.12, data.frame(ord=500), interval="prediction")'),
                          br(),
                          h4('predict(fit2.12, data.frame(ord=2000), interval="prediction")')),
                 tabPanel('Test',
                          h2('General',align='center'),
                          plotOutput('point_gen_new'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_new'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_new'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_new'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_new')),
                 tabPanel('Thinking',
                          plotOutput('point_all_new')))
             )),
    tabPanel('HW2-cigarette3.15', sidebarPanel(
      conditionalPanel(
        'input.cigarette === "Data"',
        h4("3.15cigarette.csv"),
        downloadButton("cigarette.csv", "Download")
      ),
      conditionalPanel(
        'input.cigarette === "Regression"',
        h4("Regression:"),
        tags$hr(),
        h5('1-fe_ratio?'),
        h5('2-I(fe_ratio+bl_ratio)'),
        h5('3-income-confident-interval95%')
        
      ),
      conditionalPanel(
        'input.cigarette === "Adj.r.square"',
        h4('Adj.r.square:'),
        tags$hr(),
        checkboxGroupInput('adj', 'adj.r.square:',
                           c('4-except-income','5-age+price+income','6-income-only'), selected = c('4-except-income','5-age+price+income','6-income-only')),
        h5('Note:'),
        h5('be accounted for == adj.r.square')),
      conditionalPanel(
        'input.cigarette === "Test"',
        h4('Test:'),
        tags$hr(),
        checkboxGroupInput('ttest', 'Types:',
                           c('General','Norm','Liner','Cov','Independence'), selected = c('General','Norm','Liner','Cov','Independence')))
    ),
    mainPanel(
      tabsetPanel(
        id = 'cigarette',
        tabPanel("Data", 
                 h2('Cigarette-Summary',align='center'),
                 verbatimTextOutput('summary_cig'),
                 h2('Cigarette-Data',align='center'),
                 DT::dataTableOutput("table_cig")
        ),
        tabPanel("Regression",
                 h1('Cigarette_Org-Plot',align='center'),
                 plotOutput('pointt_cig_org'),
                 h1('Cigarette_Adj-Plot',align='center'),
                 plotOutput('pointt_cig_adj'),
                 h1('fe_ratio',align='center'),
                 verbatimTextOutput('summ_fe_ratio'),
                 h1('I(fe_ratio+bl_ratio)',align='center'),
                 verbatimTextOutput('summ_bl_ratio'),
                 h1('income_confint',align='center'),
                 verbatimTextOutput('summ_income_confint')),
        tabPanel("Adj.R.Square",
                 h2('except-income',align='center'),
                 verbatimTextOutput('adj_exc_income'),
                 h2('age+price+income',align='center'),
                 verbatimTextOutput('adj_agepriceincome'),
                 h2('income-only',align='center'),
                 verbatimTextOutput('adj_income')
                 ),
        tabPanel('Test',
                 h2('General',align='center'),
                 plotOutput('point_gen_cig'),
                 h2('Norm',align='center'),
                 plotOutput('point_gao_cig'),
                 h2('Liner',align='center'),
                 plotOutput('point_lin_cig'),
                 h2('Var',align='center'),
                 verbatimTextOutput('point_var_cig'),
                 h2('Independence',align='center'),
                 verbatimTextOutput('point_den_cig')),
        tabPanel('Thinking',
                 plotOutput('point_all_cig')))
    )),
    tabPanel('HW2-model4.12', 
             sidebarPanel(
               conditionalPanel(
                 'input.model === "Data"',
                 h4("4.12-4.14model.csv"),
                 downloadButton("model.csv", "Download")
               ),
               conditionalPanel(
                 'input.model === "4.12Regression_all"',
                 h4("Regression_all:"),
                 tags$hr()
               ),
               conditionalPanel(
                 'input.model === "Test"',
                 h4('Test4.12-1:'),
                 tags$hr(),
                 checkboxGroupInput('ttest', 'Types:',
                                    c('General','Norm','Liner','Cov','Independence'), selected = c('General','Norm','Liner','Cov','Independence')),
                 tags$hr(),
                 h5('4.12-1-test all assumptions')),
               conditionalPanel(
                 'input.model === "Outlier_Plot"',
                 h4('Outlier4.12-2&3:'),
                 tags$hr(),
                 checkboxGroupInput('outiner_plot', 'Types:',
                                    c('residual&plot','Cook&plot','DFFITS&plot','hatvalue&plot','Scale-Location'), selected = c('residual&plot','Cook&plot','DFFITS&plot','hatvalue&plot','Scale-Location')),
                 tags$hr(),
                 h5('4.12-2-residual/Cook/DFFITS/hatvalues'),
                 h5('4.12-3-plot all above and Scale-Location')),
               conditionalPanel(
                 'input.model === "Outlier"',
                 h4('Outlier4.12-4:'),
                 tags$hr(),
                 checkboxGroupInput('outiner', 'Types:',
                                    c('outlier','leverage','influential'), selected = c('outlier','leverage','influential')),
                 tags$hr(),
                 h5('4.12-4-find all the outiners')),
               conditionalPanel(
                 'input.model === "Regression"',
                 h4('Regression4.13:'),
                 h5('4.13-1-test X4'),
                 h5('4.13-2-test X5'),
                 h5('4.13-3-test X6'),
                 h5('4.13-4-find find a best description of Y')
             )),
             mainPanel(
               tabsetPanel(
                 id = 'model',
                 tabPanel("Data", 
                          h2('Model-Summary',align='center'),
                          verbatimTextOutput('summary_mod'),
                          h2('Model-Data',align='center'),
                          DT::dataTableOutput("table_mod")
                 ),
                 tabPanel("Regression_all",
                          h1('Model-Regression-Plot',align='center'),
                          plotOutput('pointt_mod'),
                          h1('Model-Regression-Summary',align='center'),
                          verbatimTextOutput('summfit_mod')
                          ),
                 tabPanel('Test',
                          h2('General',align='center'),
                          plotOutput('point_gen_mod'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_mod'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_mod'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_mod'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_mod')),
                 tabPanel('Outlier_Plot',
                          h2('residual',align='center'),
                          DT::dataTableOutput("mod_residual"),
                          h2('residual-plot',align='center'),
                          plotOutput('mod_resid_plot'),
                          tags$hr(),
                          h2('Cook',align='center'),
                          DT::dataTableOutput("mod_Cook"),
                          h2('Cook-plot',align='center'),
                          plotOutput('mod_Cook_plot'),
                          tags$hr(),
                          h2('DFFITS',align='center'),
                          DT::dataTableOutput("mod_DFFITS"),
                          h2('DFFITS-plot',align='center'),
                          plotOutput('mod_DFFITS_plot'),
                          tags$hr(),
                          h2('hatvalues',align='center'),
                          DT::dataTableOutput("mod_hatvalue"),
                          h2('hatvalue-plot',align='center'),
                          plotOutput('mod_hatvalue_plot'),
                          tags$hr(),
                          h2('Scale-Location',align='center'),
                          plotOutput('mod_scale')),
                 tabPanel('Outlier',
                          h2('general',align='center'),
                          plotOutput('mod_gen'),
                          h2('outlier',align='center'),
                          plotOutput('mod_outlier'),
                          h2('leverage',align='center'),
                          plotOutput('mod_leverage'),
                          h2('influential',align='center'),
                          plotOutput('mod_influential')
                          ),
                 tabPanel('Regression',
                          plotOutput('mod_all')))
             )),
    tabPanel('HW3-cigarette-more11.8',
             mainPanel(
               tabsetPanel(
                 tabPanel("Data", 
                          h2('The same data as cigarette3.15',align='center')
                 ),
                 tabPanel("All",
                          h1('All-Plot',align='center'),
                          plotOutput('pointt_all_cig')
                 ),
                 tabPanel("Regression_Fir",
                          h1('Cig-Regression-first',align='center'),
                          plotOutput('pointt_regression_cig_fir'),
                          verbatimTextOutput('summ_cig_regression_fir')
                 ),
                 tabPanel('Test_Fir',
                          h2('General',align='center'),
                          plotOutput('point_gen_cigmore'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_cigmore'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_cigmore'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_cigmore'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_cigmore')),
                 tabPanel('Outlier_Fir',
                          h2('general',align='center'),
                          plotOutput('cig_gen'),
                          h2('outlier',align='center'),
                          plotOutput('cig_outlier'),
                          h4('conclusion:29&30 == outlier'),
                          h2('leverage',align='center'),
                          plotOutput('cig_leverage'),
                          h4('conclusion:02 == heigh-leverage'),
                          h2('influential',align='center'),
                          plotOutput('cig_influential'),
                          h4('conclusion:29&30 == influential')
                 ),
                 tabPanel('Regression_Sec',
                          h1('Cig-Regression-sec',align='center'),
                          plotOutput('pointt_regression_cig_sec'),
                          verbatimTextOutput('summ_cig_regression_sec')),
                 tabPanel('Test_Sec',
                          h2('General',align='center'),
                          plotOutput('point_gen_cigmore_sec'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_cigmore_sec'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_cigmore_sec'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_cigmore_sec'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_cigmore_sec')),
                 tabPanel('Outlier_Sec',
                          h2('general',align='center'),
                          plotOutput('cig_gen_sec'),
                          h2('outlier',align='center'),
                          plotOutput('cig_outlier_sec'),
                          h2('leverage',align='center'),
                          plotOutput('cig_leverage_sec'),
                          h2('influential',align='center'),
                          plotOutput('cig_influential_sec'),
                          h4('conclusion:07&08&31(new) == influential')
                          ),
                 tabPanel('Regression_Thir',
                          h1('Cig-Regression-Thir',align='center'),
                          plotOutput('pointt_regression_cig_thir'),
                          verbatimTextOutput('summ_cig_regression_thir')),
                 tabPanel('Test_Thir',
                          h2('General',align='center'),
                          plotOutput('point_gen_cigmore_thir'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_cigmore_thir'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_cigmore_thir'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_cigmore_thir'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_cigmore_thir')),
                 tabPanel('Outlier_Thir',
                          h2('general',align='center'),
                          plotOutput('cig_gen_thir'),
                          h2('outlier',align='center'),
                          plotOutput('cig_outlier_thir'),
                          h2('leverage',align='center'),
                          plotOutput('cig_leverage_thir'),
                          h2('influential',align='center'),
                          plotOutput('cig_influential_thir')
                 ),
                 tabPanel('Regression_Thir_Add',
                          h1('Cig-Regression-Thir-Add',align='center'),
                          plotOutput('pointt_regression_cig_thir_add'),
                          verbatimTextOutput('summ_cig_regression_thir_add')),
                 tabPanel('Test_Thir_Add',
                          h2('General',align='center'),
                          plotOutput('point_gen_cigmore_thir_add'),
                          h2('Norm',align='center'),
                          plotOutput('point_gao_cigmore_thir_add'),
                          h2('Liner',align='center'),
                          plotOutput('point_lin_cigmore_thir_add'),
                          h2('Var',align='center'),
                          verbatimTextOutput('point_var_cigmore_thir_add'),
                          h2('Independence',align='center'),
                          verbatimTextOutput('point_den_cigmore_thir_add')),
                 tabPanel('Outlier_Thir_Add',
                          h2('general',align='center'),
                          plotOutput('cig_gen_thir_add'),
                          h2('outlier',align='center'),
                          plotOutput('cig_outlier_thir_add'),
                          h2('leverage',align='center'),
                          plotOutput('cig_leverage_thir_add'),
                          h2('influential',align='center'),
                          plotOutput('cig_influential_thir_add')),
                 tabPanel('Report',plotOutput('cig_report'))
             ))),
    tabPanel('HW4-collinearity10.7',
             sidebarPanel(
               conditionalPanel(
                 'input.collinearity === "Data"',
                 h4("10.7collinearity.csv"),
                 downloadButton("collinearity.csv", "Download")
               ),
               conditionalPanel(
                 'input.collinearity === "Collinearity"',
                 h4("collinearity:"),
                 tags$hr(),
                 h5('1-condition number:collinearity?'),
                 br(),
                 h5('4-how many mll?'),
                 h5('5-which var are invovled?'),
                 h5('Conclusion:'),
                 h5('1&6----2&5'),
                 h5('3&4----4&5'),
                 tags$hr(),
                 h5('6-relationship among the var(in each set)'),
                 checkboxGroupInput('ttest', 'Content:',
                                    c('1&6','2&5','3&4','4&5'), selected = c('1&6','2&5','3&4','4&5'))
                 
                 ),
               conditionalPanel(
                 'input.collinearity === "PCA"',
                 h4("PCA:"),
                 tags$hr(),
                 checkboxGroupInput('ttest', 'Content:',
                                    c('All-CPs','Regression-All-CPs-Summary','The-Most-Two-CPs','The-Most-Two-CPs-Summary','Scree-Plot-With-Parallel-Analysis'), selected = c('All-CPs','Regression-All-CPs-Summary','The-Most-Two-CPs','The-Most-Two-CPs-Summary','Scree-Plot-With-Parallel-Analysis')),
                 tags$hr(),
                 h5('2-all PCs and regress Y on them, sig?'),
                 h5('3-relationship between the first two PCs, why?'),
                 br(),
                 h5('7-how many PCs?')
               ),
               conditionalPanel(
                 'input.collinearity === "Regression-PCA"',
                 h4("Regression-PCA:"),
                 tags$hr(),
                 h5('8-which model?')
               ),
               conditionalPanel(
                 'input.collinearity === "Regression-Ridge"',
                 h4("Regression-Ridge:"),
                 tags$hr(),
                 h5('8-which model?'))
             ),
             mainPanel(
               tabsetPanel(
                 id = 'collinearity',
                 tabPanel("Data", 
                          h2('Collinearity-Data-Summary',align='center'),
                          verbatimTextOutput('summary_coll'),
                          h2('Collinearity-Data',align='center'),
                          DT::dataTableOutput("table_coll")
                 ),
                 tabPanel("Collinearity", h3(paste('Condition Number:',kappa(cor(mll),exact=T))), 
                          h2('VIF',align='center'),
                          verbatimTextOutput('coll_vif'),
                          tags$hr(),
                          h2('corr.test',align='center'),
                          verbatimTextOutput('coll_corrtest'),
                          h2('1&6', align='center'),
                          plotOutput('coll_mull_one'),
                          h2('2&5', align='center'),
                          plotOutput('coll_mull_two'),
                          h2('3&4', align='center'),
                          plotOutput('coll_mull_three'),
                          h2('4&5', align='center'),
                          plotOutput('coll_mull_four')
                          ),
                  tabPanel("PCA",
                           h2('All-CPs',align='center'),
                           DT::dataTableOutput('coll_all_cps'),
                           h2('Regression-All-CPs-Summary',align='center'),
                           verbatimTextOutput('coll_summcps'),
                           h2('The-Most-Two-CPs',align='center'),
                           plotOutput('coll_two_cps'),
                           h2('The-Most-Two-CPs-Summary',align='center'),
                           verbatimTextOutput('coll_two_cps_summ'),
                           h2('Scree-Plot-With-Parallel-Analysis',align='center'),
                           plotOutput('coll_Scree_Plot')
                  ),
                  tabPanel('Regression-PCA',
                           h2('Eigenavector',align='center'),
                           verbatimTextOutput('coll_cps_eigenavector'),
                           h2('CPs-Beta',align='center'),
                           verbatimTextOutput('coll_cps_beta'),
                           h2('Sd',align='center'),
                           verbatimTextOutput('coll_cps_sd'),
                           h2('Conclusion',align='center'),
                           verbatimTextOutput('coll_cps_conclusion')),
                  tabPanel('Regression-Ridge',
                           h2('Ridge-Regression-based on L-W/HKB/GCV',align='center'),
                           plotOutput('coll_ridge'),
                           h2('Estimate-lambda',align='center'),
                           verbatimTextOutput('coll_ridge_lambda'),
                           h2('Estimate-based on GCV',align='center'),
                           verbatimTextOutput('coll_ridge_est'),
                           h2('Mean-Sd',align='center'),
                           verbatimTextOutput('coll_ridge_mean_sd'),
                           h2('Conclusion',align='center'),
                           h3('Conclusion',align='center')
                  )
                 )
               )
             ),
    tabPanel('HW5-report', sidebarPanel(
      conditionalPanel(
        'input.report_tel === "Data"',
        h4("telephone.csv"),
        downloadButton("telephone.csv", "Download")
      )
    ),
    mainPanel(
      tabsetPanel(
        id = 'report_tel',
        tabPanel("Data", 
                 h2('Telephone-Data-Summary',align='center'),
                 verbatimTextOutput('summary_tele'),
                 h2('Telephone-Data',align='center'),
                 DT::dataTableOutput("table_tele")
        ),
        tabPanel('Regression-Fir',
                 h1('Tele-Regression-Test',align='center'),
                 plotOutput('pointt_regression_tele_test'),
                 h1('Tele-Regression-Fir',align='center'),
                 plotOutput('pointt_regression_tele_fir'),
                 verbatimTextOutput('summ_tele_regression_fir')),
        tabPanel("Test_Fir", 
                 h2('General',align='center'),
                 plotOutput('point_gen_tele'),
                 h2('Norm',align='center'),
                 plotOutput('point_gao_tele'),
                 h2('Norm_Density',align='center'),
                 plotOutput('point_gaodensity_tele'),
                 h2('Norm_log',align='center'),
                 plotOutput('point_gaolog_tele'),
                 h2('Norm_log(y+1)',align='center'),
                 verbatimTextOutput('point_gao_tele_log'),
                 h2('Liner',align='center'),
                 plotOutput('point_lin_tele'),
                 h2('Var',align='center'),
                 verbatimTextOutput('point_var_tele'),
                 h2('Independence',align='center'),
                 verbatimTextOutput('point_den_tele')
                 ),
        tabPanel('Outlier_Fir',
                 h2('general',align='center'),
                 plotOutput('tele_gen'),
                 h2('outlier',align='center'),
                 plotOutput('tele_outlier'),
                 h2('leverage',align='center'),
                 plotOutput('tele_leverage'),
                 h2('influential',align='center'),
                 plotOutput('tele_influential')
        ),
        tabPanel('Regression-Sec','NULL'
                 # h1('Tele-Regression-sec',align='center'),
                 # plotOutput('pointt_regression_tele_sec'),
                 # verbatimTextOutput('summ_tele_regression_sec')
                 ),
        tabPanel("Test_Sec", 'NULL'
                 # h2('General',align='center'),
                 # plotOutput('point_gen_tele_sec'),
                 # h2('Norm',align='center'),
                 # plotOutput('point_gao_tele_sec'),
                 # h2('Liner',align='center'),
                 # plotOutput('point_lin_tele_sec'),
                 # h2('Var',align='center'),
                 # verbatimTextOutput('point_var_tele_sec'),
                 # h2('Independence',align='center'),
                 # verbatimTextOutput('point_den_tele_sec')
        ),
        tabPanel('Outlier_Sec','NULL'
                 # h2('general',align='center'),
                 # plotOutput('tele_gen_sec'),
                 # h2('outlier',align='center'),
                 # plotOutput('tele_outlier_sec'),
                 # h2('leverage',align='center'),
                 # plotOutput('tele_leverage_sec'),
                 # h2('influential',align='center'),
                 # plotOutput('tele_influential_sec')
        )          
      )
    )),
    tabPanel('HW6-classify-credit',
             sidebarPanel(
               conditionalPanel(
                 'input.credit === "Data"',
                 h4("credit.csv"),
                 downloadButton("credit.csv", "Download"),
                 h4("settings:"),
                 sliderInput('prop0','train/all:',min=0,max=1,value=0.7)
               ),
               conditionalPanel(
                 'input.credit === "Logit/Probit"',
                 h4("settings:"),
                 tags$hr(),
                 sliderInput('prop1','train/all:',min=0,max=1,value=0.7),
                 checkboxInput("random1", "random seed:", TRUE),
                 tags$hr(),
                 h4("type"),
                 checkboxGroupInput('whatever1', 'type:',
                                    c('logit-summary','probit-summary','logit-result','probit-result'), selected = c('logit-summary','probit-summary','logit-result','probit-result'))
               ),
               conditionalPanel(
                 'input.credit === "Tree"',
                 h4("settings:"),
                 tags$hr(),
                 sliderInput('prop2','train/all:',min=0,max=1,value=0.7),
                 checkboxInput("random2", "random seed:", TRUE),
                 tags$hr(),
                 h4("type"),
                 checkboxGroupInput('whatever2', 'type:',
                                    c('class-tree','deduce-tree'), selected = c('class-tree','deduce-tree'))
               ),
               conditionalPanel(
                 'input.credit === "Random_Forest"',
                 h4("settings:"),
                 tags$hr(),
                 sliderInput('prop3','train/all:',min=0,max=1,value=0.7),
                 checkboxInput("random3", "random seed:", TRUE)),
               conditionalPanel(
                 'input.credit === "SVM"',
                 h4("settings:"),
                 tags$hr(),
                 sliderInput('prop4','train/all:',min=0,max=1,value=0.7),
                 checkboxInput("random4", "random seed:", TRUE)),
               conditionalPanel(
                 'input.credit === "Evaluate"',
                 h4("show:"),
                 tags$hr(),
                 sliderInput('prop5','train/all:',min=0,max=1,value=0.7),
                 checkboxGroupInput('whatever3', 'index:',
                                    c('sensitivity(TPR)','specificity(1-FPR)','Positive Predictive Value','Negative Predictive Value','Accuracy'), selected = c('sensitivity(TPR)','specificity(1-FPR)','Positive Predictive Value','Negative Predictive Value','Accuracy'))
               )),
             mainPanel(
               tabsetPanel(
                 id = 'credit',
                 tabPanel("Data", 
                          h2('Training-Data-Summary',align='center'),
                          verbatimTextOutput('summary_training'),
                          h2('Traning-Data',align='center'),
                          DT::dataTableOutput("table_training"),
                          tags$hr(),
                          h2('Validate-Data-Summary',align='center'),
                          verbatimTextOutput('summary_validate'),
                          h2('Validate-Data',align='center'),
                          DT::dataTableOutput("table_validate")
                 ),
                 tabPanel("Logit/Probit", 
                          h3('Logit-Summary',align='center'),
                          verbatimTextOutput('logit_summary'),
                          h3('Probit-Summary',align='center'),
                          verbatimTextOutput('probit_summary'),
                          tags$hr(),
                          h3('Logit-Result',align='center'),
                          DT::dataTableOutput('logit_result'),
                          h3('Probit-Result',align='center'),
                          DT::dataTableOutput('probit_result'),
                          tags$hr(),
                          h3('Evaluate',align='center'),
                          verbatimTextOutput('regression_evaluate')
                 ),
                 tabPanel("Dtree&Ctree",
                          h3('Decision-Cptable',align='center'),
                          DT::dataTableOutput('decision_cptable'),
                          h3('Decision_Selection',align='center'),
                          plotOutput('decision_selection'),
                          h3('Decision-Result-Plot',align='center'),
                          plotOutput('decision_result_plot'),
                          h3('decision-Result',align='center'),
                          DT::dataTableOutput('decision_result'),
                          tags$hr(),
                          h3('Condition-Result-Plot',align='center'),
                          plotOutput('condition_result_plot'),
                          h3('Condition-Result',align='center'),
                          DT::dataTableOutput('condition_result'),
                          tags$hr(),
                          h3('Evaluate:deci//condi',align='center'),
                          verbatimTextOutput('decision_evaluate'),
                          verbatimTextOutput('condition_evaluate')
                 ),
                 tabPanel('Random_Forest',
                          h3('Forest-Summary',align='center'),
                          verbatimTextOutput('forest_summary'),
                          h3('Forest-Importance',align='center'),
                          DT::dataTableOutput('forest_importance'),
                          h3('Forest-Result',align='center'),
                          DT::dataTableOutput('forest_result'),
                          h3('Evaluate',align='center'),
                          verbatimTextOutput('forest_evaluate')
                 ),
                 tabPanel('SVM',
                          h3('SVM-Summary',align='center'),
                          verbatimTextOutput('svm_summary'),
                          h3('SVM-Result',align='center'),
                          DT::dataTableOutput('svm_result'),
                          h3('Evaluate',align='center'),
                          verbatimTextOutput('svm_evaluate')
                 ),
                 tabPanel('KNN',
                          h3('KNN-Summary',align='center'),
                          verbatimTextOutput('knn_summary'),
                          h3('KNN-Result',align='center'),
                          DT::dataTableOutput('knn_result'),
                          h3('Evaluate',align='center'),
                          verbatimTextOutput('knn_evaluate')
                 ),
                 tabPanel('NaiveBayes',
                          h3('NaiveBayes-Summary',align='center'),
                          verbatimTextOutput('nb_summary'),
                          h3('NaiveBayes-Result',align='center'),
                          DT::dataTableOutput('nb_result'),
                          h3('Evaluate',align='center'),
                          verbatimTextOutput('nb_evaluate')
                 ),
                 tabPanel('Boosting',
                          h3('Boosting-Summary-20',align='center'),
                          verbatimTextOutput('boosting_summary'),
                          h3('AdaBoost error vs number of trees-100',align='center'),
                          plotOutput('boosting_errorhun'),
                          h3('AdaBoost error vs number of trees-20',align='center'),
                          plotOutput('error_boosting'),
                          h3('Importance-20',align='center'),
                          plotOutput('importance_boosting'),
                          h3('Boosting-Result-100',align='center'),
                          DT::dataTableOutput('boosting_result'),
                          h3('Evaluate-100',align='center'),
                          verbatimTextOutput('boosting_evaluate')
                 ),
                 tabPanel('Bagging',
                            h3('Bagging-Summary-25',align='center'),
                            verbatimTextOutput('bagging_summary'),
                            h3('Bagging error vs number of trees-100',align='center'),
                            plotOutput('bagging_errorhun'),
                            h3('Bagging error vs number of trees-25',align='center'),
                            plotOutput('error_bagging'),
                            h3('Importance-25',align='center'),
                            plotOutput('importance_bagging'),
                            h3('Bagging-Result-100',align='center'),
                            DT::dataTableOutput('bagging_result'),
                            h3('Evaluate-100',align='center'),
                            verbatimTextOutput('bagging_evaluate')
                 ),
                 tabPanel('Neural_Network',
                          h3('Loss',align='center'),
                          plotOutput('network_loss'),
                          h3('Graph',align='center'),
                          plotOutput('network_graph'),
                          h3('Gradient', align='center'),
                          plotOutput('network_gradient')
                 ),
                 tabPanel('All',
                          plotOutput('machine_learning',inline=T)
                 ),
                 tabPanel('Evaluate',
                          h2('Comparison',align='center'),
                          DT::dataTableOutput('comparison'),
                          h2('Comparison-Plot',align='center'),
                          plotOutput('comparison_plot')
                 ))
             )
             
    ),
    tabPanel('HW7-order-satisfaction',
               sidebarPanel(
                 conditionalPanel(
                   'input.satisfaction === "Data"',
                   h4("satisfaction_train.csv"),
                   downloadButton("satisfaction_train.csv", "Download"),
                   h4("satisfaction_validate.csv"),
                   downloadButton("satisfaction_validate.csv", "Download")
                 ),
                 conditionalPanel(
                   'input.satisfaction === "Variable"',
                   h4("method"),
                   checkboxGroupInput('whatever1', 'method:',
                                      c('step(AIC)','anova'), selected = c('step(AIC)','anova'))
                 ),
                 conditionalPanel(
                   'input.satisfaction === "Result"',
                   checkboxGroupInput('whatever1', 'method:',
                                      c('step(AIC)','anova'), selected = c('step(AIC)','anova'))
                 )
               ),
               mainPanel(
                 tabsetPanel(
                   id = 'satisfaction',
                   tabPanel("Data", 
                            h2('Training-Data-Summary',align='center'),
                            verbatimTextOutput('summary_training_order'),
                            h2('Traning-Data',align='center'),
                            DT::dataTableOutput("table_training_order"),
                            tags$hr(),
                            h2('Validate-Data-Summary',align='center'),
                            verbatimTextOutput('summary_validate_order'),
                            h2('Validate-Data',align='center'),
                            DT::dataTableOutput("table_validate_order")
                   ),
                   tabPanel("Variable", 
                            h3('All-Summary',align='center'),
                            verbatimTextOutput('all_variable_summary'),
                            h3('Anova-Summary',align='center'),
                            verbatimTextOutput('anova_summary'),
                            h3('After-Anova-Summary',align='center'),
                            verbatimTextOutput('after_anova_summary'),
                            tags$hr(),
                            h3('Step(AIC)-Summary',align='center'),
                            verbatimTextOutput('aic_summary')
                   ),
                   tabPanel("Result",
                            h3('Anova-Result',align='center'),
                            verbatimTextOutput('anova_result'),
                            h3('AIC-Result',align='center'),
                            verbatimTextOutput('aic_result')
                   )
                 )
               )
    ),
    tabPanel('HW8-possion-search',
             sidebarPanel(
               conditionalPanel(
                 'input.search === "Data"',
                 h4("search.csv"),
                 downloadButton("search.csv", "Download")
               ),
               conditionalPanel(
                 'input.search === "Variable"',
                 h4("method"),
                 checkboxGroupInput('whatever1', 'method:',
                                    c('step(AIC)','anova'), selected = c('step(AIC)','anova'))
               )
             ),
             mainPanel(
               tabsetPanel(
                 id = 'search',
                 tabPanel("Data", 
                          h2('Data-Summary',align='center'),
                          verbatimTextOutput('summary_search'),
                          h2('Data',align='center'),
                          DT::dataTableOutput("table_search")
                 ),
                 tabPanel("Variable", 
                          h3('All-Summary',align='center'),
                          verbatimTextOutput('all_variable_summary_search'),
                          h3('Anova-Summary',align='center'),
                          verbatimTextOutput('anova_summary_search'),
                          h3('After-Anova-Summary',align='center'),
                          verbatimTextOutput('after_anova_summary_search'),
                          tags$hr(),
                          h3('Step(AIC)-Summary',align='center'),
                          verbatimTextOutput('aic_summary_search')
                 ),
                 tabPanel("Result", 
                          h3('Test',align='center'),
                          verbatimTextOutput('qcc_search'),
                          tags$hr(),
                          tags$hr(),
                          h3('None-EXP',align='center'),
                          verbatimTextOutput('none_expp'),
                          h3('EXP',align='center'),
                          verbatimTextOutput('expp'),
                          tags$hr(),
                          tags$hr(),
                          h2('Thinking',align='center'),
                          h4('0',align='center'),
                          h4('prediction',align='center')
                 )
               )
             )
    )
    
    
  )
  
)

server <- function(input, output){
  # cp_height 分页1
  output$cp_height.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("cp_height", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(cp_height, file, row.names = FALSE)
    }
  )
  # cp_height--data部分 分页1标签1
  output$table_height <- renderDataTable({    # cp_height数据表格展示
    datatable(cp_height)
  })
  output$summary <- renderPrint({summary(cp_height)})  # cp_height数据描述
  # cp_height--cor&cov部分 分页1标签2
  output$newcor <- renderText({paste('Cor_new:',cor(male*input$times1*input$times2+input$plus1+input$plus2,female_new <- female*input$times1+input$plus1))})            # 新的cor
  output$newcov <- renderText({paste('Cov_new4:',cov(male*input$times1*input$times2+input$plus1+input$plus2,female_new <- female*input$times1+input$plus1))})            # 新的cov
  output$di <- renderText({cov(male*input$times1*input$times2+input$plus1+input$plus2,female_new <- female*input$times1+input$plus1 )/covv})           # 倍数
  output$unitable <- renderTable({unitable})                         # 单位换算表
  # cp_height--regression部分 分页1标签3
  output$pointt <- renderPlot({ggplot(data=cp_height, aes(x=female, y=male)) +   # cp_height散点图与拟合图
                                        geom_smooth(method = lm) + 
                                        geom_point()})
  output$summfit <- renderPrint({summary(fit2.10)})                              # cp_height拟合总结summary(fit2.10)
  # cp_height--test部分 分页1标签3
  output$point_gen <- renderPlot({par(mfrow=c(2,2))                           # 一般检验plot(fit2.10)
                                  plot(fit2.10)})
  output$point_lin <- renderPlot({crPlots(fit2.10)})                          # 线性检验
  output$point_gao <- renderPlot({qqPlot(fit2.10)})                           # 正态检验
  output$point_var <- renderPrint({ncvTest(fit2.10)})                         # 同方差检验
  output$point_den <- renderPrint({durbinWatsonTest(fit2.10)})                # 独立性检验
  # cp_height--thinking部分 分页1标签5
  output$point_all <- renderPlot({plot(regsubsets(male~female+I(female^2)+I(female^3)+I(sqrt(female)),data=cp_height,nbest=3), scale="adjr2")})
  # cp_height--report部分 分页1标签6
  output$cp_report <- renderImage(list(src='210212.png'),deleteFile=F)
  
  # newspaper 分页2
  output$newspaper.csv <- downloadHandler(       # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("newspaper", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(newspaper, file, row.names = FALSE)
    }
  )
  # newspaper--data部分 分页2标签1
  output$table_new <- renderDataTable({                       # newspaper数据表格展示
    datatable(newspaper)
  })
  output$summary_new <- renderPrint({summary(newspaper)})    # newspaper数据描述
  # newspaper--regression部分 分页2标签2
  output$pointt_new <- renderPlot({ggplot(data=newspaper, aes(x=ord, y=sun)) +  # newspaper散点图与拟合图
                                        geom_smooth(method = lm) + 
                                        geom_point()})
  output$summfit_new <- renderPrint({summary(fit2.12)})                         # newspaper拟合总结summary(fit2.12)
  output$confint <- renderPrint({confint(fit2.12)})                             # newspaper系数置信区间
  # newspaper--predict部分 分页2标签3
  output$predict_one <- renderPrint({predict(fit2.12, data.frame(ord=500), interval="confidence")})
  output$predict_two <- renderPrint({predict(fit2.12, data.frame(ord=500), interval="prediction")})
  output$predict_three <- renderPrint({predict(fit2.12, data.frame(ord=2000), interval="prediction")})
  # newspaper--test部分 分页2标签4
  output$point_gen_new <- renderPlot({par(mfrow=c(2,2))
                                  plot(fit2.12)})
  output$point_lin_new <- renderPlot({crPlots(fit2.12)})                    # 线性检验
  output$point_gao_new <- renderPlot({qqPlot(fit2.12)})                         # 正态检验
  output$point_var_new <- renderPrint({ncvTest(fit2.12)})                   # 同方差检验
  output$point_den_new <- renderPrint({durbinWatsonTest(fit2.12)})          # 独立性检验
  # newspaper--thinking部分 分页2标签5
  output$point_all_new <- renderPlot({plot(regsubsets(sun~ord+I(ord^2)+I(ord^3)+I(sqrt(ord)),data=newspaper,nbest=3), scale="adjr2")})
  # cigarette 分页3
  output$cigarette.csv <- downloadHandler(       # cigarette数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("cigarette", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(cigarette, file, row.names = FALSE)
    }
  )
  # cigarette--data部分 分页3标签1
  output$table_cig <- renderDataTable({                       # cigarette数据表格展示
    datatable(cigarette)
  })
  output$summary_cig <- renderPrint({summary(cigarette)})
  # cigarette--regression部分 分页3标签2
  output$pointt_cig_org <- renderPlot({ggplot(data=cigarette, aes(x=seq(1,length(sales)), y=sales)) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=cig, aes(x=seq(1,length(sales)), y=fitted(fit3.15a)),col='red',size=1.2)
  })
  output$pointt_cig_adj <- renderPlot({ggplot(data=cigarette, aes(x=seq(1,length(sales)), y=sort(sales))) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=cig, aes(x=seq(1,length(sales)), y=fitted(fit3.15a)[order(cig$sales)]),col='red',size=1.2)
  })
  output$summ_fe_ratio <- renderPrint({summary(fit3.15a)})
  output$summ_bl_ratio <- renderPrint({summary(fit3.15b)})
  output$summ_income_confint <- renderPrint({confint(fit3.15a)})
  # cigarette--adj.r.square部分 分页3标签3
  output$adj_exc_income <- renderPrint({summary(fit3.15c)})
  output$adj_agepriceincome <- renderPrint({summary(fit3.15d)})
  output$adj_income <- renderPrint({summary(fit3.15e)})
  # cigarette--test部分 分页3标签4
  output$point_gen_cig <- renderPlot({par(mfrow=c(2,2))
                                      plot(fit3.15a)})
  output$point_lin_cig <- renderPlot({crPlots(fit3.15a)})                    # 线性检验
  output$point_gao_cig <- renderPlot({qqPlot(fit3.15a)})                         # 正态检验
  output$point_var_cig <- renderPrint({ncvTest(fit3.15a)})                   # 同方差检验
  output$point_den_cig <- renderPrint({durbinWatsonTest(fit3.15a)})          # 独立性检验
  # cigarette--thinking部分 分页3标签5
  output$point_all_cig <- renderPlot({plot(regsubsets(sales~.,data=cigarette,nbest=6), scale="adjr2")})
  # model 分页4
  output$model.csv <- downloadHandler(       # cigarette数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("model", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(model, file, row.names = FALSE)
    }
  )
  # model--data部分 分页4标签1
  output$table_mod <- renderDataTable({                       # cigarette数据表格展示
    datatable(mod)
  })
  output$summary_mod <- renderPrint({summary(mod)})
  # model--regression_all部分 分页4标签2
  output$pointt_mod <- renderPlot({ggplot(data=mod, aes(x=seq(1,length(Y)), y=sort(Y))) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=mod, aes(x=seq(1,length(mod$Y)), y=fitted(fit4all)[order(mod$Y)]),col='red',size=1.2)
    })
  output$summfit_mod <- renderPrint({summary(fit4all)})
  # model--test部分 分页4标签3
  output$point_gen_mod <- renderPlot({par(mfrow=c(2,2))
                                      plot(fit4all)})
  output$point_lin_mod <- renderPlot({crPlots(fit4all)})                    # 线性检验
  output$point_gao_mod <- renderPlot({qqPlot(fit4all)})                         # 正态检验
  output$point_var_mod <- renderPrint({ncvTest(fit4all)})                   # 同方差检验
  output$point_den_mod <- renderPrint({durbinWatsonTest(fit4all)})          # 独立性检验
  # model--Outlier_Plot部分 分页4标签4
  
  output$mod_residual <- renderDataTable({datatable(as.data.frame(fit4all$residuals))})
  output$mod_resid_plot <- renderPlot({plot(fit4all,which=1)})
  output$mod_Cook <- renderDataTable({datatable(as.data.frame(cooks.distance(fit4all)))})
  output$mod_Cook_plot <- renderPlot({plot(fit4all, which=4, cook.levels=4/(mod_n-mod_p-2))
                                      abline(h=4/(mod_n-mod_p-2))})
  output$mod_DFFITS <- renderDataTable({datatable(as.data.frame(dffits(fit4all)))})
  output$mod_DFFITS_plot <- renderPlot({plot(dffits(fit4all))
                                        })
  output$mod_hatvalue <- renderDataTable({datatable(as.data.frame(hatvalues(fit4all)))})
  output$mod_hatvalue_plot <- renderPlot({plot(hatvalues(fit4all))
                                          abline(h=c(2,3)*mod_p/mod_n)})
  output$mod_scale <- renderPlot({plot(fit4all,which=3)})
  # model--Outlier部分 分页4标签5
  output$mod_gen <- renderPlot({influencePlot(fit4all)})
  output$mod_outlier <- renderPlot({qqPlot(fit4all)})
  output$mod_leverage <- renderPlot({plot(hatvalues(fit4all))
                                     abline(h=c(2,3)*mod_p/mod_n)})
  output$mod_influential <- renderPlot({plot(dffits(fit4all))
                                        abline(h=2*sqrt((mod_p+1)/mod_n))})
  # model--Regression部分 分页4标签6
  output$mod_all <- renderPlot({plot(regsubsets(Y~.,data=mod,nbest=6), scale="adjr2")})
  
  # cigmore部分 分页5
  # cigmore--All部分 分页5标签1
  output$pointt_all_cig <- renderPlot({plot(regsubsets(sales~.,data=cigarette,nbest=6), scale="adjr2")})
  # cigmore--Regression_Fir部分 分页5标签2
  output$pointt_regression_cig_fir <- renderPlot({ggplot(data=cigarette, aes(x=seq(1,length(sales)), y=sort(sales))) +  # newspaper散点图与拟合图
                                                geom_line(col='blue',size=1.2)+
                                                geom_line(data=cig, aes(x=seq(1,length(sales)), y=fitted(fit3.15d)[order(cig$sales)]),col='red',size=1.2)
  })
  output$summ_cig_regression_fir <- renderPrint({summary(fit3.15d)})
  # cigmore--Test部分 分页5标签3
  output$point_gen_cigmore <- renderPlot({par(mfrow=c(2,2))
    plot(fit3.15d)})
  output$point_lin_cigmore <- renderPlot({crPlots(fit3.15d)})                    # 线性检验
  output$point_gao_cigmore <- renderPlot({qqPlot(fit3.15d)})                         # 正态检验
  output$point_var_cigmore <- renderPrint({ncvTest(fit3.15d)})                   # 同方差检验
  output$point_den_cigmore <- renderPrint({durbinWatsonTest(fit3.15d)})          # 独立性检验
  # cigmore--oulier部分 分页5标签4
  output$cig_gen <- renderPlot({influencePlot(fit3.15d)})
  output$cig_outlier <- renderPlot({qqPlot(fit3.15d)})
  output$cig_leverage <- renderPlot({plot(hatvalues(fit3.15d))
                                     abline(h=c(2,3)*cig_p/cig_n)})
  output$cig_influential <- renderPlot({plot(dffits(fit3.15d))
                                        abline(h=2*sqrt((cig_p+1)/cig_n))})
  # cigmore--Regression_Sec部分 分页5标签5
  output$pointt_regression_cig_sec <- renderPlot({ggplot(data=cigar, aes(x=seq(1,length(sales)), y=sort(sales))) +  # newspaper散点图与拟合图
                                                      geom_line(col='blue',size=1.2)+
                                                      geom_line(data=cigar, aes(x=seq(1,length(cigar$sales)), y=fitted(fit3.15op)[order(cigar$sales)]),col='red',size=1.2)})
  output$summ_cig_regression_sec <- renderPrint({summary(fit3.15op)})
  # cigmore--Test_Sec部分 分页5标签6
  output$point_gen_cigmore_sec <- renderPlot({par(mfrow=c(2,2))
    plot(fit3.15op)})
  output$point_lin_cigmore_sec <- renderPlot({crPlots(fit3.15op)})                    # 线性检验
  output$point_gao_cigmore_sec <- renderPlot({qqPlot(fit3.15op)})                         # 正态检验
  output$point_var_cigmore_sec <- renderPrint({ncvTest(fit3.15op)})                   # 同方差检验
  output$point_den_cigmore_sec <- renderPrint({durbinWatsonTest(fit3.15op)})          # 独立性检验
  # cigmore--oulier_Sec部分 分页5标签7
  output$cig_gen_sec <- renderPlot({influencePlot(fit3.15op)})
  output$cig_outlier_sec <- renderPlot({qqPlot(fit3.15op)})
  output$cig_leverage_sec <- renderPlot({plot(hatvalues(fit3.15op))
    abline(h=c(2,3)*cig_p/(cig_n-3))})
  output$cig_influential_sec <- renderPlot({plot(dffits(fit3.15op))
    abline(h=2*sqrt((cig_p+1)/(cig_n-3)))})
  
  # cigmore--Regression_thir部分 分页5标签8
  output$pointt_regression_cig_thir <- renderPlot({ggplot(data=cigare, aes(x=seq(1,length(sales)), y=sort(sales))) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=cigare, aes(x=seq(1,length(cigare$sales)), y=fitted(fit3.15opp)[order(cigare$sales)]),col='red',size=1.2)})
  output$summ_cig_regression_thir <- renderPrint({summary(fit3.15opp)})
  # cigmore--Test_thir部分 分页5标签9
  output$point_gen_cigmore_thir <- renderPlot({par(mfrow=c(2,2))
    plot(fit3.15opp)})
  output$point_lin_cigmore_thir <- renderPlot({crPlots(fit3.15opp)})                    # 线性检验
  output$point_gao_cigmore_thir <- renderPlot({qqPlot(fit3.15opp)})                         # 正态检验
  output$point_var_cigmore_thir <- renderPrint({ncvTest(fit3.15opp)})                   # 同方差检验
  output$point_den_cigmore_thir <- renderPrint({durbinWatsonTest(fit3.15opp)})          # 独立性检验
  # cigmore--oulier_thir部分 分页5标签10
  output$cig_gen_thir <- renderPlot({influencePlot(fit3.15opp)})
  output$cig_outlier_thir <- renderPlot({qqPlot(fit3.15opp)})
  output$cig_leverage_thir <- renderPlot({plot(hatvalues(fit3.15opp))
    abline(h=c(2,3)*cig_p/(cig_n-6))})
  output$cig_influential_thir <- renderPlot({plot(dffits(fit3.15opp))
    abline(h=2*sqrt((cig_p+1)/(cig_n-6)))})
  # cigmore--Regression_thir部分 分页5标签8
  output$pointt_regression_cig_thir_add <- renderPlot({ggplot(data=cigare, aes(x=seq(1,length(sales)), y=sort(sales))) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=cigare, aes(x=seq(1,length(cigare$sales)), y=fitted(fit3.15oppp)[order(cigare$sales)]),col='red',size=1.2)})
  output$summ_cig_regression_thir_add <- renderPrint({summary(fit3.15oppp)})
  # cigmore--Test_thir_add部分 分页5标签9
  output$point_gen_cigmore_thir_add <- renderPlot({par(mfrow=c(2,2))
    plot(fit3.15opp)})
  output$point_lin_cigmore_thir_add <- renderPlot({crPlots(fit3.15oppp)})                    # 线性检验
  output$point_gao_cigmore_thir_add <- renderPlot({qqPlot(fit3.15oppp)})                         # 正态检验
  output$point_var_cigmore_thir_add <- renderPrint({ncvTest(fit3.15oppp)})                   # 同方差检验
  output$point_den_cigmore_thir_add <- renderPrint({durbinWatsonTest(fit3.15oppp)})          # 独立性检验
  # cigmore--oulier_thir_add部分 分页5标签10
  output$cig_gen_thir_add <- renderPlot({influencePlot(fit3.15oppp)})
  output$cig_outlier_thir_add <- renderPlot({qqPlot(fit3.15oppp)})
  output$cig_leverage_thir_add <- renderPlot({plot(hatvalues(fit3.15oppp))
    abline(h=c(2,3)*cig_p/(cig_n-6))})
  output$cig_influential_thir_add <- renderPlot({plot(dffits(fit3.15oppp))
    abline(h=2*sqrt((cig_p+1)/(cig_n-6)))})
  # cigmore--report部分 分页5标签11
  output$cig_report <- renderImage(list(src='118.png'),deleteFile=F)
  # collinearity部分 分页6
  output$collinearity.csv <- downloadHandler(       # cigarette数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("collinearity", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(mll, file, row.names = FALSE)
    }
  )
  # collinearity--Data部分 分页6标签1
  output$table_coll <- renderDataTable({                       # cigarette数据表格展示
    datatable(mll)
  })
  output$summary_coll <- renderPrint({summary(mll)})
  # Collinearity--Collinearity部分 分页6标签2
  output$coll_vif <- renderPrint({round(vif(fit10.7),2)})
  output$coll_corrtest <- renderPrint({corr.test(data[2:7])})
  output$coll_mull_one <- renderPlot({ggplot(data=mll,aes(x=X1,y=X6))+geom_point()+geom_smooth(method = lm)})
  output$coll_mull_two <- renderPlot({ggplot(data=mll,aes(x=X2,y=X5))+geom_point()+geom_smooth(method = lm)})
  output$coll_mull_three <- renderPlot({ggplot(data=mll,aes(x=X3,y=X4))+geom_point()+geom_smooth(method = lm)})
  output$coll_mull_four <- renderPlot({ggplot(data=mll,aes(x=X4,y=X5))+geom_point()+geom_smooth(method = lm)})
  # Collinearity--PCA部分 分页6标签3
  output$coll_all_cps <- renderDataTable({    # cp_height数据表格展示
    datatable(predict(fit10.7pc))
  })
  output$coll_summcps <- renderPrint({summary(fit10.7op)})
  output$coll_two_cps <- renderPlot({ggplot(data=main,aes(y=main2,x=main1)) + 
                                        geom_point() })
  output$coll_two_cps_summ <- renderPrint({summary(fit10.7main2)})
  output$coll_Scree_Plot <- renderPlot({fa.parallel(mll, fa='pc', n.iter=100, show.legend=FALSE)})
  # Collinearity--Regression-PCA部分 分页6标签4
  output$coll_cps_eigenavector <- renderPrint({coll_eigen})
  output$coll_cps_beta <- renderPrint({beta})
  output$coll_cps_sd <- renderPrint({x.sd})
  output$coll_cps_conclusion <- renderPrint({coll_cps_conclusion})
  # Collinearity--Regression-Ridge部分 分页6标签5
  output$coll_ridge <- renderPlot({plot(fit10.7ri)
                                    abline(v=.3, col='red',lwd=1.5)
                                    abline(v=.2237086,col='green',lwd=1.5)
                                    abline(v=.5891268,col='blue',lwd=1.5)
                                    legend('topright',legend = c('HKB','L-W','GCV'), col=c('green','blue','red'), lwd=1.5)})
  output$coll_ridge_lambda <- renderPrint({select(fit10.7ri)})
  output$coll_ridge_est <- renderPrint({fit10.7pre})
  output$coll_ridge_mean_sd <- renderPrint({coll_mean_sd})
  
  # report部分 分页7
  output$telephone.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("telephone", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(telephone, file, row.names = FALSE)
    }
  )
  # report--Data部分 分页7标签1
  output$table_tele <- renderDataTable({                       # cigarette数据表格展示
    datatable(telephone)
  })
  output$summary_tele <- renderPrint({summary(telephone)})
  # report--Regression-Fir部分 分页7标签2
  output$pointt_regression_tele_test <- renderPlot({plot(regsubsets(prodiff~.,data=telephone,nbest=3), scale="adjr2")})
  output$pointt_regression_tele_fir <- renderPlot({ggplot(data=telephone, aes(x=seq(1,length(prodiff)), y=sort(prodiff))) +  # newspaper散点图与拟合图
      geom_line(col='blue',size=1.2)+
      geom_line(data=telephone, aes(x=seq(1,length(prodiff)), y=fitted(fit.rep)[order(telephone$prodiff)]),col='red',size=1.2)
  })
  output$summ_tele_regression_fir <- renderPrint({summary(fit.rep)})
  # report--Test-Fir部分 分页7标签3
  output$point_gen_tele <- renderPlot({par(mfrow=c(2,2))
    plot(fit.rep)})
  output$point_lin_tele <- renderPlot({crPlots(fit.rep)})                    # 线性检验
  output$point_gao_tele <- renderPlot({qqPlot(fit.rep)})                         # 正态检验
  output$point_gaodensity_tele <- renderPlot({residplot(fit.rep)})
  output$point_gao_tele_log <- renderPrint({summary(powerTransform(telephone$prodiff+1))})
  output$point_gaolog_tele <- renderPlot(qqPlot(fit.replog))
  output$point_var_tele <- renderPrint({ncvTest(fit.rep)})                   # 同方差检验
  output$point_den_tele <- renderPrint({durbinWatsonTest(fit.rep)})
  
  # report--Outlier-Fir部分 分页7标签4
  output$tele_gen <- renderPlot({influencePlot(fit.rep)})
  output$tele_outlier <- renderPlot({qqPlot(fit.rep)})
  output$tele_leverage <- renderPlot({plot(hatvalues(fit.rep))
    abline(h=c(2,3)*tel_p/tel_n)})
  output$tele_influential <- renderPlot({plot(dffits(fit.rep))
    abline(h=2*sqrt((tel_p+1)/tel_n))})
  # report--Regression-Sec部分 分页7标签5
  
  # report--Test_Sec部分 分页7标签6
  # report--Outlier_Sec部分 分页7标签7
  # credit部分 分页8
  output$credit.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("credit", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(credit, file, row.names = FALSE)
    }
  )
  # credit--Data部分 分页8标签1
  output$summary_training <- renderPrint({summary(credit.train)})
  output$table_training <- renderDataTable({                       # cigarette数据表格展示
    datatable(credit.train)
  })
  output$summary_validate <- renderPrint({summary(credit.validate)})
  output$table_validate <- renderDataTable({                       # cigarette数据表格展示
    datatable(credit.validate)
  })
  # credit--Logit/Probit部分 分页8标签2
  output$logit_summary <- renderPrint({summary(fit.logit)})
  output$probit_summary <- renderPrint({summary(fit.probit)})
  output$logit_result <- renderDataTable({datatable(logit.perf)})
  output$probit_result <- renderDataTable({datatable(probit.perf)})
  output$regression_evaluate <- renderPrint({perform(logit.perf)})
  # ???????
  # credit--Tree部分 分页8标签3
  output$decision_cptable <- renderDataTable({datatable(dtree$cptable)})
  output$decision_selection <- renderPlot({plotcp(dtree)})
  output$decision_result_plot <- renderPlot({prp(dtree.pruned, type=2, extra=104,fallen.leaves=T)})
  output$decision_result <- renderDataTable({datatable(dtree.perf)})
  
  output$condition_result_plot <- renderPlot({plot(fit.ctree)})
  output$condition_result <- renderDataTable({datatable(ctree.perf)})
  output$decision_evaluate <- renderPrint({perform(dtree.perf)})
  output$condition_evaluate <- renderPrint({perform(ctree.perf)})
  # ???????
  # credit--Random_Forest部分 分页8标签4
  output$forest_summary <- renderPrint({fit.forest})
  output$forest_importance <- renderDataTable({datatable(importance(fit.forest, type=2))})
  # ???importance表格
  output$forest_result <- renderDataTable({datatable(forest.perf)})
  output$forest_evaluate <- renderPrint({perform(forest.perf)})
  # credit--SVM部分 分页8标签5
  output$svm_summary <- renderPrint({summary(fit.svm)})
  output$svm_result <- renderDataTable({datatable(svm.perf)})
  output$svm_evaluate <- renderPrint({perform(svm.perf)})
  # credit--knn部分 分页8标签7
  output$knn_summary <- renderPrint({summary(fit.knn)})
  output$knn_result <- renderDataTable({datatable(knn.perf)})
  output$knn_evaluate <- renderPrint({perform(knn.perf)})
  # credit--naiveBayes部分 分页8标签8
  output$nb_summary <- renderPrint({summary(fit.nb)})
  output$nb_result <- renderDataTable({datatable(nb.perf)})
  output$nb_evaluate <- renderPrint({perform(nb.perf)})
  # credit--boosting部分 分页8标签9
  output$boosting_summary <- renderPrint({fit.boosting})
  output$boosting_errorhun <- renderImage(list(src='boostinghun.png'),deleteFile = F)
  output$error_boosting <- renderPlot({plot(error.boosting$error,type="l")})
  output$importance_boosting <- renderPlot({barplot(fit.boosting$importance)})
  output$boosting_result <- renderDataTable({datatable(boosting.perf)})
  output$boosting_evaluate <- renderPrint({perform(boosting.perf)})
  # credit--bagging部分 分页8标签10
  output$bagging_summary <- renderPrint({fit.bagging})
  output$bagging_errorhun <- renderImage(list(src='bagginghun.png'),deleteFile = F)
  output$error_bagging <- renderPlot({plot(error.bagging$error,type="l")})
  output$importance_bagging <- renderPlot({barplot(fit.bagging$importance)})
  output$bagging_result <- renderDataTable({datatable(bagging.perf)})
  output$bagging_evaluate <- renderPrint({perform(bagging.perf)})
  
  # credit--net部分 分页8标签11
  output$network_loss <- renderImage(list(src='net_loss.png'),deleteFile = F)
  output$network_graph <- renderImage(list(src='net_graph.png'),deleteFile = F)
  output$network_gradient <- renderImage(list(src='net_gradient.png'),deleteFile = F)
  # credit--all部分  分页8标签12
  output$machine_learning <- renderImage(list(src='machine_learning.jpg'),deleteFile=F)
  # credit--comparison部分 分页8标签13
  output$comparison <- renderDataTable({datatable(evalute.table)})
  output$comparison_plot <- renderPlot({ggplot(data=new.perfor,aes(x=variable,y=value,linetype=name,color=name, group=name))+
      geom_line(size=1.2) +
      geom_point(size=4)})
  
  # order部分 分页9
  output$satisfaction_train.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("satisfaction_train", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(sa.train, file, row.names = FALSE)
    }
  )
  
  output$satisfaction_validate.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("satisfaction_validate", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(sa.test, file, row.names = FALSE)
    }
  )
  # order--Data部分 分页9标签1
  output$summary_training_order <- renderPrint({summary(sa.train)})
  output$table_training_order <- renderDataTable({                       # cigarette数据表格展示
    datatable(sa.train)
  })
  output$summary_validate_order <- renderPrint({summary(sa.test)})
  output$table_validate_order <- renderDataTable({                       # cigarette数据表格展示
    datatable(sa.test)
  })
  # order--Variable部分 分页9标签2
  output$all_variable_summary <- renderPrint({summary(fit.order)})
  output$anova_summary <- renderPrint({Anova(fit.order,type="III")})
  output$after_anova_summary <- renderPrint({summary(fit.order.anova)})
  output$aic_summary <- renderPrint({summary(fit.order.aic)})
  
  # order--Result部分 分页9标签3
  output$anova_result <- renderPrint({order.perf.anova})
  output$aic_result <- renderPrint({order.perf.aic})                # cigarette数据表格展示
  
  # search部分 分页10
  output$search.csv <- downloadHandler(      # cp_height数据下载按钮，分页1标签1侧边栏
    filename = function() {
      paste("search", ".csv", sep = "")
    },
    content = function(file) {
      write.csv(search, file, row.names = FALSE)
    }
  )
  
  
  # search--Data部分 分页10标签1
  output$summary_search <- renderPrint({summary(search)})
  output$table_search <- renderDataTable({                       # cigarette数据表格展示
    datatable(search)
  })
  
  # search--Variable部分 分页10标签2
  output$all_variable_summary_search <- renderPrint({summary(fit.search)})
  output$anova_summary_search <- renderPrint({Anova(fit.search,type="III")})
  output$after_anova_summary_search <- renderPrint({summary(fit.search.anova)})
  output$aic_summary_search <- renderPrint({summary(fit.search.aic)})
  output$qcc_search <- renderPrint({qcc.overdispersion.test(search$click, type="poisson")})
  # search--Result部分 分页10标签3
  output$none_expp <- renderPrint({coef(fit.search.anova)})
  output$expp <- renderPrint({exp(coef(fit.search.anova))})

  }

shinyApp(ui, server)
