dir=getwd()
print(dir)
setwd(dir)
library(pacman)
pacman::p_load(rpart,rattle,ROSE,gains,randomForest)

# Classifcation model: The goal is to predict the likelihood of the Fed to raise or decrease/unchange the effective fund rate.
# - Model testing: Logit regression (main variables only), logit regression (main var + interaction terms), classification tree, classification forest
# 
# 
# Regression model: The goal is to forecast how much the change in the Fed interest rate is likely to be
# - Modeling testing: Linear regression, regression tree, regression forest.



#Convert a variable into factor
df=read.csv('FFR_Regression.csv',stringsAsFactors = T)
df$Fed.Rate.Hike=as.factor(df$Fed.Rate.Hike)
model_linear=lm(data=df,value~CPI+PPI+INDPRO+PAYROLL)
summary(model_linear)
vif(model_linear)
plot(model_linear$residuals)
round(model_linear$coefficients,3)
predict_linear=predict(data=df,model_linear)
rmse_linear=sqrt(mean((df$value-predict_linear)^2))
rmse_linear
#1.9229



#RMSE is relatively low, which suggest the model has a low error. However, given the objective value is interest rate, 1.9 RMSE seems quite significant 


#Building Logit model
model_logit=glm(data=df,Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL,family = binomial)
summary(model_logit)
round(model_logit$coefficients,3)
predict_logit=predict(data=df,model_logit,type='response')

#Model Performance Evaluation
# 1. ROC curve
roc_logit1=roc.curve(df$Fed.Rate.Hike,predict_logit)
#ROC is 0.609
# 2. Mc Fadden R squared
#Mc Fadden Rsquared
null_model=glm(data=df, Fed.Rate.Hike~1,family='binomial')
fadden_logit1=1-(logLik(model_logit)/logLik(null_model))
fadden_logit1

# Mc Fadden R squared is 0.028 which is considered low

# 3. Confusion matrix (use cut-off point 0.5)
matrix1=table(df$Fed.Rate.Hike, predict_logit>0.5)
matrix1

#Accuracy
accuracy_1=sum(diag(matrix1))/sum(matrix1)

#0.573

#Sensitivity
sensitivty_1=matrix1[2,2]/sum(matrix1[2,])

#0.556

#Specificity
specificity_1=matrix1[1,1]/sum(matrix1[1,])
0.589

#Create  full model to perform stepwise method
df=df[,-1] #get rid of date column

full_model=glm(data=df,Fed.Rate.Hike~(CPI+PPI+PCE+INDPRO+PAYROLL)^2,family='binomial')

step_model=step(full_model,scope=list(upper=full_model,lower=null_model), direction='both',trace=T)

summary(step_model)

predict_step=predict(data=df,step_model,type='response')

#Model Performance for upgraded_logit (stepwise model)

# 1. ROC
roc_logit2=roc.curve(df$Fed.Rate.Hike,predict_step)

# AUC is 0.681 -> signficantly better than the previous one

# 2. Mc Fadden R squared

fadden_step_logit=1-(logLik(step_model)/logLik(null_model))
 0.070

# the former logit model has higher Mc Fadden R squared

# 3. Confusion Matrix 
matrix2= table(df$Fed.Rate.Hike, predict_step>0.5)

#Accuracy
accuracy_logit2=sum(diag(matrix2))/sum(matrix2)
0.629

# More accurate than the original logit model

#Sensitivity
sensitivity_logit2=matrix2[2,2]/sum(matrix2[2,])
0.559

# Just slightly better in predicting the likelihood of rate hike than the former

#Specificity
specificity_logit2=matrix2[1,1]/sum(matrix2[1,])
0.699

# Signficantly better in predicting the likelihood of rate decrease or unchange



#Build a classification tree model to prediction the likelihood of the Fed Hike

model_tree=rpart(data=df,Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL,method='class')
summary(model_tree)

fancyRpartPlot(model_tree)

#build a complex tree
complex_class_tree= rpart(data=df,Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL,method='class',control = rpart.control(cp=0.000001))
plotcp(complex_class_tree)

cp.optimal=complex_class_tree$cptable[which.min(complex_class_tree$cptable[,'xerror']),'CP']


# Now use the optimal cp to prune the tree

model_tree_pruned=prune(complex_class_tree,cp=cp.optimal)
fancyRpartPlot(model_tree_pruned)
predict_class_tree=predict(data=df, model_tree_pruned)

#Performance evaluation 

matrix3=table(df$Fed.Rate.Hike, predict_class_tree[,2]>0.5)

accuracy_class_tree=sum(diag(matrix3))/sum(matrix3)

sensitivity_class_tree=matrix3[2,2]/sum(matrix3[2,])
specificity_class_tree=matrix3[1,1]/sum(matrix3[1,])

roc__class_tree=roc.curve(df$Fed.Rate.Hike,predict_class_tree[,2])
#AUC 0.0081

model_tree_pruned$variable.importance #in decision tree, PCE is significant

gains(as.numeric(df$Fed.Rate.Hike),predict_class_tree[,2])

#Build a regression tree model to predict the interest rate

model_reg_tree=rpart(data=df, value~CPI+PPI+PCE+INDPRO+PAYROLL,method='anova')
summary(model_reg_tree)
fancyRpartPlot(model_reg_tree)

#Build a complex tree for regression tree
complex_regression_tree= rpart(data=df, value~CPI+PPI+PCE+INDPRO+PAYROLL,method='anova',control = rpart.control(cp=0.000001))

fancyRpartPlot(complex_regression_tree)
plotcp(complex_regression_tree)
complex_regression_tree$cptable
optimal_cp2=complex_regression_tree$cptable[which.min(complex_regression_tree$cptable[,'xerror']),'CP']

#Now use the optimal cp to prune the tree

model_reg_tree=prune(complex_regression_tree,cp=optimal_cp2)
fancyRpartPlot(model_reg_tree)
summary(model_reg_tree)

#Now we can use the regression tree to make a prediction
predict_reg_tree=predict(data=df$value,model_reg_tree)

#Regression tree Performnace  evaluation

rmse_reg_tree=sqrt(mean((df$value-predict_reg_tree)^2))

rsq_reg_tree=1-(sum((df$value-predict_reg_tree)^2)/sum((df$value-mean(df$value))^2))    
rsq_reg_tree

# Forecasting error score is quite low, significantly lower than linear model

#Build random forest for classification
model_forest_class= randomForest(data=df, df$Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL)
summary(model_forest_class)

#check importance variables of classification forest

importance(model_forest_class)

# CPI and PPI appears to be the most importances variables

varImpPlot(model_forest_class)

plot(model_forest_class)

#Based on the plot, 350 trees seems to be achieve a stable performance. Therefore we can set ntree=350

set.seed(12)
oob.err=double()
for (i in 1:4){
  individual_rf=randomForest(data=df,df$Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL,ntree=350, mtry=1+i)
  #out of bag error (based on the embedded rf output)
  oob.err[i]=individual_rf$err.rate[350,1]
}
oob.err

# Given this seed, 4 variables give the lowest oob.err

model_forest_class=randomForest(data=df, df$Fed.Rate.Hike~CPI+PPI+PCE+INDPRO+PAYROLL, ntree=350, mtry=4)

predict_class_forest=predict(data=df$Fed.Rate.Hike, model_forest_class)

#Performance evaluation 
matrix4=table(df$Fed.Rate.Hike,predict_class_forest)

accuracy_class_forest=sum(diag(matrix4))/sum(matrix4)

sensitivity_class_forest=matrix4[2,2]/sum(matrix4[2,])

specificity_class_forest=matrix4[1,1]/sum(matrix4[1,])

roc_class_forest=roc.curve(df$Fed.Rate.Hike,predict_class_forest)
roc_class_forest

#Build ression Forest to predict the Fed interest rate

model__forest=randomForest(data=df, value~CPI+PPI+PCE+INDPRO+PAYROLL)
plot(model_reg_forest)

# According to the chart, ntree=500 appearts to generate the lowest error, mtry ranging form 2 to 5 (i ranges from 1 to 4)

set.seed(1)
oob.err=double()
for (i in 1:4){
  individual_rf2=randomForest(data=df,value~CPI+PPI+PCE+INDPRO+PAYROLL,ntree=500, mtry=1+i)
  #out of bag error (based on the embedded rf output)
  oob.err[i]=individual_rf2$mse[500]
} 

# mtry=2 yeilds the lowest oob error
# 
# Now, rebuild the regrssion forest using ntree=500 and mtry=2

model_reg_forest=randomForest(data=df, value~CPI+PPI+PCE+INDPRO+PAYROLL, mtry=2,ntree=500)
predict_reg_forest=predict(data=df$value, model_reg_forest)

# Regression forest performance evaluation

rmse_reg_forest=sqrt(mean((df$value-predict_reg_forest)^2))

rsq_reg_forest= 1-(sum((df$value-predict_reg_forest)^2)/sum((df$value-mean(df$value))^2))                   


#Overall Performance

# Classfication models 

AUC
roc_logit1 # Main variables only
0.609

roc_logit2 # Include interaction terms
0.681

roc__class_tree # Classification tree 
0.801

roc_class_forest #Classfication Forest
0.676

# Classification tree has the highest AUC

# Accuracy (use cutoff point=0.5)

accuracy_1 
0.573

accuracy_logit2
0.629

accuracy_class_tree
0.746

accuracy_class_forest
0.676

# Classification tree has the highest accuracy


Specificity
specificity_1
0.589

specificity_logit2
0.699

specificity_class_tree
0.813

specificity_class_forest
0.709

# Classification tree has the highest specificty 


Sensitivity 
sensitivty_1
0.556

sensitivity_logit2
0.559

sensitivity_class_tree
0.677

sensitivity_class_forest
0.642

# Classification tree has the highest sensitivity


# For Logit1(main variables only) and logit2 (including interaction terms) we also compared Mc Fadden rsq

fadden_logit1
0.028

fadden_step_logit
0.070

# though both have very low Mc fadden rsq, model 2 still shows that it has a better performance 

# Among all classification model, classification tree demonstrated the best performance in all criteria. 



# Regression model (measure the interest rate change)
# 
# R square (meaningful but not really appropriate)

summary(model_linear)$r.squared
0.731

rsq_reg_tree
0.973

rsq_reg_forest
0.985

# Regression Forest results the highest R squared

RMSE

rmse_linear
1.923

rmse_reg_tree
0.608

rmse_reg_forest
0.449
# 
# Regression forest generates the lowest RMSE, which demonstrates its ability to minimize predictive errors
# 
# Regression forest appears to be the best model in predicting the how much the change in the Fed interest rate is likely to be