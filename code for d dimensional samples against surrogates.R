rm(list=ls())
gc()
#library(fOptions)
install.packages("lhs")
library(lhs)
install.packages("DiceKriging")
library(DiceKriging)
install.packages("neuralnet")
library(neuralnet)
install.packages("fOptions")
library(fOptions)

zhou98 <- function(xx)
{
  d <- length(xx)
  
  xxa <- 10 * (xx-1/3)
  xxb <- 10 * (xx-2/3)
  
  norma <- sqrt(sum(xxa^2))
  normb <- sqrt(sum(xxb^2))
  
  phi1 <- (2*pi)^(-d/2) * exp(-0.5*(norma^2))
  phi2 <- (2*pi)^(-d/2) * exp(-0.5*(normb^2))
  
  y <- (10^d)/2 * (phi1 + phi2)
  return(y)
}
###defining the variables and matrices to store the result###
n=50
dimen<-c(2,3,5)
msUNREG<-msPQREG<-msQSREG<-msLHREG<-msHSREG<-matrix(0,nrow=50,ncol=3)
mseUNDK<-msePQD<-mseQS<-mseLH<-mseHS<-matrix(0,nrow=50, ncol=3)
msnnUN<-msnnUNT<-msnnPQ<-msnnPQT<-msnnLH<-msnnLHT<-msnnHS<-msnnHST<-matrix(0,nrow=50, ncol=3)
for(k in 1:length(dimen)){
  d2<-dimen[k]
  for(i in 1:50){
    set.seed(i)
    ##########uniform data#########
    
    UN<-matrix(c(runif(n*d2)), nrow=n,ncol=d2, byrow=TRUE)
    UN<-(UN-min(UN))/(max(UN)-min(UN))
    UNtrain<-UN[1:(n/2),]
    UNtest<-UN[((n/2)+1):n,]
    UNytrain<-apply(UNtrain,1,zhou98)
    dim(UNytrain)<-c(n/2,1)
    UNytest<-apply(UNtest,1,zhou98)
    dim(UNytrain)<-c(n/2,1)
    colnames(UNtrain) <- paste("x", 1:d2, sep = "")
    colnames(UNtest) <- paste("x", 1:d2, sep = "")
    ########DATA FOR NN######
    UNytrain2<-(UNytrain-min(UNytrain))/(max(UNytrain)-min(UNytrain))
    UNytest2<-(UNytest-min(UNytest))/(max(UNytest)-min(UNytest))
    UNytest2<-as.matrix(UNytest2)
    colnames(UNytrain2)<-"yUN"
    colnames(UNytest2)<-"yUN"
    UNtraindat<-cbind(UNtrain,UNytrain2)
    UNtestdat<-cbind(UNtest,UNytest2)
    ########UNIFORM regression data#####
    UNtestdat2<-data.frame(UNtest)
    UNtraindat2<-cbind(UNtrain,UNytrain)
    UNtraindat2<-data.frame(UNtraindat2)
    
    ######## 1. UNIF REGRESSION######
    PredictorVariables <- paste("x", 1:d2, sep="")
    Formula <- formula(paste("UNytrain ~ ", 
                             paste(PredictorVariables, collapse=" + ")))
    UNreg1<-lm(Formula, UNtraindat2)
    UNpred<-predict(UNreg1,newdata=UNtestdat2)
    UNpred<-as.matrix(UNpred)
    msUNREG[i,k]<-mean((UNytest-UNpred)^2)
    
    ####### 1. Dicekriging uniform ######
    mUN<-km(design= UNtrain, response=UNytrain,covtype="matern5_2", nugget=0.001)
    mUN
    predictUN <- predict(mUN, UNtest, "UK")
    yhatUN<-predictUN$mean
    #emperror1<-cbind(UNytest,yhatUN)
    mseUNDK[i,k]<-mean((UNytest-yhatUN)^2)
    
    #### 1. UNIFORM NN with activ func########
    PredictorVariables <- paste("x", 1:d2, sep="")
    Formula <- formula(paste("yUN ~ ", 
                             paste(PredictorVariables, collapse=" + ")))
    nnUN = neuralnet(Formula, data=UNtraindat, hidden=3,
                     act.fct= "tanh" ,linear.output = FALSE) #threshold=0.01)
    
    predUN<-predict(nnUN, UNtestdat[,1:d2])
    msnnUN[i,k]<-mean(( UNytest2-predUN)^2)
    
    #############NN WO AF######UNytrain2
    PVUN <- paste("x", 1:d2, sep="")
    FormulaT <- formula(paste("yUN ~ ", 
                              paste(PVUN, collapse=" + ")))
    nnUNT = neuralnet(FormulaT, data=UNtraindat, hidden=3,
                      linear.output = TRUE)
    
    predUNT<-predict(nnUNT, UNtestdat[,1:d2])
    msnnUNT[i,k]<-mean(( UNytest2-predUNT)^2)
    
    
    ################SOBOL SAMPLING########################
    #PQ<-runif.sobol(n=n,dimension=3,init =TRUE)
    PQ<-runif.sobol(n=n, dimension=d2,init=TRUE)
    PQtrain<-PQ[1:(n/2),]
    PQtest<-PQ[((n/2)+1):n,]
    PQytrain<-apply(PQtrain,1,zhou98)
    dim(PQytrain)<-c(n/2,1)
    PQytest<-apply(PQtest,1,zhou98)
    dim(PQytest)<-c(n/2,1)
    colnames(PQtrain) <- paste("p", 1:d2, sep = "")
    colnames(PQtest) <- paste("p", 1:d2, sep = "")
    ######Neural netwok DATA SOBOL######
    PQytrain2<-(PQytrain-min(PQytrain))/(max(PQytrain)-min(PQytrain))
    PQytest2<-(PQytest-min(PQytest))/(max(PQytest)-min(PQytest))
    colnames(PQytrain2)<-"yPQ"
    colnames(PQytest2)<-"yPQ"
    PQtraindat<-cbind(PQtrain,PQytrain2)
    PQtestdat<-cbind(PQtest,PQytest2)
    #####SOIBOL data for regression####
    PQtraindat2<-cbind(PQtrain,PQytrain)
    PQtraindat2<-data.frame(PQtraindat2)
    PQtestdat2<-data.frame(PQtest)
    
    
    
    ################## REGRESSION SOBOL#######
    PVPQ <- paste("p", 1:d2, sep="")
    Formula2 <- formula(paste("PQytrain ~ ", 
                              paste(PVPQ, collapse=" + ")))
    PQreg1<-lm(Formula2, PQtraindat2)
    PQpred<-predict(PQreg1,newdata=PQtestdat2)
    PQpred<-as.matrix(PQpred)
    msPQREG[i,k]<-mean((PQytest-PQpred)^2)
    
    ############## DICEKRIGING SOBOL#########
    mPQD<-km(design=PQtrain, response=PQytrain,covtype="matern5_2", nugget=0.001)
    mPQD
    predictPQD <- predict(mPQD, PQtest, "UK")
    yhatPQD<-predictPQD$mean
    #emperror1<-cbind(ytest,yhat)
    msePQD[i,k]<-mean((PQytest-yhatPQD)^2)
    
    ############## Neural Network with AF#############
    PPQ <- paste("p", 1:d2, sep="")
    Formula1 <- formula(paste("yPQ ~ ", 
                              paste(PPQ, collapse=" + ")))
    nnPQ = neuralnet(Formula1, data=PQtraindat, hidden=3,
                     act.fct= "tanh" ,linear.output = FALSE) #threshold=0.01)
    
    predPQ<-predict(nnPQ, PQtestdat[,1:d2])
    msnnPQ[i,k]<-mean(( PQytest2-predPQ)^2)
    
    ######## NN WO AF (Regression)#######
    PPQT <- paste("p", 1:d2, sep="")
    FormulaT <- formula(paste("yPQ ~ ", 
                              paste(PPQT, collapse=" + ")))
    nnPQT = neuralnet(FormulaT, data=PQtraindat, hidden=3,
                      linear.output = TRUE) #threshold=0.01)
    
    predPQT<-predict(nnPQT, PQtestdat[,1:d2])
    msnnPQT[i,k]<-mean((PQytest2-predPQT)^2)
    
    
    
    #########################LATIN HYPERCUBE###################
    ################ (LATIN HYPERCUBE SAMPLING)#################
    LH<-randomLHS(n,d2)
    LH<-(LH-min(LH))/(max(LH)-min(LH))
    LHtrain<-LH[1:(n/2),]
    
    LHtest<-LH[((n/2)+1):n,]
    LHytrain<-apply(LHtrain,1,zhou98)
    dim(LHytrain)<-c(n/2,1)
    
    LHytest<-apply(LHtest,1,zhou98)
    dim(LHytest)<-c(n/2,1)
    colnames(LHtrain) <- paste("l", 1:d2, sep = "")
    colnames(LHtest) <- paste("l", 1:d2, sep = "")
    ###### DATA FOR NN#########
    LHytrain2<-(LHytrain-min(LHytrain))/(max(LHytrain)-min(LHytrain))
    LHytest2<-(LHytest-min(LHytest))/(max(LHytest)-min(LHytest))
    colnames(LHytrain2)<-"yLH"
    colnames(LHytest2)<-"yLH"
    LHtraindat<-cbind(LHtrain,LHytrain2)
    LHtestdat<-cbind(LHtest,LHytest2)
    
    ############ DATA FOR REGRESSION###
    ######DATA FOR REGRESSION####
    LHtraindat2<-cbind(LHtrain,LHytrain)
    LHtraindat2<-data.frame(LHtraindat2)
    LHtestdat2<-data.frame(LHtest)
    #########   End LATIN HYPERCUBE#######
    
    ###### 4. LHS Regression#####
    
    PVLH <- paste("l", 1:d2, sep="")
    Formula4 <- formula(paste("LHytrain ~ ", 
                              paste(PVLH, collapse=" + ")))
    LHreg<-lm(Formula4, LHtraindat2)
    LHpred<-predict(LHreg,newdata=LHtestdat2)
    LHpred<-as.matrix(LHpred)
    msLHREG[i,k]<-mean((LHytest-LHpred)^2)
    
    ############LH DICEKRIGING####
    mLH<-km(design= LHtrain, response=LHytrain,covtype="matern5_2", nugget=0.001)
    mLH
    predictLH <- predict(mLH, LHtest, "UK")
    yhatLH<-predictLH$mean
    #emperror1<-cbind(ytest,yhat)
    mseLH[i,k]<-mean((LHytest-yhatLH)^2)
    
    ############## NN AF#############
    PLH <- paste("l", 1:d2, sep="")
    Formula3 <- formula(paste("yLH ~ ", 
                              paste(PLH, collapse=" + ")))
    nnLH = neuralnet(Formula3, data=LHtraindat, hidden=3,
                     act.fct= "tanh" ,linear.output = FALSE) #threshold=0.01)
    
    predLH<-predict(nnLH, LHtestdat[,1:d2])
    msnnLH[i,k]<-mean(( LHytest2-predLH)^2)
    
    ######## NN WO AF#######
    PLH <- paste("l", 1:d2, sep="")
    Formula3 <- formula(paste("yLH ~ ", 
                              paste(PLH, collapse=" + ")))
    nnLHT = neuralnet(Formula3, data=LHtraindat, hidden=3,
                      linear.output = TRUE) #threshold=0.01)
    
    predLHT<-predict(nnLHT, LHtestdat[,1:d2])
    msnnLHT[i,k]<-mean(( LHytest2-predLHT)^2)
    
    #}
    #}
    
    ############################################
    ########## (HALTON SAMPLING)#########
    HS<-runif.halton(n, d2, init=TRUE)
    HS[,1]<-(HS[,1]-min(HS[,1]))/(max(HS[,1])-min(HS[,1]))
    HStrain<-HS[1:(n/2),]
    HStest<-HS[((n/2)+1):n,]
    Hstest<-HS[((n/2)+1):n,]
    HSytrain<-apply(HStrain,1,zhou98)
    dim(HSytrain)<-c(n/2,1)
    HSytest<-apply(HStest,1,zhou98)
    dim(HSytest)<-c(n/2,1)
    colnames(HStrain) <- paste("h", 1:d2, sep = "")
    colnames(HStest) <- paste("h", 1:d2, sep = "")
    #####  DATA FOR NN######
    HSytrain2<-(HSytrain-min(HSytrain))/(max(HSytrain)-min(HSytrain))
    HSytest2<-(HSytest-min(HSytest))/(max(HSytest)-min(HSytest))
    colnames(HSytrain2)<-"yHS"
    colnames(HSytest2)<-"yHS"
    HStraindat<-cbind(HStrain,HSytrain2)
    HStestdat<-cbind(HStest,HSytest2)
    ######DATA FOR REGRESSION####
    HStraindat2<-cbind(HStrain,HSytrain)
    HStraindat2<-data.frame(HStraindat2)
    HStestdat2<-data.frame(HStest)
    
    ###### HALTON REGRESIION#####
    PVHS <- paste("h", 1:d2, sep="")
    Formula5 <- formula(paste("HSytrain ~ ", 
                              paste(PVHS, collapse=" + ")))
    HSreg<-lm(Formula5, HStraindat2)
    HSpred<-predict(HSreg,newdata=HStestdat2)
    HSpred<-as.matrix(HSpred)
    msHSREG[i,k]<-mean((HSytest-HSpred)^2)
    
    ######DICEKRIGING HALTON#####
    mHS<-km(design= HStrain, response=HSytrain,covtype="matern5_2", nugget=0.001)
    mHS
    predictHS <- predict(mHS, HStest, "UK")
    yhatHS<-predictHS$mean
    #emperror1<-cbind(ytest,yhat)
    mseHS[i,k]<-mean((HSytest-yhatHS)^2)
    
    ###########################
    ####### HALTON NNAF##############
    PHS <- paste("h", 1:d2, sep="")
    Formula4 <- formula(paste("yHS ~ ", 
                              paste(PHS, collapse=" + ")))
    nnHS = neuralnet(Formula3, data=LHtraindat, hidden=3,
                     act.fct= "tanh" ,linear.output = FALSE) #threshold=0.01)
    predHS<-predict(nnHS, HStestdat[,1:d2])
    msnnHS[i,k]<-mean(( HSytest2-predHS)^2)
    ####### NNT HALTON########
    nnHST = neuralnet(Formula4 ,data=HStraindat, hidden=3,
                      linear.output = TRUE) #threshold=0.01)
    
    predHST<-predict(nnLHT, HStestdat[,1:d2])
    msnnHST[i,k]<-mean(( HSytest2-predHST)^2)
  }
}


########################################
################### BOXPLOTS############
############### Plots for mean square errors for Surrogate modelsagainst each sampling scheme####
MSEUNif<-cbind(msUNREG[,1],mseUNDK[,1],msnnUN[,1],msnnUNT[,1])
MSEUNif<-log(MSEUNif)
colnames(MSEUNif)<-c("regression","Dicekriging","NN(tanh)", "NN(reg)")
boxplot(MSEUNif,main="Boxplots of Mean Square Error using Uniform samples 3-d(all models)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("linear regression","Dicekriging", "NN(AF)", "NN(reg)"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

MSESOBOL<-cbind(msPQREG[,1],msePQD[,1],msnnPQ[,1],msnnPQT[,1]) #each mean square matrix contaimns each column for one dimension so [,1] means combining d=3 for all models using this design
MSESOB<-log(MSESOBOL)
#MSESOB<-paste(MSESOB)
colnames(MSESOB)<-c("regression","Dicekriging","NN(tanh)", "NN(reg)")
boxplot(MSESOB,main="Boxplots of Mean Square Error using Sobol samples 3-d(all models)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("linear regression","Dicekriging", "NN(AF)", "NN(reg)"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

MSELHS<-cbind(msLHREG[,1],mseLH[,1],msnnLH[,1],msnnLHT[,1])
MSELH3D<-log(MSELHS)
colnames(MSELHS)<-c("regression","Dicekriging","NN(tanh)", "NN(reg)")
boxplot(MSELHS,main="Boxplots of Mean Square Error using LHS samples 3-d(all models)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("linear regression","Dicekriging", "NN(AF)", "NN(reg)"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

MSEHS<-cbind(msHSREG[,1],mseHS[,1],msnnHS[,1],msnnHST[,1])
MSEHS3D<-log(MSEHS)
colnames(MSEHS3D)<-c("regression","Dicekriging","NN(tanh)", "NN(reg)")
boxplot(MSEHS3D,main="Boxplots of Mean Square Error using Halton samples 3-d(all models)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("linear regression","Dicekriging", "NN(AF)", "NN(reg)"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

#############################################
#################  ALL DESIGNS AGAINST EACH SURROGATE MODEL#####
#par(mfrow=c(2,1))
##1###
MSEDK<-cbind(mseUNDK[,1],mseLH[,1],msePQD[,1], mseHS[,1]) #mseQS[,1]
MSEDK<-log(MSEDK)
colnames(MSEDK)<-c("Uniform","SOBOL","LH","Halton")
boxplot(MSEDK,main="Boxplots of Mean Square Error for DK against all designs (3D)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("UNIFORM","LATIN HYPERCUBE", "SOBOL", "HALTON"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)
##2###
MSEREG<-cbind(msUNREG[,1],msPQREG[,1],msLHREG[,1], msHSREG[,1])
MSEREG<-log(MSEREG)
colnames(MSEREG)<-c("Uniform","SOBOL","LH","Halton")
boxplot(MSEREG,main="Boxplots of Mean Square Error for REG against all designs",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("UNIFORM","LATIN HYPERCUBE", "SOBOL", "HALTON"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

###3###
MSENN<-cbind(msnnUN[,1],msnnPQ[,1],msnnLH[,1], msnnHS[,1])
MSENN<-log(MSENN)
colnames(MSENN)<-c("Uniform","SOBOL","LH","Halton")
boxplot(MSENN,main="Boxplots of Mean Square Error for NN(AF) against all designs (3D)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("UNIFORM","LATIN HYPERCUBE", "SOBOL", "HALTON"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)

#####4#####
MSENNT<-cbind(msnnUNT[,1],msnnPQT[,1],msnnLHT[,1], msnnHST[,1])
MSENNT<-log(MSENNT)
colnames(MSENNT)<-c("Uniform","SOBOL","LH","Halton")
boxplot(MSENNT,main="Boxplots of Mean Square Error for NN(NO,AF) against all designs (3D)",ylab="log", cex.main=0.8, col=c("red","blue", "green","purple"))
legend("topright", legend=c("UNIFORM","LATIN HYPERCUBE", "SOBOL", "HALTON"), col=c("red","blue", "green","purple"),lty=1, cex=0.6)


