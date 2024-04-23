library(caret)
library(glmnet)
library(kernlab)
library(plotly)
library(dplyr)
library(pROC)

dbAlz <- read.csv("~/POSDOCTORADO/Python Seleccion Caracteristicas/Datasets_Con_OSampler/adnimerge-EObsNA_2_SanovsEnfermo_OSampler.csv")
#dbAlz <- read.csv("~/POSDOCTORADO/Python Seleccion Caracteristicas/adnimerge-EObsNA_2_SanovsEnfermo.csv")

##Vamos a separar los datos en 70 /30 para train y test
dT <- dbAlz
set.seed(5)
setenta<-createDataPartition(dT$DX.bl,
                             times = 1,
                             p=0.70,
                             list=TRUE)
indices<-setenta$Resample1

#con los indices creamos las particiones de 70/30
train<-dT[indices,]
test<-dT[-indices,]

bd <- dbAlz[sample(nrow(dbAlz)),]
bd$DX.bl<-ifelse(bd$DX.bl==1,'Y','N')

indices <-createDataPartition(bd$DX.bl, p=0.70, list=FALSE)
dTr <- bd[ indices,]
dTe <- bd[-indices,]
parametros <- trainControl(
  method="cv",
  number=10,
  savePredictions = 'final',
  classProbs = TRUE
)

#RFE - LR 
#predictores <-c('SEXBIN', 'WHR', 'CHOL..mg.dl.', 'TCHOLU', 'HA.TX', 'SBP', 'DBP', 'SBPU', 'DBPU')

#RFE - SVM
#predictores <-c('SEXBIN', 'WHR', 'Crea..mg.dl.', 'CHOL..mg.dl.', 'TCHOLU', 'SBP', 'DBP', 'SBPU', 'DBPU')

#RFE - RF
#predictores <-c('Age', 'LIPIDS.TX', 'TG.B..mg.dl.', 'DBP', 'DBPU')

#SIglo XXI KNN
#predictores <-c("Crea..mg.dl.", "TGU", "TCHOLU", "SEXBIN", "WHR", "SBP", "SBPU", "CHOL..mg.dl.", "UREA..mg.dl.")

#Siglo XXI Nearcent
#predictores <-c("Crea..mg.dl.","TGU","SEXBIN","SBPU","LDLc..mg.dl.","SBP","BMI","WHR","Age","TG.B..mg.dl.","CHOL..mg.dl.","LDLU","UREA..mg.dl.","TCHOLU","HDLU","LIPIDS.TX")   

#Siglo XXI LogReg
#predictores <-c("SEXBIN","Crea..mg.dl.","SBP","TGU","TCHOLU","CHOL..mg.dl.","SBPU","LDLc..mg.dl.","LDLU","HA.TX","DBP","WHR","LIPIDS.TX","HDLc..mg.dl.","TG.B..mg.dl.","HDLU","Age","BMI")         

#Siglo XXI SVM
#predictores <-c("SEXBIN","Crea..mg.dl.","TGU","SBP","LDLU","SBPU","LDLc..mg.dl.","TCHOLU","CHOL..mg.dl.","BMI","Age","WHR","UREA..mg.dl.","LIPIDS.TX","HDLc..mg.dl.","TG.B..mg.dl.","HDLU","HA.TX") 

#Siglo XXI NNET
#predictores <-c("Crea..mg.dl.","TGU")    

#LASSO Siglo XXI
#predictores <-c("SEXBIN","Age","WHR","BMI","UREA..mg.dl.","LIPIDS.TX","HDLc..mg.dl.","TG.B..mg.dl.","HA.TX","DBP","SBPU") 

#Repeated in the best models
predictores <-c("CDRSB") 

salida <- "DX.bl"
#Entrenamiento del modelo SVM
set.seed(1)
modelo_SVM<-train(dTr[predictores],
                  dTr[,salida],
                  method='svmRadial',
                  trControl=parametros,
                  tuneLength=3)
#Obtenemos las predicciones de SVM en el conjunto 
#de prueba y las ponemos en una nueva columna que 
#se llama prediccionesSVM
dTe$prediccionesSVM<-predict(object = modelo_SVM, dTe[,predictores])
cm <- confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$prediccionesSVM),positive = 'Y')
cm

dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$prediccionesSVM <- ifelse(dTe$prediccionesSVM=='Y',1,0)
ROC_SVM<-roc(dTe$DX.bl,dTe$prediccionesSVM,
             levels=c(0,1),plot=FALSE,
             ci=TRUE, smooth=FALSE,
             direction='auto',col="red",percent = TRUE, print.auc = TRUE)
ROC_SVM$auc

ROC<-roc(dTe$DX.bl,dTe$prediccionesSVM,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="red",percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 8, main="SVM")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')
dTe$prediccionesSVM <- ifelse(dTe$prediccionesSVM==1,'Y','N')
#Entrenamiento del modelo NN
set.seed(1)
modelo_NN<-train(dTr[,predictores],
                 dTr[,salida],
                 method='nnet',
                 trControl=parametros,
                 tuneLength=3)
#Obtenemos las predicciones de NN en el conjunto 
#de prueba y las ponemos en una nueva columna que 
#se llama prediccionesNN
dTe$prediccionesNN<-predict(object = modelo_NN, dTe[,predictores])
cm <- confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$prediccionesNN),positive = 'Y')
cm

dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$prediccionesNN <- ifelse(dTe$prediccionesNN=='Y',1,0)

ROC_NN<-roc(dTe$DX.bl,dTe$prediccionesNN,
            levels=c(0,1),plot=FALSE,
            ci=TRUE, smooth=FALSE,
            direction='auto',col="blue",percent = TRUE, print.auc = TRUE)
ROC_NN$auc
ROC<-roc(dTe$DX.bl,dTe$prediccionesNN,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="blue",percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 10, main="NNET")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')
dTe$prediccionesNN <- ifelse(dTe$prediccionesNN==1,'Y','N')

#Entrenamiento del modelo RL
set.seed(1)
modelo_GLM<-train(dTr[,predictores],
                  dTr[,salida],
                  method='glm',
                  trControl=parametros,
                  tuneLength=3)
#Obtenemos las predicciones de RL en el conjunto 
#de prueba y las ponemos en una nueva columna que 
#se llama prediccionesRL

dTe$prediccionesGLM<-predict(object = modelo_GLM, dTe[,predictores])
cm <- confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$prediccionesGLM),positive = 'Y')
cm

dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$prediccionesGLM <- ifelse(dTe$prediccionesGLM=='Y',1,0)
ROC_GLM<-roc(dTe$DX.bl,dTe$prediccionesGLM,
             levels=c(0,1),plot=FALSE,
             ci=TRUE, smooth=FALSE,
             direction='auto',col="darkgreen",percent = TRUE, print.auc = TRUE)

ROC_GLM$auc
ROC<-roc(dTe$DX.bl,dTe$prediccionesGLM,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="darkgreen",percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 10, main="LOGREG")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')

#Entrenamiento del modelo KNN
set.seed(1)
modelo_KNN<-train(dTr[,predictores],
                  dTr[,salida],
                  method='knn',
                  trControl=parametros,
                  tuneLength=3)
#Obtenemos las predicciones de KNN en el conjunto 
#de prueba y las ponemos en una nueva columna que 
#se llama prediccionesKNN

dTe$prediccionesKNN<-predict(object = modelo_KNN, dTe[,predictores])
cm <- confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$prediccionesKNN),positive = 'Y')
cm

dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$prediccionesKNN <- ifelse(dTe$prediccionesKNN=='Y',1,0)
ROC_KNN<-roc(dTe$DX.bl,dTe$prediccionesKNN,
             levels=c(0,1),plot=FALSE,
             ci=TRUE, smooth=FALSE,
             direction='auto',col="purple",percent = TRUE, print.auc = TRUE)

ROC_KNN$auc
ROC<-roc(dTe$DX.bl,dTe$prediccionesKNN,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="purple",percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 10, main="KNN")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')

#Entrenamiento del modelo NEARCENT
set.seed(1)
modelo_NEARCENT<-train(dTr[,predictores],
                  dTr[,salida],
                  method='pam',
                  trControl=parametros,
                  tuneLength=3)
#Obtenemos las predicciones de NEARCENT en el conjunto 
#de prueba y las ponemos en una nueva columna que 
#se llama prediccionesNEARCENT

dTe$prediccionesNEARCENT<-predict(object = modelo_NEARCENT, dTe[,predictores])
cm <- confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$prediccionesNEARCENT),positive = 'Y')
cm

dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$prediccionesNEARCENT <- ifelse(dTe$prediccionesNEARCENT=='Y',1,0)
ROC_NEARCENT<-roc(dTe$DX.bl,dTe$prediccionesNEARCENT,
             levels=c(0,1),plot=FALSE,
             ci=TRUE, smooth=FALSE,
             direction='auto',col="brown",percent = TRUE, print.auc = TRUE)

ROC_NEARCENT$auc
ROC<-roc(dTe$DX.bl,dTe$prediccionesNEARCENT,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="brown",percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 10, main="Nearcent")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')

dTe$prediccionesGLM <- ifelse(dTe$prediccionesGLM==1,'Y','N')
dTe$prediccionesNN
dTe$prediccionesSVM
dTe$prediccionesKNN
dTe$prediccionesNEARCENT
###Ensamble por votación
dTe$pMV<-as.factor(ifelse(dTe$prediccionesSVM=='Y'&dTe$prediccionesNN=='Y','Y',
                          ifelse(dTe$prediccionesSVM=='Y'&dTe$prediccionesGLM=='Y','Y',
                                 ifelse(dTe$prediccionesSVM=='Y'&dTe$prediccionesKNN=='Y','Y',
                                        ifelse(dTe$prediccionesSVM=='Y'&dTe$prediccionesNEARCENT=='Y','Y',
                                               ifelse(dTe$prediccionesGLM=='Y'&dTe$prediccionesKNN=='Y','Y',
                                                      ifelse(dTe$prediccionesGLM=='Y'&dTe$prediccionesNEARCENT=='Y','Y',
                                                             ifelse(dTe$prediccionesNN=='Y'&dTe$prediccionesKNN=='Y','Y',
                                                                    ifelse(dTe$prediccionesNN=='Y'&dTe$prediccionesNEARCENT=='Y','Y',
                                 ifelse(dTe$prediccionesNN=='Y'&dTe$prediccionesGLM=='Y','Y','N'))))))))))
TrPositive <- ifelse(as.factor(dTe$DX.bl)==as.factor(dTe$pMV) & as.factor(dTe$DX.bl) == 'Y',1, 0)
print(TrPositive)
cm<-confusionMatrix(as.factor(dTe$DX.bl),as.factor(dTe$pMV))
cm
dTe$DX.bl <- ifelse(dTe$DX.bl=='Y',1,0)
dTe$pMV <- ifelse(dTe$pMV=='Y',1,0)

#print(dTe$DX.bl)
#print(dTe$pMV)

ROC_E<-roc(dTe$DX.bl,dTe$pMV,
           levels=c(0,1),plot=FALSE,
           ci=TRUE, smooth=FALSE,
           direction='auto',col="black", percent = TRUE, print.auc = TRUE)

ROC_E$auc
ROC<-roc(dTe$DX.bl,dTe$pMV,
         levels=c(0,1),plot=TRUE,
         ci=TRUE, smooth=FALSE,
         direction='auto',col="black", percent = TRUE, print.auc = TRUE, print.auc.x = 85,
         print.auc.y = 10,main="Ensemble model")

dTe$DX.bl <- ifelse(dTe$DX.bl==1,'Y','N')
dTe$prediccionesSVM <- ifelse(dTe$prediccionesSVM==1,'Y','N')


