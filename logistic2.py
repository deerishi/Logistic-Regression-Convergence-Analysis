import numpy as np
from collections import defaultdict 
import csv
import matplotlib.pylab as plt
from copy import copy


np.set_printoptions(threshold='nan')

class Logistic:
    
    def __init__(self):
        pass
    
    def logit(self,X,w):
        pie=np.zeros((X.shape[0],1))
        for i in range(0,X.shape[0]):
            x=X[i]
            t1=np.exp(-1*np.dot(x,w))
            t1=t1[0]
            pie[i]=1/(1+t1)
        
        return pie
        
    def logit2(self,X,w):
        t1=np.exp(-1*np.dot(X,w))
        
        t1=t1[0]
        #print 'np.dot(X,w) is ',np.dot(X,w)
        
        t1=1/(1+t1)
        #print 't1 is ',t1
        return t1
        
    def contructHessian(self,X,w):
        
        pie=self.logit(X,w)
        d=pie*(1-pie)
        d=d.reshape(pie.shape[0],)
        R=np.diag(d)
        hessian=np.dot(np.dot(X.T,R),X)
        self.hessian=hessian    
    
    def LR_CalcObj(self,Xtrain,yTrain,what):

        w=what
        obj=0
        for i in range(0,Xtrain.shape[0]):
            cterm=np.dot(Xtrain[i],w)
            obj+=yTrain[i]*cterm - np.log(1+np.exp(cterm))
            
        return obj
    
            
    def train(self,X,y,tolerance,max_iter,plotobj):
        
        self.X_train=np.hstack((np.ones((X.shape[0],1)),X))
        self.y_train=y.reshape(y.shape[0],1)
        
        weights=np.zeros((self.X_train.shape[1],1))
        obj=[]
        
        wnorm=[]
        oldObj=self.LR_CalcObj(self.X_train,self.y_train,weights)
        #obj.append(oldObj)
        counter=0
        while counter < max_iter :
            
            pie=self.logit(self.X_train,weights)
            diff=pie-self.y_train
            gradL=np.dot(self.X_train.T,diff)
            self.contructHessian(self.X_train,weights)
            new_weights=weights - np.dot(np.linalg.inv(self.hessian),gradL)
            wnorm.append(np.linalg.norm(new_weights))
            newObj=self.LR_CalcObj(self.X_train,self.y_train,new_weights)
            obj.append(newObj)
            #if abs(newObj-oldObj) <= tolerance :
                #self.weights=new_weights
                #return
            
            weights=new_weights
            counter+=1
            
        self.weights=weights 
        if plotobj==True:
            plt.plot(range(0,max_iter),obj)
            #plt.plot(range(0,max_iter),wnorm)
            plt.title('Objective function of logistic regession')
            plt.xlabel('Iterations')
            plt.ylabel('Objective Function')
            plt.show()
        return obj
            
    def test(self,X):
        labels=[]
        self.X_test=np.hstack((np.ones((X.shape[0],1)),X))
        for i in range(0,self.X_test.shape[0]):
            y1=self.logit2(self.X_test[i],self.weights)
            if y1 >=  0.5:
                labels.append(1)
            else:
                labels.append(0)
        
        labels=np.asarray(labels)
        return labels
     
    #def Test(self,c0,c1,u0,u1) 
       
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
    
    def retrain(self):
        datasetsX=[]
        labelsy=[]
        for i in range(1,11):
            Xname="data"+str(i)+".csv"
            fx=open(Xname)
            xReader=csv.reader(fx)
            X=[]
            y=[]
            for row in xReader:
                row=[float(x) for x in row]
                X.append(row)
            X=np.asarray(X)
            datasetsX.append(X)

            yname="labels"+str(i)+'.csv'
            fy=open(yname)
            yReader=csv.reader(fy)
            for row in yReader:
                row=[int(x) for x in row]
                if row[0]==5 :
                    y.append(0)
                elif row[0]==6:
                    y.append(1)
                
            y=np.asarray(y)
            labelsy.append(y)
       
        trainX=datasetsX[0]
        trainy=labelsy[0]
        
        for i in range(1,10):
            trainX=np.vstack((trainX,datasetsX[i]))
            trainy=np.hstack((trainy,labelsy[i]))
            
        print 'size of training is ',trainX.shape,trainy.shape
        

        self.train(trainX,trainy,0.001,20,True)
            #print 'Now testing------>'
        predicted=self.test(trainX)
        accuracy=self.checkAccuracy(predicted,trainy)  

        print 'accuracy after retraining all data is ',accuracy
        print 'the weights after training is ',self.weights.T
        
    def crossValidate(self,K=10):
        datasetsX=[]
        labelsy=[]
        for i in range(1,11):
            Xname="data"+str(i)+".csv"
            fx=open(Xname)
            xReader=csv.reader(fx)
            X=[]
            y=[]
            for row in xReader:
                row=[float(x) for x in row]
                X.append(row)
            X=np.asarray(X)
            datasetsX.append(X)

            yname="labels"+str(i)+'.csv'
            fy=open(yname)
            yReader=csv.reader(fy)
            for row in yReader:
                row=[int(x) for x in row]
                if row[0]==5 :
                    y.append(0)
                elif row[0]==6:
                    y.append(1)
                
            y=np.asarray(y)
            labelsy.append(y)
            

        #print 'labels are ',labelsy[0].shape,datasetsX[0]
        #now we make cross validation datasets
        kcross=[]
        for i in range(0,K):
            kcross.append(i)
        for i in range(0,K-1):
            kcross.append(i)
        accuracies=[]
        for i in range(0,K):
            testX=datasetsX[kcross[i]]
            testY=labelsy[kcross[i]]
            
            trainX=datasetsX[kcross[i+1]]
            trainy=labelsy[kcross[i+1]]

            for j in range(i+2,i+K):
                trainX=np.vstack((trainX,datasetsX[kcross[j]]))
                trainy=np.hstack((trainy,labelsy[kcross[j]]))

             
            self.train(trainX,trainy,0.001,20,False)
            #print 'Now testing------>'
            predicted=self.test(testX)
            accuracy=self.checkAccuracy(predicted,testY)  
            accuracies.append(accuracy)
            #print 'accuracy is ',accuracy
       
        return  np.mean(accuracies)  

logistic=Logistic()


accuracy=logistic.crossValidate(K=10)
print 'accuracy is ',accuracy,'%'
logistic.retrain()
