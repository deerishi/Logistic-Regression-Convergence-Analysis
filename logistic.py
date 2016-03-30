import numpy as np
from collections import defaultdict 
import csv
import matplotlib.pylab as plt
from copy import copy


np.set_printoptions(threshold='nan')


class GMM:
    def __init__(self):
        pass

    def logit(self,X,w):
        pie=np.zeros((X.shape[0],1))
        for i in range(0,X.shape[0]):
            x=X[i]
            t1=np.exp(-1*np.dot(x,w))
            t1=t1[0]
            pie[i]=1/(1+t1)
            
        t1=np.exp(-1*np.dot(X,w))
        #t1=t1[0]
        pie=1/(1+t1)
        return pie
    
    def logit2(self,X,w):
        t1=np.exp(-1*np.dot(X,w))
        
        t1=t1[0]
        #print 'np.dot(X,w) is ',np.dot(X,w)
        
        t1=1/(1+t1)
        #print 't1 is ',t1
        return t1
        
        
    def contructHessian(self,X,w): 
        R=np.zeros((X.shape[0],X.shape[0]))
        for i in range(0,X.shape[0]):
            prob=self.logit2(X[i],w)
            R[i,i]=prob*(1-prob)
        
        #print 'R is ',R
        pie=self.logit(X,w)
        d=pie*(1-pie)
        d=d.reshape(d.shape[0],)
        R=np.diag(d)
        
        hessian=np.dot(X.T,np.dot(R,X))/self.X_train.shape[0]
        return hessian
 
    def train(self, X, y,tolerance,max_iter):
       
        self.X_train=np.hstack((np.ones((X.shape[0],1)),X))
        self.y_train=y.reshape(y.shape[0],1)
        #print 'X_train is ',self.X_train
        old_weights=np.zeros((self.X_train.shape[1],1))
        #old_weights=np.random.rand(self.X_train.shape[1],1)
        finished=False
        counter=0
        while finished==False and counter<max_iter:
            
            pie=self.logit(self.X_train,old_weights)
            #print 'pie is ',pie.shape
            hessian=self.contructHessian(self.X_train,old_weights)
            #print 'hessian shape is ',hessian.shape
            #print 'old_weights are ', old_weights
            #print 'hessian is ',np.linalg.det(hessian)
            gradL=np.dot(self.X_train.T,pie)/self.X_train.shape[0]
            #print 'gradL is ', gradL.shape
            new_weights=old_weights - np.dot(np.linalg.inv(hessian),gradL)
            
            diff=new_weights-old_weights
            #print 'np.linalg.norm(diff) is ',np.linalg.norm(diff)
            finished = ( np.linalg.norm(diff) <= tolerance)
            old_weights=copy(new_weights)
            #print 'counter = ',counter
            counter+=1
            
        self.weights=copy(old_weights)
        #print 'final weights are ',self.weights
        
        
    
    def test(self,X):
        labels=[]
        
        self.X_test=copy(X)
        self.X_test=np.hstack((np.ones((self.X_test.shape[0],1)),self.X_test))
        for i in range(0,self.X_test.shape[0]):
            y1=self.logit2(self.X_test[i],self.weights)
           
            if y1 >= 0.5:
                labels.append(1)
            else:
                labels.append(0)
        
        labels=np.asarray(labels)
        return labels
     

       
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
        

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

             
            self.train(trainX,trainy,0.001,20)
            #print 'Now testing------>'
            predicted=self.test(testX)
            accuracy=self.checkAccuracy(predicted,testY)  
            accuracies.append(accuracy)
            print 'accuracy is ',accuracy
                
        return  np.mean(accuracies)        
                
gmm=GMM()


accuracy=gmm.crossValidate(K=10)
print 'accuracy is ',accuracy,'%'


                           
        
            
