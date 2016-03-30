import numpy as np
from collections import defaultdict 
import csv
import matplotlib.pylab as plt


np.set_printoptions(threshold='nan')

class GMM:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train=X
        self.Y_train=y
        #print 'np.where(y==0)[0] = ',np.where(y==0)[0].shape[0]
        #print 'np.where(y==1)[0] = ',np.where(y==1)[0].shape[0]
        c0=float(np.where(y==0)[0].shape[0])/float(y.shape[0])
        c1=float(np.where(y==1)[0].shape[0])/float(y.shape[0])
        
        #print 'c0= ',c0,' c1 = ',c1
        
        y0s=np.where(y==0)[0]
        y1s=np.where(y==1)[0]
        
        u0=X[y0s].sum(axis=0)/float(y0s.shape[0])
        u1=X[y1s].sum(axis=0)/float(y1s.shape[0])
        
        #print 'u0= ',u0,' u1 = ',u1
        
        cov1=np.dot((X[y0s]-u0).T,X[y0s]-u0)/float(X.shape[0])
        cov2=np.dot((X[y1s]-u1).T,X[y1s]-u1)/float(X.shape[0])
        
        covariance_matrix=cov1+cov2
        #print 'the shape of cov1 is ',cov1.shape,' cov2 is ',cov2.shape
        
        self.c0=c0
        self.c1=c1
        self.u0=u0
        self.u1=u1
        self.covariance_matrix=covariance_matrix
        #return c0,u0,c1,u1,covariance_matrix
    
      
    def calculateNormalProbability(self,x,u,sigma):
        
        t1=np.dot(np.linalg.inv(sigma),(x-u).T)
        t2=np.dot((x-u),t1)
        return np.exp(-0.5*t2)  
        
    def test(self,X):
        labels=[]
        #print 'c0 ',self.c0
        #print 'c1 ',self.c1
        #print 'u0 is ',self.u0
        #print 'u1 is ',self.u1
        #print 'covariance_matrix is ',self.covariance_matrix
        
        #now we have all th parameters, we just need to find the accruacy by bayes rule 
        #P(C=1|X)=kP(c=1)*P(x|C=1)
        for i in range(0,X.shape[0]):
            y1=self.c1 * self.calculateNormalProbability(X[i],self.u1,self.covariance_matrix)
            y0=self.c0 * self.calculateNormalProbability(X[i],self.u0,self.covariance_matrix)
            if y1/y0 >=  1:
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
        self.train(trainX,trainy)
        predic
        #print 'the weights after training is ',self.weights 
        print 'after retraining the parameters are'
        w=np.dot(np.linalg.inv(self.covariance_matrix),self.u0-self.u1)
        w0=-0.5*np.dot(np.dot(self.u0.T,np.linalg.inv(self.covariance_matrix)),self.u0) +0.5*np.dot(np.dot(self.u1.T,np.linalg.inv(self.covariance_matrix)),self.u1) + np.log(self.c0/self.c1) 
        print 'w0 ',w0
        print 'w ',w
        
    
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

             
            self.train(trainX,trainy)
            #print 'Now testing------>'
            predicted=self.test(testX)
            accuracy=self.checkAccuracy(predicted,testY)  
            accuracies.append(accuracy)
         
                
        return  np.mean(accuracies)        
                
gmm=GMM()


accuracy=gmm.crossValidate(K=10)
print 'accuracy is ',accuracy,'%'
gmm.retrain()

                           
        
            
