
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random
import seaborn as sns
##
##generating data:
# df=pd.DataFrame()
#
# X1=np.random.uniform(low=-40, high=40, size=200)
# X2=np.random.uniform(low=-40, high=40, size=200)
# Y=np.zeros(200)
# df['X1']=X1
# df['X2']=X2
# df['Y']=Y
#
# for i in range (200):
#     y=df['X1'].iloc[i]+3*df['X2'].iloc[i]-2
#     if y>0:
#         df['Y'].iloc[i]=1
#     if y<=0:
#         df['Y'].iloc[i]=-1
#
# # Storing data:
# df.to_csv('file2.csv',index=False)
#Reading data:
new_df=pd.read_csv('file2.csv')


##Perceptron function:
def perceptron(w,x,bias):
    o=np.dot(w,x)+bias
    return o
def classify(o):
    if o>0:
        return 1
    else:
        return -1


##Vector of weights:
# weights=np.random.uniform(low=0, high=1, size=2)
# bias=np.random.uniform(low=0, high=1, size=1)
# open_file = open('weights.pkl', "wb")
# pickle.dump(weights, open_file)
# open_file.close()
#
# open_file = open('bias.pkl', "wb")
# pickle.dump(bias, open_file)
# open_file.close()
open_file = open('weights.pkl', "rb")
weights = pickle.load(open_file)
open_file.close()
open_file = open('bias.pkl', "rb")
bias = pickle.load(open_file)
bias=bias[0]
open_file.close()


##PARAMETERS


##Problem 1
Ib=new_df[['X1','X2']]
n=len(Ib)
Y=new_df['Y']
Weights=[]
biaslist=[]
# standardize features
I= Ib.copy()
I['X2']=np.array((Ib['X2'] - Ib['X2'].mean())/ Ib['X2'].std())
I['X1'] = (Ib['X1'] - Ib['X1'].mean()) / Ib['X1'].std()
##Printing data:
# plt.figure(6)
# # color= ['red' if l == -1 else 'green' for l in Y]
# # plt.scatter(I['X1'], I['X2'], color=color)
# # plt.legend(["x*2" , "x*3"], ncol = 2 , loc = "lower right")
# sns.scatterplot(x=Ib['X1'], y=Ib['X2'], hue=Y,palette="deep")
# plt.title("Data points in 2D")
# plt.show()
##Qa
def plot_train_error(I,Y,epochs,bias,n,lr):
    w=weights.copy()
    errors=[]
    epochsl=[]
    for epoch in range(epochs):
        error=0
        errorlrn=[0,0]
        errorbias=0
        for i in range(n):
            x=np.array(I.iloc[i])
            o=perceptron(w,x,bias)
            errorbias=errorbias+lr*(t-o)
            # w=w+lr*(t-o)*x
            # bias=bias+lr*(t-o)
        w=w+errorlrn
        bias=bias+errorbias
        for i in range(n):
            xt=np.array(I.iloc[i])
            ot=classify(perceptron(w,xt,bias))
            tt=Y.iloc[i]
            if tt*ot<0:
                error=error+1
        #print(error)
        errors.append(error/n)
        epochsl.append(epoch+1)
        # print(w)
        # print(bias)
    plt.figure(1)
    plt.plot(epochsl,errors)
    # print(errors)
    plt.title('The training error E vs number of epochs (25 epochs) with a learning rate =0.001')
    plt.xlabel('Epochs')
    plt.ylabel('Training error')
    plt.show()

##Qb
def plotdecisionsurface(I,Y,epochs,bias,n,lr):
    w=weights.copy()
    for epoch in range(epochs):
        error=0
        errorlrn=0
        errorbias=0
        for i in range(n):
            x=np.array(I.iloc[i])
            o=perceptron(w,x,bias)
            t=Y.iloc[i]
            #w=w+lr*(t-o)*x
            #bias=bias+lr*(t-o)
            errorlrn=errorlrn+lr*(t-o)*x
            errorbias=errorbias+lr*(t-o)
        w=w+errorlrn
        bias=bias+errorbias
        if epoch==5 or epoch==10 or epoch==50 or epoch==100:
            Weights.append(w)
            biaslist.append(bias)
    XD=[-45,45]
    YD=-np.dot(Weights[0][0]/Weights[0][1],XD)-biaslist[0]/Weights[0][1]
    #YD=-np.dot(Weights[0][0]/Weights[0][1],XD)
    plt.figure(2)
    plt.title("Decision surface for I=5, I=10, I=50, I=100 with red is class -1 and green is class 1")
    ax1=plt.subplot(2, 2, 1)
    plt.plot(XD,YD,'k')
    color= ['red' if l == -1 else 'green' for l in Y]
    plt.scatter(Ib['X1'], Ib['X2'], color=color)
    plt.fill_between(XD,YD,-40,alpha=0.30, color='red')
    plt.fill_between(XD,YD,45,alpha=0.30, color='green')

    YD=-np.dot(Weights[1][0]/Weights[1][1],XD)-biaslist[1]/Weights[1][1]
    #YD=-np.dot(Weights[1][0]/Weights[1][1],XD)
    ax2=plt.subplot(2, 2, 2)
    ax2.plot(XD,YD,'k')
    color= ['red' if l == -1 else 'green' for l in Y]
    ax2.scatter(Ib['X1'], Ib['X2'], color=color)
    ax2.fill_between(XD,YD,-45,alpha=0.30, color='red')
    ax2.fill_between(XD,YD,45,alpha=0.30, color='green')

    YD=-np.dot(Weights[2][0]/Weights[2][1],XD)-biaslist[2]/Weights[2][1]
    #YD=-np.dot(Weights[2][0]/Weights[2][1],XD)
    ax3=plt.subplot(2, 2, 3)
    plt.plot(XD,YD,'k')
    color= ['red' if l == -1 else 'green' for l in Y]
    plt.scatter(Ib['X1'], Ib['X2'], color=color)
    plt.fill_between(XD,YD,-45,alpha=0.30, color='red')
    plt.fill_between(XD,YD,45,alpha=0.30, color='green')

    YD=-np.dot(Weights[3][0]/Weights[3][1],XD)-biaslist[3]/Weights[3][1]
    #YD=-np.dot(Weights[3][0]/Weights[3][1],XD)
    ax4=plt.subplot(2, 2, 4)
    plt.plot(XD,YD,'k')
    color= ['red' if l == -1 else 'green' for l in Y]
    plt.scatter(Ib['X1'], Ib['X2'], color=color)
    plt.fill_between(XD,YD,-45,alpha=0.30, color='red')
    plt.fill_between(XD,YD,45,alpha=0.30, color='green')
    ax1.title.set_text('Decision surface for I=5')
    ax2.title.set_text('Decision surface for I=10')
    ax3.title.set_text('Decision surface for I=50')
    ax4.title.set_text('Decision surface for I=100')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax4.set_xlabel('X1')
    ax4.set_ylabel('X2')
    plt.show()
# # # ##Qc
def plot_train_error_lr_batch(I,Y,epochs,randomw,biasrandom,n):
    lr=1
    errorslr=[]
    lrl=[0.1,0.01,0.001,0.0001]
    for i in range (4):
        w=randomw.copy()
        epochsl=[]
        errors=[]
        bias=biasrandom
        lr=lrl[i]
        for epoch in range(epochs):
            error=0
            errorlrn=0
            errorbias=0

            for i in range(n):
                x=np.array(I.iloc[i])
                o=perceptron(w,x,bias)
                t=Y.iloc[i]
                errorbias=errorbias+lr*(t-o)
            w=w+errorlrn
            bias=bias+errorbias
            for i in range(n):
                xt=np.array(I.iloc[i])
                ot=classify(perceptron(w,xt,bias))
                tt=Y.iloc[i]
                if tt*ot<0:
                    error=error+1
            errors.append(error/n)
            epochsl.append(epoch+1)
        errorslr.append(errors)
    plt.figure(3)
    ax1=plt.subplot(2, 2, 1)
    ax1.plot(epochsl,errorslr[0])
    ax2=plt.subplot(2, 2, 2)
    ax2.plot(epochsl,errorslr[1])
    ax3=plt.subplot(2, 2, 3)
    ax3.plot(epochsl,errorslr[2])
    ax4=plt.subplot(2, 2, 4)
    ax4.plot(epochsl,errorslr[3])
    ax1.title.set_text('Training Error with lr=0.1')
    ax2.title.set_text('Training Error with lr=0.01')
    ax3.title.set_text('Training Error with lr=0.001')
    ax4.title.set_text('Training Error with lr=0.0001')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Error')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Training Error')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Training Error')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Training Error')
    plt.show()
##Qd
def plot_train_error_lr_stoch(I,Y,epochs,randomw,biasrandom,n):
    lrl=[0.1,0.01,0.001,0.0001]
    errorslr=[]
    for i in range (4):
        w=randomw.copy()
        epochsl=[]
        errors=[]
        bias=biasrandom
        lr=lrl[i]
        for epoch in range(epochs):
            error=0
            for i in range(n):
                x=np.array(I.iloc[i])
                w=w+lr*(t-o)*x
                bias=bias+lr*(t-o)
            for i in range(n):
                xt=np.array(I.iloc[i])
                ot=classify(perceptron(w,xt,bias))
                tt=Y.iloc[i]
                if tt*ot<0:
                    error=error+1
            errors.append(error/n)
            epochsl.append(epoch+1)
        errorslr.append(errors)
    plt.figure(4)
    ax1=plt.subplot(2, 2, 1)
    ax1.plot(epochsl,errorslr[0])
    ax2=plt.subplot(2, 2, 2)
    ax2.plot(epochsl,errorslr[1])
    ax3=plt.subplot(2, 2, 3)
    ax3.plot(epochsl,errorslr[2])
    ax4=plt.subplot(2, 2, 4)
    ax4.plot(epochsl,errorslr[3])
    ax1.title.set_text('Training Error with lr=0.1')
    ax2.title.set_text('Training Error with lr=0.01')
    ax3.title.set_text('Training Error with lr=0.001')
    ax4.title.set_text('Training Error with lr=0.0001')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Error')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Training Error')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Training Error')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Training Error')
    plt.show()

