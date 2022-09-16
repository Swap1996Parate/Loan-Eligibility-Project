#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[2]:


import os
os.chdir("D:\ETL Hive Data Science Lectures\Project  2\Project 2")


# # Reading data Set

# In[3]:


import pandas as pd
A = pd.read_csv("training_set (2).csv")


# In[4]:


A.head()


# # Missing data Treatment

# In[5]:


A.isna().sum()


# In[6]:


A.info()


# In[7]:


A.Credit_History


# In[8]:


A.Credit_History.mode()


# In[9]:


A.Credit_History = A.Credit_History.fillna(1.0)


# In[10]:


from p2module import replacer
replacer(A)


# In[11]:


A.isna().sum()


# # Define Y

# In[12]:


Y = A[["Loan_Status"]]


# # Dropping Statistically unimportant 

# In[13]:


A.head()


# In[14]:


X = A.drop(labels=["Loan_ID","Loan_Status"],axis=1)


# # Exploratory data Analysis(EDA)

# In[15]:


cat = []
con = []
for i in X.columns:
    if(X[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[ ]:





# In[16]:


imp_cols = []
from p2module import ANOVA,chisquare
for i in con:
    q = ANOVA(A,"Loan_Status",i)
    print("-------------")
    print("Loan_Status vs",i)
    print("Pval: ",q)
    if(q < 0.15):
        imp_cols.append(i)


# In[ ]:





# In[17]:


for i in cat:
    q = chisquare(A,"Loan_Status",i)
    print("-------------")
    print("Loan_Status vs",i)
    print("Pval: ",q)
    if(q < 0.15):
        imp_cols.append(i)


# In[18]:


imp_cols


# # Preprocessing

# In[19]:


X.skew()


# In[20]:


from numpy import log

def skew_rem(df,col):
    q = []
    for i in df[col]:
        if(i !=0):
            q.append(log(i))
        else:
            q.append(i)
    df[col] = q


# In[21]:


skew_rem(X,'ApplicantIncome')
skew_rem(X,'CoapplicantIncome')


# In[22]:


X.skew()


# In[23]:


from p2module import preprocessing
Xnew = preprocessing(X[imp_cols])


# In[24]:


Xnew.head()


# # Dividing Data into Training and Testing Splits

# In[25]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Create a Multiple Linear Regression Model

# In[26]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

model = lr.fit(xtrain,ytrain)


# # Create Prediction

# In[27]:


pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)


# # Error | Loss Overfitting

# In[28]:


from sklearn.metrics import accuracy_score 
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[29]:


tr_acc


# In[ ]:





# # Try Tree Model

# In[30]:


X = A.drop(labels=["Loan_ID","Loan_Status"],axis=1)
Xnew = preprocessing(X)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model = dtc.fit(xtrain,ytrain)

pred_tr = model.predict(xtrain)
prde_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[31]:


tr_acc


# In[32]:


ts_acc


# In[33]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
def tree(dtc):
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    return tr_acc,ts_acc


# In[34]:


tree(dtc)


# In[35]:


tr = []
ts = []
for i in range(1,30,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,max_depth=i)
    p,q = tree(dtc)
    tr.append(p)
    ts.append(q)


# In[36]:


import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# In[37]:


tr = []
ts = []
for i in range(1,100,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,min_samples_leaf=i)
    p,q = tree(dtc)
    tr.append(p)
    ts.append(q)
    
import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)
plt.xticks(range(0,100,5))


# In[38]:


tr = []
ts = []
for i in range(2,30,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,min_samples_split=i)
    p,q = tree(dtc)
    tr.append(p)
    ts.append(q)
    
import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# # Try Adaboost

# In[39]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(DecisionTreeClassifier(random_state = 21,max_depth=2),n_estimators=30)
tree(abc)


# In[40]:


for i in range(2,50):
    abc = AdaBoostClassifier(DecisionTreeClassifier(random_state=21,max_depth=2),n_estimators=i)
    print(tree(abc))


# # Prediction by using best model
##Read testing Data
# In[41]:


B = pd.read_csv("testing_set (2).csv")


# In[42]:


B.head()


# In[43]:


X = B.drop(labels=["Loan_ID"],axis=1)
replacer(X)
Xnew = preprocessing(X)
pred =  model.predict(Xnew)


# In[ ]:





# In[ ]:





# In[44]:


X = B.drop(labels=["Loan_ID"],axis=1)
replacer(X)
Xnew = preprocessing(X)
pred = model.predict(Xnew)


# In[45]:


B['Loan_Status_pred'] = pred


# In[46]:


B.head()


# In[47]:


pd.DataFrame([Xnew.columns,dtc.feature_importances_]).T


# # Prepare Training data

# In[48]:


P1 = A[A.Loan_Status == "Y"]


# In[49]:


P2 = B[B.Loan_Status_pred == "Y"]


# In[50]:


P2 = P2.rename({"Loan_Status_Pred":"Loan_Status"},axis=1)


# In[51]:


trd = pd.concat([P1,P2])


# # Missing Data Treatment

# In[52]:


replacer(trd)


# In[53]:


trd.isna().sum()


# In[54]:


cat = []
con = []
for i in trd.columns:
    if(trd[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[55]:


cat


# In[56]:


con


# In[57]:


cat.remove("Loan_ID")
cat.remove("Loan_Status")
con.remove("LoanAmount")


# # define X and Y

# In[58]:


Y = trd[["LoanAmount"]]
X = trd.drop(labels=["LoanAmount","Loan_ID","Loan_Status"],axis=1)
from sklearn.preprocessing import StandardScaler
ss1 = StandardScaler()
X1 = pd.DataFrame(ss1.fit_transform(X[con]),columns=con)
X2 = pd.get_dummies(X[cat])
X2.index = range(len(X2))
Xnew = X1.join(X2)


# In[59]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# In[60]:


trd.corr()[["LoanAmount"]]


# In[61]:


for i in X.columns:
    if(X[i].dtypes == "object"):
        print("------------------------")
        print("Loan AMt vs",i)
        print(ANOVA(trd,i,"LoanAmount"))


# # create Multilinear Regression model

# In[62]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain,ytrain)


# In[63]:


def find_overfit_con(model_obj,xtrain,xtest,ytrain,ytest):
    model = model_obj.fit(xtrain,ytrain)
    pred_ts = model.predict(xtest)
    pred_tr = model.predict(xtrain)
    from sklearn.metrics import mean_absolute_error
    print("training error:",mean_absolute_error(ytrain,pred_tr))
    print("testing error:",mean_absolute_error(ytest,pred_ts))


# In[64]:


find_overfit_con(lm,xtrain,xtest,ytrain,ytest)


# # Prepare Data for Making Prediction

# In[65]:


test = B[B.Loan_Status_pred == "N"]


# In[66]:


test2 = test.drop(labels=["Loan_ID","LoanAmount"],axis=1)


# In[67]:


replacer(test2)
test2.isna().sum()


# In[68]:


from p2module import catconsep
cat,con = catconsep(test2)


# In[69]:


Xnew.shape


# In[70]:



Xt1 = pd.DataFrame(ss1.transform(test2[con]),columns=con)
Xt2 = pd.get_dummies(test2[cat])
Xt2.index = range(len(Xt2))
Xtnew = Xt1.join(Xt2)


# In[71]:


Xtnew.shape


# In[72]:


pred3=model.predict(Xtnew)


# In[ ]:





# In[ ]:





# In[73]:


pred4 = []
for i in range(len(pred3)):
    pred4.extend(pred3[i])


# In[74]:


pred_dict = {"Loan_ID":test.Loan_ID,"LoanAmount_pred": pred4}


# In[75]:


Df = pd.DataFrame(pred_dict)


# In[76]:


len(pred4)


# In[78]:


len(test.Loan_ID)


# In[86]:


Df.to_csv("Prediction_Project2.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




