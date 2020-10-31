##importing the data
import pandas as pd
data1=pd.read_csv('E:\\assignment\\naive baise\\SalaryData_Train.csv')
data2=pd.read_csv('E:\\assignment\\naive baise\\SalaryData_Test.csv')

##merging the datasets
frame=[data1,data2]
data=pd.concat(frame)

data.info()

##checking for null  values
data.isnull().sum()

##so we do not have any null values in our dataset

## as we do have some catagorical column also, so we need to covert them to innnteger formal for calculations

##training dataset
data.age,_= pd.factorize(data.age)
data.workclass,_= pd.factorize(data.workclass)
data.education,_= pd.factorize(data.education)
data.maritalstatus,_= pd.factorize(data.maritalstatus)
data.occupation,_= pd.factorize(data.occupation)
data.relationship,_ = pd.factorize(data.relationship)
data.race,_ = pd.factorize(data.race)
data.sex,_ = pd.factorize(data.sex)
data.capitalgain,_ = pd.factorize(data.capitalgain)
data.capitalloss,_= pd.factorize(data.capitalloss)
data.hoursperweek,_ = pd.factorize(data.hoursperweek)
data.native,_ = pd.factorize(data.native)
data.Salary,_ = pd.factorize(data.Salary)

##importing traing and testing data 
x= data.iloc[:,:-1]
y= data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



##importing the model and fitting it to data
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

##parameters study,making predictions with the test data and accuracy test and
from sklearn import metrics 
y_pred= model.predict(x_test)
accuracy=metrics.accuracy_score(y_pred,y_test)

##we got an accuracy of 74%

from sklearn.metrics import confusion_matrix
conf_Matrix = confusion_matrix(y_test, y_pred)
print(conf_Matrix)
##so from our confusion matrix we conclude that
#true positive= 5132
#true negative= 1579
#false positive= 1730
#false negative=604

