
##importing the data
import pandas as pd
data = pd.read_csv('E:\\assignment\\naive baise\\sms_raw_NB.csv',encoding='ISO-8859-1')



##exploring the data
data.describe()
data.info()



##total ham and spam counts inside the data
data.groupby('type').describe()


##making a new column detecting how long the text messages are:
data['length']=data['text'].apply(len)
data.head()
data.length.describe()

##visualising the data
data['length'].plot(bins=50,kind='hist')
data.hist(column='length' , by= 'type' , bins= 50)


##training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['text'],data['type'],test_size=0.20)


 #Instantiate the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. 
testing_data = count_vector.transform(X_test)

##Implementation of Naive Bayes Machine Learning Algorithm
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)


predictions = naive_bayes.predict(testing_data)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

##so we got a accuracy of 99%