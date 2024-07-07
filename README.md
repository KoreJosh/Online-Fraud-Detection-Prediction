# Online-Fraud-Detection-Prediction

![](https://github.com/KoreJosh/Online-Fraud-Detection-Prediction/assets/97749198/ee41ef93-eb0a-4eb5-b7b0-1e535094aeb9)

## Problem Statement
**Problem Statement: Online Fraud Prediction**

The rise of e-commerce and online transactions has brought about a corresponding increase in fraudulent activities, posing significant risks to businesses and consumers alike. Detecting and preventing online fraud in real-time is paramount to safeguarding financial assets, preserving trust in online platforms, and ensuring a secure digital environment.

The objective of this project is to develop an effective machine learning model for online fraud prediction. By analyzing transactional data, user behaviors, and other relevant features, the model aims to accurately identify fraudulent activities and distinguish them from legitimate transactions. This includes detecting various forms of fraud such as unauthorized access, identity theft, payment fraud, and account takeovers.

The ultimate goal is to deploy a scalable and robust fraud detection system that operates in real-time, allowing businesses to proactively mitigate risks and take appropriate actions to prevent financial losses and protect their customers. Through continuous monitoring and adaptive learning, the model will evolve to adapt to emerging fraud patterns and ensure ongoing effectiveness in combating online fraud.

## Data Source
The dataset contains 6362620 rows( each rows represent a patient), and 11 columns of various medical condtions.

## Tools
- Python's Jupter notebook, Scikit-Learn

## Data Cleaning / Preparation

- Data Loading and Inspection
  - Using the .shape(), it shows that the dataset has over 6 million rows and 11 columns
- Handling missing values and duplicates
  - Using .isnull().sum() function, no missing values was recorded, also using the .duplicated().sum() function, no duplicate values was recorded.
    
## Exploratory Data Analysis
Exploring the transaction type , the following amount of transaction were made based on each form of transaction:
|Transaction Type|Amount in Numbers|
|-----------|-----------|
|CASH_OUT |2237500|
|PAYMENT|2151495|
|CASH_IN|1399284|
|TRANSFER|532909|
|DEBIT|41432|


![distribution transaction type](https://github.com/KoreJosh/Online-Fraud-Detection-Prediction/assets/97749198/f934f950-8aa1-4e3e-af85-6696b9e72f8e)



### Transformation of the 'is fraud' column:
transforming the categorical features into numerical. 
Here I will also transform the values of the isFraud column 
into No Fraud and Fraud labels to have a better understanding of the output:

```
df['type']= df['type'].map({'CASH_OUT':1,'PAYMENT':2,'CASH_IN':3,'TRANSFER':4, 'DEBIT':5})
df["isFraud"] = df["isFraud"].map({0: "No Fraud", 1: "Fraud"})
```

### Online Payments Fraud Detection Model

# Splitting the dataset to dependent and independent variable.

```
x = np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(df[["isFraud"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Training using a Decision Tree Clasifier Model.
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

```
We got a model score of 99% using the model.score() function:
```
model.score(xtest, ytest)
```


## Result / Findings

With inputed details of the dependent features the price of thr house rent can be predicted.

```
#Classifing whether a transaction is fraud or not, using certain features
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])

```
In summary, our decision tree model for online fraud detection has delivered outstanding performance, achieving an impressive accuracy score of 99%. This outcome demonstrates the effectiveness of our predictive approach in swiftly identifying and preventing fraudulent activities in real-time. With this robust model at our disposal, businesses can fortify their fraud prevention measures, protecting valuable financial assets and ensuring the integrity of online transactions. As we persist in refining and enhancing our detection strategies, we remain steadfast in our dedication to combatting online fraud and nurturing a secure digital environment for all stakeholders involved.

