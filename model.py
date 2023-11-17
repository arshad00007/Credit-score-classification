# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Remove unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

# load  data
credit_df = pd.read_csv("credit.csv",low_memory=False)

# can drop ID, Name and SSN as they are identifiers and not useful for visualization
credit_df1 = credit_df.drop(['ID', 'Name', 'SSN'], axis=1)

#  drop, Num_Bank_Accounts, Type_of_Loan as they do not contribute to target
credit_df1.drop(['Num_Bank_Accounts','Type_of_Loan'],axis =1 , inplace = True )

# replace NM with nan
credit_df1.loc[credit_df1['Payment_of_Min_Amount'] == 'NM', 'Payment_of_Min_Amount'] = np.nan

# fill nan using mode within each group
credit_df1['Payment_of_Min_Amount'] = credit_df1.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(x.mode()[0]))



#Box plot of numerical columns

num_col = credit_df1.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(20,30))

for i, variable in enumerate(num_col):
                     plt.subplot(5,4,i+1)
                     plt.boxplot(credit_df1[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)

# drop the other columns
num_col1=credit_df1.drop(['Customer_ID','Month','Age','Num_Credit_Card','Interest_Rate','Num_of_Loan','Num_of_Delayed_Payment','Credit_History_Age','Credit_Score','Payment_of_Min_Amount','Payment_Behaviour','Credit_Mix','Occupation'], axis = 1)

# Identify the outliers and remove 

for i in num_col1:
    Q1=credit_df1[i].quantile(0.25) # 25th quantile
    Q3=credit_df1[i].quantile(0.75) # 75th quantile
    IQR = Q3-Q1
    Lower_Whisker = Q1 - 1.5*IQR 
    Upper_Whisker = Q3 + 1.5*IQR
    credit_df1[i] = np.clip(credit_df1[i], Lower_Whisker, Upper_Whisker)

# drop Customer ID
credit_df1 = credit_df1.drop('Customer_ID', axis=1)

# drop payment behaviour
credit_df1 = credit_df1.drop('Payment_Behaviour', axis=1)

# ONE HOT ENCODING
replace_map = {'Payment_of_Min_Amount': {'Yes': 1, 'No': 0} }

credit_df1.replace(replace_map, inplace=True)

   # It's replacing 'Yes' with 1 and 'No' with 0 in that specific column

# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()

for i in ['Occupation']:  
    credit_df1[i]=label_encoder.fit_transform(credit_df1[i])
    le_name_mapping =dict((zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print(le_name_mapping)


credit_df1=credit_df1.drop('Credit_Mix',axis=1)

# Mapping Credit score
replace_map = {'Credit_Score': {'Poor': 0, 'Good': 2, 'Standard': 1 }}
credit_df1.replace(replace_map, inplace=True)

#scale_cols = ['Age',  'Annual_Income', 'Monthly_Inhand_Salary',
  #     'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
   #    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
  #     'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio','Credit_History_Age',
   #    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

#from sklearn.preprocessing import StandardScaler

#std = StandardScaler()

#credit_df1[scale_cols]= std.fit_transform(credit_df1[scale_cols])


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Select features and target
X = credit_df1.drop('Credit_Score', axis=1)  # Features
y = credit_df1['Credit_Score']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

from sklearn.metrics import classification_report

import xgboost as xgb
xgboost = xgb.XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=100, num_class=3)
xgboost.fit(X_train, y_train)
xgb_pred = xgboost.predict(X_test)
print(classification_report(y_test, xgb_pred))



#Serialize the python object using pickle
import pickle
pickle.dump(xgboost, open('xgboost.pkl', 'wb'))

pickle.dump(X_train, open('X_train.pkl', 'wb'))

print(xgboost.predict(X_test))


print(X_train.columns)
