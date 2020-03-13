
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('C:\\Users\shashikant\Desktop\Decision Tree\decision2.csv')
df

dummy_company = pd.get_dummies(df.company)
dummy_company


dummy_job = pd.get_dummies(df.job)
dummy_job

dummy_degree = pd.get_dummies(df.degree)
dummy_degree

df1 = pd.concat([df,dummy_company],axis = 'columns')
df1

df2 = pd.concat([df1,dummy_job],axis = 'columns')
df2

df3 = pd.concat([df2,dummy_degree],axis = 'columns')
df3

new_df = df3
new_df

x = new_df.drop(['salary','company','job','degree'],axis='columns').values
x

y = new_df[['salary']].values
y


new_df

model = DecisionTreeRegressor()

model.fit(x,y)

model.score(x,y)

model.predict([[0, 1, 0, 0, 1, 0, 0, 1]])






