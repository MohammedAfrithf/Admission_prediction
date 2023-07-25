from flask import Flask,render_template, url_for, request , redirect
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import os

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import pandas as pd
import json

app=Flask(__name__,template_folder='admission',static_folder='admission')

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
  return render_template('predict.html')


@app.route('/calc',methods=['GET','POST'])
def calc():
  if request.method=="POST":
    location=request.form['location']
    cutoff=request.form['cutoff']
    department=request.form['department']
    autonomous=request.form['autonomous']
    datapreprocessing(location,cutoff,department,autonomous)    
  return render_template('answer.html',college=out_college,chance=out_coa)


def datapreprocessing(location,cutoff,department,autonomous):
  with open('Admission.csv') as file:  
      df1=pd.read_csv(file)
      df2=df1.dropna()
      global label_name
      label_dist = preprocessing.LabelEncoder()
      label_dept = preprocessing.LabelEncoder()
      label_name = preprocessing.LabelEncoder()
      global dataf
      dataf=df2
      newdist=df2[['DISTRICT']].copy()
      newdept=df2[['Department']].copy()
      dataf.rename(columns={'DISTRICT': 'ENC_DISTRICT'}, inplace=True)
      dataf.rename(columns={'Department': 'ENC_Department'}, inplace=True)
      dataf['ENC_DISTRICT']= label_dist.fit_transform(df2['ENC_DISTRICT']) 
      dataf['ENC_Department']= label_dept.fit_transform(df2['ENC_Department'])
      dataf['College_Name']= label_name.fit_transform(df2['College_Name']) 
      newdist[['ENC_DISTRICT']]=dataf[['ENC_DISTRICT']]
      newdist.drop_duplicates()
      list_dist=newdist.set_index('DISTRICT').T.to_dict('records')
      dict_dist=list_dist[0]
      newdept[['ENC_Department']]=dataf[['ENC_Department']]
      newdept.drop_duplicates()
      list_dept=newdept.set_index('Department').T.to_dict('records')
      dict_dept=list_dept[0]
      in_cutoff=float(cutoff)
      in_autonomous=int(autonomous)
      in_location = dict_dist[location]
      in_department = dict_dept[department]
      input=[[in_location,in_department,in_cutoff]]
      in_college=predictCollegeName(input)
      input=[[in_college,in_location,in_department,in_cutoff]]
      global out_college
      global out_coa
      in_coa=predictChanceOfAdmit(input)
      accuracy=findaccuracy(ac1,ac2)
      print(accuracy)
      out_college =label_name.inverse_transform(in_college)
      out_college=out_college[0]
      out_coa = in_coa[0]
      out_coa=out_coa*100
      out_coa=round(out_coa,2)


def predictCollegeName(input):
  dv=dataf.iloc[:,[2,4,5]].values
  iv=dataf.iloc[:,1].values
  x_train, x_test, y_train, y_test = train_test_split(dv, iv, test_size=0.2,random_state=42)
  classifier = DecisionTreeClassifier(criterion="entropy",splitter="best") 
  classifier.fit(dv, iv)
  y_pred = classifier.predict(input)
  in_college=y_pred[0]
  global ac1
  ac1 =classifier.score(x_test,y_test)
  return y_pred


def predictChanceOfAdmit(input):
  x=dataf.iloc[:,[1,2,4,5]].values
  y=dataf.iloc[:,6].values
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
  regr = DecisionTreeRegressor(max_depth=7)
  regr.fit(x,y)
  y_pred=regr.predict(input)
  global ac2
  ac2=regr.score(x_test,y_test)
  return y_pred

def findaccuracy(ac1,ac2):
  return (ac1+ac2)/2


if __name__ == "__main__":
      app.secret_key = os.urandom(24)
      app.run()


