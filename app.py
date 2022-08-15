#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
os.chdir('F:\webapp')


# In[17]:


pip install flask


# In[20]:


import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open('model/credit_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Gender = flask.request.form['Gender']
        Owned_Car = flask.request.form['Owned_Car']
        Owned_Realty = flask.request.form['Owned_Realty']
        Total_Children = flask.request.form['Total_Children']
        Total_Income = flask.request.form['Total_Income']
        Income_Type = flask.request.form['Income_Type']
        Education_Type = flask.request.form['Education_Type']
        Family_Status = flask.request.form['Family_Status']
        Housing_Type = flask.request.form['Housing_Type']
        Owned_Mobile_Phone = flask.request.form['Owned_Mobile_Phone']
        Owned_Work_Phone = flask.request.form['Owned_Work_Phone']
        Owned_Phone = flask.request.form['Owned_Phone']
        Owned_Email = flask.request.form['Owned_Email']
        Job_Title = flask.request.form['temperature']
        Total_Family_Members = flask.request.form['Total_Family_Members']
        Applicant_Age = flask.request.form['Applicant_Age']
        Years_of_Working = flask.request.form['Years_of_Working']
        Total_Bad_Debt = flask.request.form['Total_Bad_Debt']
        Total_Good_Debt = flask.request.form['Total_Good_Debt']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Gender,Owned_Car,Owned_Realty,Total_Children,Total_Income,Income_Type,Education_Type,Family_Status,Housing_Type,Owned_Mobile_Phone,Owned_Work_Phone,Owned_Phone,Owned_Email,Job_Title,Total_Family_Members,Applicant_Age,Years_of_Working,Total_Bad_Debt,Total_Good_Debt]],
                                       columns=['Gender','Owned_Car','Owned_Realty','Total_Children','Total_Income','Income_Type','Education_Type','Family_Status','Housing_Type','Owned_Mobile_Phone','Owned_Work_Phone','Owned_Phone','Owned_Email','Job_Title','Total_Family_Members','Applicant_Age','Years_of_Working','Total_Bad_Debt','Total_Good_Debt'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Gender':Gender,
                                                     'Owned_Car':Owned_Car,
                                                     'Owned_Realty':Owned_Realty,
                                                     'Owned_Realty':Owned_Realty,
                                                     'Total_Income':Total_Income,
                                                     'Income_Type':Income_Type,
                                                     'Education_Type':Education_Type,
                                                     'Family_Status':Family_Status,
                                                     'Housing_Type':Housing_Type,
                                                     'Owned_Mobile_Phone':Owned_Mobile_Phone,
                                                     'Owned_Work_Phone':Owned_Work_Phone,
                                                     'Owned_Phone':Owned_Phone,
                                                     'Owned_Email':Owned_Email,
                                                     'Job_Title':Job_Title,
                                                     'Total_Family_Members':Total_Family_Members,
                                                     'Applicant_Age':Applicant_Age,
                                                     'Years_of_Working':Years_of_Working,
                                                     'Total_Bad_Debt':Total_Bad_Debt,
                                                     'Total_Good_Debt':Total_Good_Debt
                                                     },
                                     result=prediction
                                     )

if __name__ == '__main__':
    app.run()


# In[ ]:




