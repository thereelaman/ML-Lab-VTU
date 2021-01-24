import bayespy as bp
import numpy as np
import pandas as pd

ageEnum = {'SuperSeniorCitizen':0, 'SeniorCitizen':1, 'MiddleAged':2, 'Youth':3, 'Teen':4}
genderEnum = {'Male':0, 'Female':1}
familyHistoryEnum = {'Yes':0, 'No':1}
dietEnum = {'High':0, 'Medium':1, 'Low':2}
lifeStyleEnum = {'Athlete':0, 'Active':1, 'Moderate':2, 'Sedetary':3}
cholesterolEnum = {'High':0, 'BorderLine':1, 'Normal':2}
heartDiseaseEnum = {'Yes':0, 'No':1}

data = []
lines = np.array(pd.read_csv('7\heart_disease_data.csv'))
dataset = list(lines)

for x in dataset:
    data.append([ageEnum[x[0]],genderEnum[x[1]],familyHistoryEnum[x[2]],dietEnum[x[3]],lifeStyleEnum[x[4]],cholesterolEnum[x[5]],heartDiseaseEnum[x[6]]])

data = np.array(data)
N = len(data)

def dataAssignment(n_class, col_pos):
    p_data = bp.nodes.Dirichlet(1.0*np.ones(n_class))
    assignedData = bp.nodes.Categorical(p_data, plates=(N,))
    assignedData.observe(data[:,col_pos])
    return assignedData


# Input data column assignment
age = dataAssignment(5,0)
gender = dataAssignment(2,1)
familyHistory = dataAssignment(2,2)
diet = dataAssignment(3,3)
lifeStyle = dataAssignment(4,4)
cholesterol = dataAssignment(3,5)

# Prepare nodes and establish edges
# np.ones(2) -> HeartDisease has 2 options Yes/No
# plates(5, 2, 2, 3, 4, 3) -> corresponds to options present for domain values

p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = bp.nodes.MultiMixture([age, gender, familyHistory, diet, lifeStyle, cholesterol], bp.nodes.Categorical, p_heartdisease)

heartdisease.observe(data[:,6])
p_heartdisease.update()

m = 0
while m == 0:
    print("\n")
    res = bp.nodes.MultiMixture([
    int(input('Enter Age: ' + str(ageEnum) + " : ")), 
    int(input('Enter Gender: ' + str(genderEnum) + " : ")),
    int(input('Enter FamilyHistory: ' + str(familyHistoryEnum)+ " : ")),
    int(input('Enter dietEnum: ' + str(dietEnum)+ " : ")), 
    int(input('EnterLifeStyle: ' + str(lifeStyleEnum)+ " : ")), 
    int(input('Enter Cholesterol: '+ str(cholesterolEnum)+ " : "))
    ], bp.nodes.Categorical,p_heartdisease).get_moments()[0][heartDiseaseEnum['No']]
 
    print("Probability(HeartDisease) = " + str(res))
    #print(Style.RESET_ALL)
    m = int(input("Enter for Continue:0, Exit :1 "))