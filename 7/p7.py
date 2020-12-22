#Bayesian Network Program

import bayespy as bp
import numpy as np
import csv
from colorama import init
from colorama import Fore, Back, Style

init()
# Define Parameter Enum values
#Age
ageEnum = {'SuperSeniorCitizen':0, 'SeniorCitizen':1, 'MiddleAged':2,
'Youth':3, 'Teen':4}
# Gender
genderEnum = {'Male':0, 'Female':1}
# FamilyHistory
familyHistoryEnum = {'Yes':0, 'No':1}
# Diet(Calorie Intake)
dietEnum = {'High':0, 'Medium':1, 'Low':2}
# LifeStyle
lifeStyleEnum = {'Athlete':0, 'Active':1, 'Moderate':2, 'Sedetary':3}
# Cholesterol
cholesterolEnum = {'High':0, 'BorderLine':1, 'Normal':2}
# HeartDisease
heartDiseaseEnum = {'Yes':0, 'No':1}
#heart_disease_data.csv
with open('7\heart_disease_data.csv') as csvfile:
 lines = csv.reader(csvfile)
 dataset = list(lines)
 data = []

for x in dataset:
    data.append([ageEnum[x[0]],genderEnum[x[1]],familyHistoryEnum[x[2]],dietEnum[x[3]],lifeStyleEnum[x[4]],cholesterolEnum[x[5]],heartDiseaseEnum[x[6]]])

# Training data for machine learning todo: should import from csv
data = np.array(data)

print (data)
N = len(data)
print(f"N={N}")

# Input data column assignment
p_age = bp.nodes.Dirichlet(1.0*np.ones(5))
print(f'p_age={p_age}')
age = bp.nodes.Categorical(p_age, plates=(N,))
print(f"age={age}")
age.observe(data[:,0])
print(f"OBSERVE AGE{age.observe(data[:,0])}")
p_gender = bp.nodes.Dirichlet(1.0*np.ones(2))
print(f"p_gender={p_gender}")
gender = bp.nodes.Categorical(p_gender, plates=(N,))
print(f"gender={gender}")
gender.observe(data[:,1])
p_familyhistory = bp.nodes.Dirichlet(1.0*np.ones(2))
print(f"p_familyhistory={p_familyhistory}")
familyhistory = bp.nodes.Categorical(p_familyhistory, plates=(N,))
print(f"familyhistory={familyhistory}")
familyhistory.observe(data[:,2])
p_diet = bp.nodes.Dirichlet(1.0*np.ones(3))
print(f"p_diet={p_diet}")
diet = bp.nodes.Categorical(p_diet, plates=(N,))
print(f"diet={diet}")
diet.observe(data[:,3])
p_lifestyle = bp.nodes.Dirichlet(1.0*np.ones(4))
print(f"p_lifestyle={p_lifestyle}")
lifestyle = bp.nodes.Categorical(p_lifestyle, plates=(N,))
print(f"lifestyle={lifestyle}")
lifestyle.observe(data[:,4])
p_cholesterol = bp.nodes.Dirichlet(1.0*np.ones(3))
print(f"p_cholesterol={p_cholesterol}")
cholesterol = bp.nodes.Categorical(p_cholesterol, plates=(N,))
print(f"cholesterol={cholesterol}")
cholesterol.observe(data[:,5])
#print(data)
# Prepare nodes and establish edges
# np.ones(2) -> HeartDisease has 2 options Yes/No
# plates(5, 2, 2, 3, 4, 3) -> corresponds to options present for domain values
p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
print(f"p_heartdisease={p_heartdisease}")
heartdisease = bp.nodes.MultiMixture([age, gender, familyhistory, diet,
lifestyle, cholesterol], bp.nodes.Categorical, p_heartdisease)
#print(f"heartdisease={heartdisease}")
heartdisease.observe(data[:,6])
p_heartdisease.update()
print(data)

# Interactive Test
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