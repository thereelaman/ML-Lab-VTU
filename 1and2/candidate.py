import numpy as np
import pandas as pd
 
data = pd.DataFrame(data = pd.read_csv("1and2/finds.csv"))
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])
 
def learn(concepts,target):
    specific_h = concepts[0].copy()
    rangeH = range(len(specific_h))

    general_h = [["?" for i in rangeH] for i in rangeH]
    
    for i,h in enumerate(concepts):
        if target[i] == "Yes":
            for x in rangeH:
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"    
 
        if target[i] == "No":
            for x in rangeH:
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"    
        
    while ['?','?','?','?','?','?'] in general_h:
        general_h.remove(['?','?','?','?','?','?'])
 
    return specific_h, general_h
 
s_final, g_final = learn(concepts,target)
print("Final S: ",s_final)
print("Final G: ",g_final)