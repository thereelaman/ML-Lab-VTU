import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv("1and2/finds.csv"))
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])


def learn(concepts, target):
    specific_h = concepts[0].copy()
    rangeH = range(len(specific_h))
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in rangeH:
            	if (h[x] != specific_h[x]):
                	specific_h[x] = "?"

    return (specific_h)


specific_h = learn(concepts, target)
print(specific_h)
