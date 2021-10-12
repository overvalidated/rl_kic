#%%
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from statsmodels.tsa.stattools import pacf

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
#%%
LEN = 60
x = np.arange(LEN).reshape(-1, 1)
y = np.array(([1,1,1,0,0]*30)[:LEN]).reshape(-1, 1)
# y = np.random.random(size=(LEN, 1)).round()
model = DecisionTreeClassifier()
model.fit(x, y)
# print(y)
m = model.decision_path(x)
arr = np.array(m.todense())
counts = Counter([tuple(i) for i in arr])
total_sum, number = 0, 0
for i in counts:
    total_sum += sum(i)
    number += counts[i]
print('Average len:', total_sum/number)
print('pacf:', np.mean(np.abs(pacf(y))))
# %%

# %%
