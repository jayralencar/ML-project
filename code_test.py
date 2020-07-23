import numpy as np

a = np.random.dirichlet(np.ones(3),size=1)

print(a)
print(sum(a[0]))