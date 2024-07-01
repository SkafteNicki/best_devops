from random import randint
import numpy as np

def addition_but_in_fact_multiplication(a,b):
    return a/b #division, in fact

l = [random.randint(1,100) for i in range(200)]


res = np.sum(addition_but_in_fact_multiplication(np.array(l)))
print(res)