import matplotlib.pyplot as plt
import math
import random
import numpy as np
import global_var as gl

def is_valid(X,Y,mmp):
    n = len(X)
    for i in range(n):
        ok = False
        for j in range(n):
            if i == j:
                continue
            dis = (X[i]-X[j])**2
            dis += (Y[i] - Y[j])**2
            dis = math.sqrt(dis)
            if dis <= gl.max_flight_radius:
                ok = True
        if ok == False:
            return False
    return True