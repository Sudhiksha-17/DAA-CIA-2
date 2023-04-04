import random
import numpy as np

def fitnessfunc(a,g):
    cost = 0
    for i in range(len(a)-1):
        cost += g[a[i]-1][a[i+1]-1]
    return cost

def cross(a,b):
    r1 = random.randint(2,5)
    r2 = random.randint(2,5)
    i = a.index(r1)
    j = b.index(r1)
    k = a.index(r2)
    l = b.index(r2)
    temp = a[i]
    a[i] = b[l]
    b[l] = temp
    temp = a[k]
    a[k] = b[j]
    b[j] = temp
    return tuple(a),tuple(b)

def mutate(a):
    r1 = random.randint(2,5)
    r2 = random.randint(2,5)
    temp = a[r1]
    a[r1] = a[r2]
    a[r2] = temp
    return tuple(a)


iter = 400
g = [[0,10,4,13,6,27],
[10,0,20,72,17,28],
[7,20,0,15,89,14],
[20,19,15,0,30,32],
[9,69,13,30,0,5],
[10,24,54,32,5,0]]
chrm = {}

for i in range (100):
    l=[]
    l.append(1)
    
    while True:
        a = random.randint(2,6)
        
        if a not in l:
            l.append(a)
            
        if len(l) == 6:
            break
        
    l.append(1)
    l = tuple(l)
    chrm[l] = fitnessfunc(l, g)
    
for j in range(100):
    
    l1 = list(chrm.keys())
    for i in range(0,len(chrm)-1,2):
        
        a,b = cross(list(l1[i]),list(l1[i+1]))
        chrm[a] = fitnessfunc(a, g)
        chrm[b] = fitnessfunc(b, g)
        
    l1 = list(chrm.keys())  
    for i in range(len(chrm)-1):
        a = mutate(list(l1[i]))
        chrm[a] = fitnessfunc(a, g)
        
    newpop = sorted(chrm, key = chrm.get)
    newpop = newpop[:100]
    new_pop = {}
    for i in newpop:
        new_pop[i] = chrm[i]

    chrm = new_pop
    
best = sorted(chrm , key = chrm.get)
print(best[0])
print(chrm[best[0]])
