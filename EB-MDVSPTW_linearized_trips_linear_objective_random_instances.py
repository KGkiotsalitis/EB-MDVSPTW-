import gurobipy as gp #import gurobipy library in Python as gp
from gurobipy import GRB
import pandas as pd #import pandas library as pd. It offers data structures and operations for manipulating numerical tables and time series
import numpy as np #import numpy library. It adds support for large, multi-dimensional arrays and matrices
import os #provides functions for interacting with the operating system
import ast #library that processes trees of the Python abstract syntax grammar

print(gp.gurobi.version())

#Initialize the Gurobi model
model = gp.Model()
#model.Params.OutputFlag = 0
#model.Params.LogToConsole = 0
#model.setParam("OutputFlag", 0)

################
# INPUT
################

#2 vehicles
#K=(1,2) #set of vehicles
#O={1:11,2:12} #vehicles' origin depots. The origin depot of vehicle 1 is 11 and vehicle 2 is 12
#D={1:21,2:22} #vehicles' destination depots. The destination depot of vehicle 2 is 21 and vehicle 2 is 22
#V=(1,2,3,4,5,6) #set of all trips
#Vk={1:[1,2,3,4,5,6],2:[1,2,3,4,5,6]} #set of trips that can be performed per vehicle
#Z=(1,2,3,4) #set of chargers
#F=[1001,1011,1002,1012,1003,1013,1004,1014] #set of charging events
data_header_charg_event=np.loadtxt('Data/D2_S2_C10_a_charging_event_sequence.txt',max_rows=1,dtype=int)
data_main_body_charg_event=np.loadtxt('Data/D2_S2_C10_a_charging_event_sequence.txt',skiprows=1,dtype=int)
F_end = data_header_charg_event #last charging events at each charger
print('F_end',F_end)
omega={}
for i in range(0,len(data_main_body_charg_event)):
    omega[data_main_body_charg_event[i,0]] = data_main_body_charg_event[i,1]
print('omega',omega)
#F_no_end = [1001,1002,1003,1004] #charging events that are not the last ones in their charger
#omega={1001:1011,1002:1012,1003:1013,1004:1014} #sequence of charging events at the same charger
#l={O[1]: 0, O[2]: 0, 1: 20, 2: 420, 3: 40, 4: 440, 5:820, 6:840, D[1]: 800, D[2]: 800, 1001: 20, 1002: 420, 1003: 40, 1004: 440, 1011: 220, 1012: 640, 1013: 1040, 1014: 1240}
#u={O[1]: 20, O[2]: 20, 1: 240, 2: 640, 3: 260, 4: 1060, 5: 2820, 6:4840, D[1]: 6000, D[2]: 6000, 1001: 270, 1002: 670, 1003: 1090, 1004: 1990, 1011: 470, 1012: 870, 1013: 3790, 1014: 3890}
#N=(O[1],O[2], 1, 2, 3, 4, 5, 6, D[1], D[2], 1001, 1011, 1002, 1012, 1003, 1013, 1004, 1014)

#COMPUTE TRAVEL TIMES BY USING THE DISTANCES BETWEEN DIFFERENT LOCATIONS
#O_N = {i:[0,0] for i in N} #origin location of every node initialized as 0
#D_N = {i:[0,0] for i in N} #destination location of every node [for charging nodes and depots the origin and destination location is the same]
data_header=np.loadtxt('Data/D2_S2_C10_a_trips.txt',max_rows=1,dtype=int)
data_header_decimal=np.loadtxt('Data/D2_S2_C10_a_trips.txt',max_rows=1,dtype=float)
data_main_body=np.loadtxt('Data/D2_S2_C10_a_trips.txt',skiprows=1,dtype=int)
print('data_header',data_header)
print('data_header_decimal',data_header_decimal)
Vehicles=data_header[0]; Trips=data_header[1]; Charging_Events=data_header[2]; greek_l=data_header[3]; p_max=data_header[4]; p_min=data_header[5]; travel_cost=data_header[6]
r=data_header_decimal[7]; e_consumption = data_header_decimal[8] #kWh per km
#greek_l = 1 #unit waiting cost of a vehicle
#phi_max={k:1000 for k in K}# F max- kWh
#phi_min={k:10 for k in K}# F min-- kWh
r=float(r)  #charging rate in kWh of energy per minute
#inverse_r=0.12
inverse_r=float(1/r)
print('greek_l,r,inverse_r',greek_l,r,inverse_r)
K=tuple(np.arange(1,Vehicles+1))
phi_max={k:p_max for k in K}# F max- kWh
phi_min={k:p_min for k in K}# F min-- kWh
print('phi_min,phi_max',phi_min,phi_max)
print('Vehicles,Trips,Events,K',Vehicles,Trips,Charging_Events,K)
O={};D={};l={};u={};O_N={};D_N={};N=[]
print(data_main_body)
print(data_main_body[0,0])
for k in K:
    j=k
    k=k-1
    O[j]=data_main_body[k,0]
    D[j]=data_main_body[k+Vehicles,0]
    l[O[j]]=data_main_body[k,5]; u[O[j]]=data_main_body[k,6]
    l[D[j]] = data_main_body[k+Vehicles,5]; u[D[j]] = data_main_body[k+Vehicles,6]
    O_N[O[j]]=[data_main_body[k,1],data_main_body[k,2]]
    D_N[O[j]] = [data_main_body[k,3], data_main_body[k,4]]
    O_N[D[j]]=[data_main_body[k+Vehicles,1],data_main_body[k+Vehicles,2]]
    D_N[D[j]] = [data_main_body[k+Vehicles,3], data_main_body[k+Vehicles,4]]
    N.append(O[j]);N.append(D[j])
print('O_N,D_N',O_N,D_N)
print('l,u',l,u)
V_dict={};V=[];j=0
for i in range(2*len(K),2*len(K)+Trips):
    j=j+1
    V_dict[j]=data_main_body[i,0]
    l[V_dict[j]]=data_main_body[i,5]; u[V_dict[j]]=data_main_body[i,6]
    O_N[V_dict[j]]=[data_main_body[i,1], data_main_body[i,2]]
    D_N[V_dict[j]]=[data_main_body[i,3], data_main_body[i,4]]
    l[V_dict[j]]=data_main_body[i,5]; u[V_dict[j]]=data_main_body[i,6]
    N.append(V_dict[j]); V.append(V_dict[j])
print('O_N,D_N', O_N, D_N)
print('l,u', l, u)
F_dict={};F=[];j=0
for i in range(2*len(K)+Trips,2*len(K)+Trips+Charging_Events):
    j=j+1
    F_dict[j]=data_main_body[i,0]
    l[F_dict[j]]=data_main_body[i,5]; u[F_dict[j]]=data_main_body[i,6]
    O_N[F_dict[j]]=[data_main_body[i,1], data_main_body[i,2]]
    D_N[F_dict[j]]=[data_main_body[i,3], data_main_body[i,4]]
    l[F_dict[j]]=data_main_body[i,5]; u[F_dict[j]]=data_main_body[i,6]
    N.append(F_dict[j]);F.append(F_dict[j])
print('O_N,D_N', O_N, D_N)
print('l,u', l, u)

print('N',N)
print('V',V)
print('F',F)

F_no_end = [] #charging events that are not the last ones in their charger
for i in F:
    if i not in F_end:
        F_no_end.append(i)
print('F_no_end',F_no_end)

print('K,V,F,O,D',K,V,F,O,D)
print('O_N,D_N',O_N,D_N)
print('l,u',l,u)
Nk={};Vk={}
for k in K:
    print('k',k,[O[k],D[k]])
    Nk[k]=V+F+[O[k],D[k]]
    Vk[k]=V
#Nk={1:[O[1],1,2,3,4,5,6,1001,1011,1002,1012,1003,1013,1004,1014,D[1]],2:[O[2],1,2,3,4,5,6,1001,1011,1002,1012,1003,1013,1004,1014,D[2]]} #O[k],Vk[k],F,D[k]
print('Nk',Nk,'Vk',Vk)

print('O_N',O_N)
print('D_N',D_N)
#########################################################################
t_tilde={} #travel time to complete trip i
eta={} #consumed energy when performing task i

from scipy.spatial import distance
for i in V:
    latitude_of_node_i_start=O_N[i][0]
    longitude_of_node_i_start=O_N[i][1]
    latitude_of_node_i_end=D_N[i][0]
    longitude_of_node_i_end=D_N[i][1]
    t_tilde[i] = distance.euclidean([latitude_of_node_i_start,longitude_of_node_i_start],[latitude_of_node_i_end,longitude_of_node_i_end])#*(1/1000)
    print('lat_start,lon_start,lat_end,lon_end',i,latitude_of_node_i_start,t_tilde[i])
    eta[i] = e_consumption*distance.euclidean([latitude_of_node_i_start,longitude_of_node_i_start],[latitude_of_node_i_end,longitude_of_node_i_end])#*(1/1000)

print('t_tilde',t_tilde)
#for i in V:
    #print(t_tilde[i])
print('eta',eta)
#for i in V:
    #print(eta[i])
t={} #travel time between the end location of task i and the start location of task j
bijk={} #travel cost from i to j
theta={} #consumed energy when deadheading from task i to task j

for i in N:
    for j in N:
        latitude_of_node_i_end=D_N[i][0]
        longitude_of_node_i_end=D_N[i][1]
        latitude_of_node_j_start=O_N[j][0]
        longitude_of_node_j_start = O_N[j][1]
        t[i,j]=distance.euclidean([latitude_of_node_i_end,longitude_of_node_i_end],[latitude_of_node_j_start,longitude_of_node_j_start])#*(1/1000)
        bijk[i,j]=travel_cost*t[i,j]
        theta[i,j] = e_consumption * distance.euclidean([latitude_of_node_i_end, longitude_of_node_i_end],[latitude_of_node_j_start,longitude_of_node_j_start])#(1 / 1000)
        if i==j:
            t[i,j]=0; theta[i,j]=0

print('theta', theta)
print('t',t)

A={}
for k in K:
    A_a=[(O[k],j) for j in Nk[k] if j!=O[k] if u[j]>=l[O[k]]+t[O[k],j]] #arcs_from_origin_depot
    A_b=[(i,D[k]) for i in Nk[k] if i not in [O[k],D[k]]] #arcs_to_destination_depot
    A_c=[(i,j) for i in Vk[k] for j in Vk[k] if i!=j if l[i]+t_tilde[i]+t[i,j]<=u[j]] #arcs_from_trip_to_trip
    A_d=[(i,j) for i in Vk[k] for j in F if l[i]+t_tilde[i]+t[i,j]<=u[j]] #arcs_from_trip_to_charging event
    A_f=[(i,j) for i in F for j in Vk[k] if l[i]+t[i,j]<=u[j]] #arcs_from_chargingevent_to_trip
    A[k]=A_a+A_b+A_c+A_d+A_f
    A_a.clear();A_b.clear();A_c.clear();A_d.clear();A_f.clear()

print('arcs',A)
#A={1:A1_a+A1_b+A1_c+A1_d+A1_f,2:A2_a+A2_b+A2_c+A2_d+A2_f} #possible arcs per vehicle

q={i:0 for i in V} #closest charging node from the end of trip i
for i in V:
    distance=+np.infty
    for j in F:
        if t[i,j]<=distance:
            distance=t[i,j]
            q[i]=j
    print('theta',theta[i,q[i]])
print('q(i)',q)

M=10000000 #very large positive number

#VARIABLES

x={}; sigma={}; sigma_tilde={}; z={}
for k in K:
    for i,j in A[k]:
        x[i,j,k] = model.addVar(vtype=gp.GRB.BINARY, name='x%s' % str([i,j,k])) #binary flow variable where xijk=1 if vehicle k uses arc ij
        sigma[i,j,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-2 * M, ub=+2 * M,
                              name='sigma%s' % str([i,j,k]))  # binary flow variable where xijk=1 if vehicle k uses arc ij
        sigma_tilde[i,j,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-2 * M, ub=+2 * M,
                                    name='sigma_tilde%s' % str([i,j,k]))  # binary flow variable where xijk=1 if vehicle k uses arc ij
        if i in Vk[k]+F+[O[k]]:
            z[i,j,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000000000,name='z%s' % str([i,j,k]))  # objective function

T={}; e={}; e_bar={}; g={}
for k in K:
    for i in Nk[k]:
        T[i,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, name='T%s' % str([i,k]))
    for i in Vk[k]+F+[D[k]]:
        e[i,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000000000, name='e%s' % str([i,k])) #SOC of vehicle k when it arrives at node task i
    for i in Vk[k]+F+[O[k]]:
        e_bar[i,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000000000, name='e_bar%s' % str([i,k])) #SOC of vehicle k when it completes task i
    for i in Vk[k]+F:
        g[i,k] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000000000, name='g%s' % str([i,k]))  # SOC change of vehicle k when performing node task i

tau = model.addVars(F,K,vtype=gp.GRB.CONTINUOUS, lb=-10000000000, name='tau') #required time period to recharge vehicle k at charging event i
s_tilde = model.addVars(F,K,vtype=gp.GRB.CONTINUOUS, lb=-10000000000, name='s_tilde')
y = model.addVars(F,K,vtype=gp.GRB.BINARY, name='y')


#CONSTRAINTS
model.addConstrs( sum( sum ( x[i,j,k] for i,j in A[k] if j==jj) for k in K ) == 1 for jj in V) # eq.(8)
model.addConstrs( sum( sum ( x[i,j,k] for i,j in A[k] if j==jj) for k in K ) <= 1 for jj in F) # eq.(9)

model.addConstrs(sum(x[i, j, k] for i, j in A[k] if i == O[k]) == 1 for k in K)  # eq.(10)
model.addConstrs(sum(x[i, j, k] for i, j in A[k] if j == D[k]) == 1 for k in K)  # eq.(10)

model.addConstrs( sum(x[i,j,k] for i,j in A[k] if j==jj) == sum(x[j,i,k] for j,i in A[k] if j==jj) for k in K for jj in Vk[k]+F) #eq.(11)

model.addConstrs(tau[i,k]==(phi_max[k] - e[i,k])*inverse_r for i in F for k in K) #eq.(15)
model.addConstrs(T[i,k]>=l[i] for k in K for i in Nk[k]) #eq.(16)
model.addConstrs(T[i,k]<=u[i] for k in K for i in Nk[k]) #eq.(16)

model.addConstrs(e_bar[O[k],k]==phi_max[k] for k in K) #eq.(18)
model.addConstrs(e_bar[j,k]==e[j,k]-g[j,k] for k in K for j in Vk[k]+F) #eq.(19)

model.addConstrs(e[j,k]<=(e_bar[i,k]-theta[i,j])+(1-x[i,j,k])*M for k in K for i,j in A[k]) #eq.(21)
model.addConstrs(g[i,k]==eta[i] for k in K for i in Vk[k]) #eq.(22)
model.addConstrs(g[i,k]==e[i,k]-phi_max[k] for i in F for k in K) #eq.(23)
model.addConstrs(e[i,k]>=phi_min[k] for k in K for i in Vk[k]+F+[D[k]]) #eq.(24)
model.addConstrs(e_bar[i,k]>=phi_min[k] + float(theta[i,q[i]]) for k in K for i in Vk[k]) #eq.(25)
model.addConstrs( y[ii,k] == sum (x[i,j,k] for i,j in A[k] if i==ii) for ii in F for k in K) #eq.(26)

model.addConstrs(T[i,k]+t_tilde[i]+t[i,j]-T[j,k]+sigma[i,j,k]<=0 for k in K for i,j in A[k] if i in Vk[k]) #eq (31)
model.addConstrs(T[i,k]+t[i,j]-T[j,k]+sigma[i,j,k]<=0 for k in K for i,j in A[k] if i==O[k]) #eq (32)
model.addConstrs(T[i,k]+tau[i,k]+t[i,j]-T[j,k]+sigma[i,j,k]<=0 for k in K for i,j in A[k] if i in F) #eq.(33)
model.addConstrs(sigma[i,j,k]<=M*(1-x[i,j,k]) for k in K for i,j in A[k]) #eq.(34)
model.addConstrs(sigma[i,j,k]>=-M*(1-x[i,j,k]) for k in K for i,j in A[k]) #eq.(35)

model.addConstrs(e[j,k]>=e_bar[i,k]-theta[i,j]+sigma_tilde[i,j,k] for k in K for i,j in A[k]) #eq.(36)
model.addConstrs(sigma_tilde[i,j,k]<=M*(1-x[i,j,k]) for k in K for i,j in A[k]) #eq.(37)
model.addConstrs(sigma_tilde[i,j,k]>=-M*(1-x[i,j,k]) for k in K for i,j in A[k]) #eq.(38)
model.addConstrs(T[i,k0] + tau[i,k0] + s_tilde[i,k0] <= T[omega[i],k] + M*(1- sum( x[l,r,k] for l,r in A[k] if l==omega[i] ) ) for i in F_no_end for k in K for k0 in K) #eq.(39)
model.addConstrs(s_tilde[i,k0]<=M*(1-y[i,k0]) for i in F for k0 in K) #eq.(40)
model.addConstrs(s_tilde[i,k0]>=-M*(1-y[i,k0]) for i in F for k0 in K) #eq.(41)

model.addConstrs(z[i,j,k]<=M*x[i,j,k] for k in K for i,j in A[k] if i in Vk[k]+F+[O[k]]) #eq.(42)
model.addConstrs(z[i,j,k]>=-M*x[i,j,k] for k in K for i,j in A[k] if i in Vk[k]+F+[O[k]]) #eq.(43)
model.addConstrs(z[i,j,k]<=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+t_tilde[i]+t[i,j]))+M*(1-x[i,j,k]) for k in K for i,j in A[k] if i in Vk[k]) #eq.(44)
model.addConstrs(z[i,j,k]<=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+t[i,j]))+M*(1-x[i,j,k]) for k in K for i,j in A[k] if i==O[k]) #eq.(44b)
model.addConstrs(z[i,j,k]<=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+tau[i,k]+t[i,j]))+M*(1-x[i,j,k]) for k in K for i,j in A[k] if i in F) #eq.(45)
model.addConstrs(z[i,j,k]>=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+t_tilde[i]+t[i,j]))-(1-x[i,j,k])*M for k in K for i,j in A[k] if i in Vk[k]) #eq.(46)
model.addConstrs(z[i,j,k]>=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+t[i,j]))-(1-x[i,j,k])*M for k in K for i,j in A[k] if i==O[k]) #eq.(46b)
model.addConstrs(z[i,j,k]>=bijk[i,j]+greek_l*(T[j,k]-(T[i,k]+tau[i,k]+t[i,j]))-(1-x[i,j,k])*M for k in K for i,j in A[k] if i in F) #eq.(47)

#VALID INEQUALITIES
model.addConstrs(e[j,k]<=phi_max[k] for k in K for j in Vk[k]+F+[D[k]]) #eq.(45)
model.addConstrs(sum(y[i,k]*l[i] for k in K) <= T[omega[i],k] + M*(1- sum( x[l,r,k] for l,r in A[k] if l==omega[i] ) ) for i in F_no_end for k in K) #eq.(47)

#OBJECTIVE FUNCTION
obj = sum( sum(z[i,j,k] for i,j in A[k] if i in Vk[k]+F+[O[k]]) for k in K)

#Add objective function to model and declare that we solve a minimization problem
model.setObjective(obj,GRB.MINIMIZE)

# Solve the model and return results.
#model.params.NonConvex = 2  # allow to handle quadratic equality constraints - which are always non-convex
"""
model.optimize()
if model.status == GRB.OPTIMAL:  # check if the solver is capable of finding an optimal solution
    model.printAttr('X')
    print(model.status, 'optimal')
    print('Obj: %g' % model.objVal)
else:
    print(model.status, 'not optimal')

#model.computeIIS()
#model.write("model.ilp")
# print results
model.setParam("IntFeasTol", 1e-09)
model.setParam("IntegralityFocus",1)
model.Params.IntegralityFocus = 1
model.printQuality()
#model.Params.IntFeasTol = 1e-09

for v in model.getVars():
    #if v.x>0.01:
        print('%s %s %g' % (v.varName, '=', v.x))
"""



model.optimize()
if model.status == GRB.OPTIMAL:  # check if the solver is capable of finding an optimal solution
    model.printAttr('X')
    print(model.status, 'optimal')
    print('Obj: %g' % model.objVal)
else:
    print(model.status, 'not optimal')

# print results
for v in model.getVars():
    if v.x > 0:
        print('%s %g' % (v.varName, v.x))