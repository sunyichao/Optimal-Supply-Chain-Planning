''' SCENARIO 1 OPTIMAL WAREHOUSE LOCATION '''

import xlrd
import pandas as pd
import numpy as np
import copy
import cplex
from cplex.exceptions import CplexError
from gurobipy import *
import math
import matplotlib.pyplot as plt
import gmplot
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise
import pprint

#======================================================
# Reading worksheets form xlsx file
#======================================================
# Global Variables
workbook=pd.ExcelFile( "Network Planning Case Study.xlsx")
df_plants=pd.read_excel(workbook,'Plants')
df_customers=pd.read_excel(workbook,'Customers')
df_product=pd.read_excel(workbook,'Product')
df_annual_demand=pd.read_excel(workbook,'Annual Demand')
df_distances=pd.read_excel(workbook,'Distances')



#======================================================
# Calculate transportation cost of a year (2012/13/14)
# Arguments- 1. Dataframe of Annual Demand
#            2. Dataframe of distances
#            3. year
#======================================================
def trans_cost(ann_dem,dist,year):
    df_dist=copy.copy(dist)     
    df_dist=df_dist[['Plant Id','Customer ID','Distance']].dropna()
    df_dist[['Plant Id','Customer ID']]=df_dist[['Plant Id','Customer ID']].astype(int)
    df_ad=copy.copy(ann_dem)
    df_ad=df_ad.loc[df_ad['Time Period']==year]
    
    #Adding plant IDs that are serving correponding products
    plants=[]
    for row in df_ad['Product ID']:
        if row==1:
            plants.append(1)
        elif row==2:
            plants.append(2)
        elif row==3:
            plants.append(3)
        else:
            plants.append(4)
    df_ad['Plant Id']=plants

    #Merging with dataframe of distances to get distances of each customer-plant pair
    df=pd.merge(df_ad,df_dist,on=['Customer ID','Plant Id'])

    # Calculating annual transportation cost of product to correspoding customer
    df['Transportation Cost']=df['Demand (in tonnes)']* df['Distance'] * 2 / 10
    # Calculating total annual revenue generated through a customer for a product
    df['Total Revenue']=df['Demand (in tonnes)']* df['Revenue ($)']
    # Calculating percentage transportation cost of total revenue for a product,customer pair
    df['Percentage cost']=df['Transportation Cost']*100/df['Total Revenue']

    cost=df['Transportation Cost'].sum()*100/df['Total Revenue'].sum()
    print('Transportation cost of product 1: ', df.loc[df['Product ID']==1,'Transportation Cost'].sum())
    print('Transportation cost of product 2: ', df.loc[df['Product ID']==2,'Transportation Cost'].sum())
    print('Transportation cost of product 3: ', df.loc[df['Product ID']==3,'Transportation Cost'].sum())
    print('Transportation cost of product 4: ', df.loc[df['Product ID']==4,'Transportation Cost'].sum())
    print('Transportation cost of product 5: ', df.loc[df['Product ID']==5,'Transportation Cost'].sum())
    print('Total Transportation cost in year '+str(year)+'= '+str(df['Transportation Cost'].sum()))

    # convert to csv file
    #df.to_csv('df.csv')
    return df['Transportation Cost'].sum()



#=====================================================================
# Function to calculate impact on transportation cost after scenario 1
# Arguments- 1. Dataframe of annual demand
#            2. Dataframe of distances
#            3. Year
#=====================================================================
def scenario_one_trans_cost(ann_dem,dist,year):
    df_dist=copy.copy(dist)
    df_ad=copy.copy(ann_dem)

    #Creating a numpy 2D array of distances between customers
    dist_cc=create_adj(df_dist)

    #Creating a numpy 2D array of distances between plant(rows)-customer(column)
    dist_pc=create_dist(df_dist)
    
    #cl=cluster_cust(dist_cc)       # Calling function to create clusters of customers

    #Creating a list of demand of each customer (all products) in the provided year
    dem=tot_annual_demand(df_ad,year)

    #Creating numpy 2D array of demand as product(rows)-customers(columns)
    prod_dem=ann_product_demand(df_ad,year)     


    #Calling function to find min number of warehouses and their locations using LP
    model_linear=location1(dist_cc,dem,dist_pc,prod_dem)
    result(model_linear)
    
    #Calling function to find min number of warehouses and their locations using QP
    model_quad=location(dist_cc,dem,dist_pc,prod_dem)    
    new_cost=model_quad.objVal

    
    #Adding details of warehouses and customers
    #s1_assignment.csv is created in function location
    df=pd.read_csv('s1_assignment.csv')
    df['Customer Name'],df['Customer City'],df['Customer State']=df_customers['Name'],df_customers['City'],df_customers['State']
    df=df.set_index('Warehouse Location ID')
    df.sort_index(inplace=True)
    for index,rows in df.copy().iterrows():
        df.at[index,'Warehouse City']=df_customers.at[index-1,'City']
        df.at[index,'Warehouse State']=df_customers.at[index-1,'State']
    df=df.reset_index()
    for index,rows in df.copy().iterrows():
        df.at[index,'Warehouse-Customer Distance']=dist_cc[rows['Warehouse Location ID']-1][rows['Customer ID']-1]
    df=df.set_index('Warehouse Location ID')
    df=df[['Warehouse City','Warehouse State','Customer ID','Customer Name','Customer City','Customer State','Warehouse-Customer Distance']]
    df.to_csv('s1_assignment1.csv')

    #Creating a csv file containing list and locations of warehouses.
    df_warehouse=df[['Warehouse City','Warehouse State']]
    df_warehouse=df_warehouse.drop_duplicates()

    df_warehouse.to_csv('s1_warehouses.csv')

    
    '''     Used for solution generated by clustering customers and finding warehouse in each cluster.
    lis=[]
    new_cost=[]
    for i in np.arange(len(cl)):
        li,cos=cluster_location(cl[i],dist_cc,dem,dist_pc,prod_dem)
        lis.append(li)
        new_cost.append(cos)'''

    
    prev_cost=trans_cost(df_annual_demand,df_distances,2014)
    Percentage_Change=((new_cost-prev_cost)/prev_cost)*100
    print('Transportation Cost in Scenario 1= '+str(new_cost))
    print('Percentage Change in cost= '+str(Percentage_Change))

    return
    


#===========================================================
# Create adjacency matrix of distances between all customers
#===========================================================
def create_adj(distance):   #Requires dataframe of distances
    df_dist=copy.copy(distance)
    df_dist=df_dist[['Customer ID.1', 'Customer ID.2',    'Distance.1']]
    dist_mat=np.array([])
    flag=0
    for item in np.split(df_dist['Distance.1'],50):
        if (flag==1):
            dist_mat=np.vstack((dist_mat,np.array(list(item))))
        if (flag==0):
            dist_mat=np.hstack((dist_mat,np.array(list(item))))
            flag=1
    return dist_mat #Returning numpy 2D array of distances between customers

#==========================================
# Create distance matrix for plant-customer
# =========================================
def create_dist(distance):      #Requires dataframe of distances
    df_distance=copy.copy(distance)
    df_distance=df_distance[['Plant Id','Customer ID','Distance']].dropna()
    dist_matrix=np.array([])
    flag=0
    for item in np.split(df_distance['Distance'],4):
        if flag==1:
            dist_matrix=np.vstack((dist_matrix,np.array(list(item))))
        if flag==0:
            dist_matrix=np.hstack((dist_matrix,np.array(list(item))))
            flag=1
    return dist_matrix  #Returning numpy 2D array of distances between plant(rows)-customer(column)


#====================================================================
# Function to calculate annual demand (all products) of each customer
#====================================================================
def tot_annual_demand(annual_demand,year):      #Requires dataframe of annual demand, year
    ann_dem=copy.copy(annual_demand)
    ann_dem=ann_dem.loc[ann_dem['Time Period']==year]
    demand=[]
    for i in np.arange(1,51):
        temp=ann_dem.loc[ann_dem['Customer ID']==i]
        demand.append(temp['Demand (in tonnes)'].sum())
    return demand   #Returning a list of demand of each customer in the provided year

#====================================================================
# Function to calculate demand of each customer by product
#====================================================================
def ann_product_demand(annual_demand,year):         #Requires dataframe of annual demand, year
    ann_dem=copy.copy(annual_demand)
    ann_dem=ann_dem.loc[ann_dem['Time Period']==year]
    prod_demand=np.array([])
    flag=0
    for i in [1,2,3,4,5]:
        if flag==1:
            temp=ann_dem.loc[ann_dem['Product ID']==i]
            prod_demand=np.vstack((prod_demand,temp['Demand (in tonnes)']))
        if flag==0:
            temp=ann_dem.loc[ann_dem['Product ID']==i]
            prod_demand=np.hstack((prod_demand,temp['Demand (in tonnes)']))
            flag=1
    #print(prod_demand.shape)
    return prod_demand  #Returning numpy 2D array of demand as product(rows)-customers(columns)

#==============================================================
# Function to generate a bar graph of customer demand
#==============================================================
def bar_graph(demand):
    dem=copy.copy(demand)
    cust=[i for i in np.arange(1,51)]
    plt.bar(np.arange(len(cust)),dem, align='center', alpha=0.5)
    plt.xticks(np.arange(len(cust)),cust)
    plt.ylabel('Demand')
    plt.show()


#=====================================================================================================
# Function to determine optimal warehouse locations among all customer locations using Gurobi using LP
# Arguments- 1. Numpy array of distances between customers
#            2. List of annual demand of each customer (all products)
#            3. Numpy array of distances between plants(rows) and customers(columns)
#            4. Numpy array of demand of products (rows) by each customer (columns)
#=====================================================================================================
def location1(dist_cc,demand,dist_pc,prod_dem):
    dist=copy.copy(dist_cc)
    prod_dem=copy.copy(prod_dem)
    #print(prod_dem)
    net_demand=sum(demand)
    dist1=copy.copy(dist_pc)        # 4x50 array of distance between plants and customers
    dist1=np.vstack((dist1,dist1[3][:]))    #Duplicating last row to create an array of distances between each product's plant (rows) and customers (columns)
    r=copy.copy(dist)
    r[r<=500]=1         
    r[r>500]=0

    try:
        #Creating a Gurobi model
        model=Model("Location")

        #Using dictionary's keys as reference indices for decision variables x
        assignment={}
        for i in np.arange(1,6):
            for j in np.arange(1,51):
                for k in np.arange(1,51):
                    assignment[(i,j,k)]=dist[j-1][k-1]*0.2 + dist1[i-1][j-1]*0.2
        warehouse={}
        for j in np.arange(1,51):
            warehouse[j]=
            
        
        #Adding assignment and warehouse variables
        x=model.addVars(assignment.keys(),lb=0,ub=GRB.INFINITY,obj=assignment,vtype=GRB.CONTINUOUS,name="x")
        z=model.addVars(warehouse.keys(),lb=0,ub=1,obj=warehouse,vtype=GRB.BINARY,name='z')
        

        #Adding Constraints
        model.addConstrs((quicksum(quicksum(x[i,j,k]*r[j-1][k-1] for i in [1,2,3,4,5]) for k in np.arange(1,51))>=0.8*quicksum(quicksum(x[i,j,k] for i in [1,2,3,4,5]) for k in np.arange(1,51)) for j in np.arange(1,51)),name="Demand1")
        model.addConstrs((quicksum(x[i,j,k] for j in np.arange(1,51))>=prod_dem[i-1][k-1] for i in [1,2,3,4,5] for k in np.arange(1,51)),name="demand2")
        model.addConstrs((z[j]>=(quicksum(quicksum(x[i,j,k] for i in [1,2,3,4,5]) for k in np.arange(1,51))/net_demand) for j in np.arange(1,51)),name='facility open')
        model.addConstr((quicksum(z[j] for j in np.arange(1,51))<=4),name='max warehouse')
        
        model.optimize()

        # Displaying non-zero decision variables
        '''
        for v in model.getVars():
            if (v.x>0):
                print('%s %g' % (v.varName, v.x))
        '''
        
                
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError as a:
        print('Encountered an attribute error '+str(a))

    return model

#================================================================
# Output result after LP optimization
# Argument- Gurobi model returned by location1 after optimization
#================================================================
def result(m):
    # Creating a dataframe of result
    cols=['Customer ID','Product ID','Warehouse Location ID','Annual Supply']
    arr=np.zeros((250,3),dtype=np.int)
    for i in [1,2,3,4,5]:
        for k in np.arange(1,51):
            arr[(i-1)*50+k-1,0]=k
            arr[(i-1)*50+k-1,1]=i
    arr1=np.zeros((250,1),dtype=np.float16)
    df_assignment=pd.DataFrame(np.concatenate((arr,arr1),axis=1),columns=cols)
    df_assignment=df_assignment.set_index(['Customer ID','Product ID'])
    df_assignment['Transportation Cost']=0
    
    for i in np.arange(1,6):
        for j in np.arange(1,51):
            for k in np.arange(1,51):
                a=m.getVarByName("x["+str(i)+","+str(j)+","+str(k)+"]")
                if a.X>0.1:
                    df_assignment.at[(k,i),'Warehouse Location ID']=j
                    df_assignment.at[(k,i),'Annual Supply']=a.X
                    df_assignment.at[(k,i),'Transportation Cost']=a.X*a.Obj
    df_assignment=df_assignment.reset_index()
    df_assignment=df_assignment.rename(columns={'Warehouse Location ID':'ID'})
    df_assignment=pd.merge(df_assignment,df_customers[['ID','City','State']],on='ID',how='left')
    df_assignment=df_assignment.rename(columns={'ID':'Warehouse Location ID','City':'Warehouse City','State':'Warehouse State'})
    df_assignment=df_assignment.set_index(['Customer ID','Product ID'])
    df_assignment.to_csv('s1_new_assignment.csv')
    return df_assignment

    
#==========================================================================================================
# Function to determine optimal warehouse locations amnong all locations of customers using Gurobi using QP
# Arguments- 1. Numpy array of distances between customers
#            2. List of annual demand of each customer (all products)
#            3. Numpy array of distances between plants(rows) and customers(columns)
#            4. Numpy array of demand of products (rows) by each customer (columns)
#==========================================================================================================
def location(dist_cc,demand,dist_pc,prod_dem):
    dist=copy.copy(dist_cc)
    dem=copy.copy(demand)
    dist1=copy.copy(dist_pc)        # 4x50 array of distance between plants and customers
    dist1=np.vstack((dist1,dist1[3][:]))    #Duplicating last row to create an array of distances between each product's plant (rows) and customers (columns)
    r=copy.copy(dist)
    r[r<=500]=1         
    r[r>500]=0

    try:
        #Creating a Gurobi model
        model=Model("Location")

        #Using dictionary's keys as reference indices for decision variables x
        assignment={}
        for i in np.arange(1,51):
            for j in np.arange(1,51):
                assignment[(i,j)]=dist[i-1][j-1]*dem[j-1]*0.2

        #Adding assignment variables
        x=model.addVars(assignment.keys(),lb=0,ub=1,vtype=GRB.BINARY,name="x")

        #Adding facility varibales. 1 if open and 0 otherwise
        z=model.addVars(np.arange(1,51),lb=0,ub=1,vtype=GRB.BINARY,name="z")

        #Adding quadratic objective
        obj=QuadExpr()
        for i in np.arange(1,51):
            for j in np.arange(1,51):
                obj+=dist[i-1][j-1]*dem[j-1]*0.2*x[i,j]     #Linear terms
        for i in np.arange(1,51):
            for p in [1,2,3,4,5]:
                for j in np.arange(1,51):
                    obj+=0.2*dist1[p-1][i-1]*prod_dem[p-1][j-1]*x[i,j]*z[i]     #Quadratic terms
        

        model.setObjective(obj,GRB.MINIMIZE)

        #Adding constraints
        model.addConstrs((quicksum(dem[j-1]*r[i-1][j-1]*x[i,j] for j in np.arange(1,51))>= 0.8*quicksum(dem[j-1]*x[i,j] for j in np.arange(1,51)) for i in np.arange(1,51)),name="demand_radius")
        model.addConstrs((x.sum('*',j)==1 for j in np.arange(1,51)),name="1-1")
        model.addConstrs((x.sum(i,'*')<=50*z[i] for i in np.arange(1,51)),name="serve if open")
        model.addConstr((z.sum('*')<=4),name="max warehouses")


        model.optimize()
        print('Obj: %g' % int(model.objVal))
        '''
        for v in model.getVars():
            if v.x>0:
                print('%s %g' % (v.varName, v.x))
        '''

        # Exporting results
        if model.status==GRB.Status.OPTIMAL:
            x_value=model.getAttr('x',x)
            z_value=model.getAttr('x',z)
            
            
            cols=['Customer ID','Warehouse Location ID']
            arr=np.zeros((50,2),dtype=np.int)
            df_assignment=pd.DataFrame(arr,columns=cols)
            df_assignment['Customer ID']=[x for x in np.arange(1,51)]
            for index,rows in df_assignment.copy().iterrows():
                c=rows['Customer ID']
                for w in np.arange(1,51):
                    if x_value[w,c]==1:
                        df_assignment.at[c-1,'Warehouse Location ID']=w
            df_assignment=df_assignment.set_index('Warehouse Location ID')
            df_assignment.to_csv('s1_assignment.csv')
            
            

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError as a:
        print('Encountered an attribute error '+str(a))

    return model




#===========================================================================
# Function to determine optimal warehouse location in a cluster of customers
# Arguments- 1. List containing customer IDs in the cluster
#            2. Numpy array of distances between customers
#            3. List of annual demand of each customer (all products)
#            4. Numpy array of distances between plants(rows) and customers(columns)
#            5. Numpy array of demand of products (rows) by each customer (columns)
#===========================================================================
def cluster_location(cluster,dist_mat,demand,dist_mat1,prod_dem):
    
    dem=[demand[i] for i in [c-1 for c in copy.copy(cluster)]]
    c_=[c-1 for c in copy.copy(cluster)]

    #Creating assignment variables
    var_names=[]
    for i in cluster:
        for j in cluster:
            var_names.append("X_"+str(i)+"-"+str(j))
            
    #Adding facility variables
    var_names=var_names+["Z_"+str(i) for i in cluster]

    #Fetching distance matrix of customers present in the cluster
    dist_matrix=fetch_adj(copy.copy(dist_mat),cluster)
    y=[1 if x<=500 else 0 for x in [i for j in dist_matrix.tolist() for i in j]]
    for i in np.arange(len(cluster)):
        dist_matrix[:,i]*=dem[i]    #Multiplying distances with demand and 0.2 to get transportation cost
        dist_matrix[:,i]*=0.2

    #Adding objective coefficients
    obj_2=[]
    dist_mat1=np.vstack((dist_mat1,dist_mat[3][:]))
    for i in c_:
        sum2=0
        for k in [0,1,2,3,4]:
            sum1=0
            for j in c_:
                sum1+=prod_dem[k][j]

            sum2+=sum1*dist_mat1[k][i]*0.2
        obj_2.append(sum2)
    
    obj_coeff=[i for j in dist_matrix.tolist() for i in j]+obj_2

    #Adding bounds on decision variables
    upper_bound=[1 for x in np.arange(len(cluster)**2)]+[1 for i in [c-1 for c in copy.copy(cluster)]]
    lower_bound=[0 for x in np.arange(len(cluster)**2)]+[0 for i in [c-1 for c in copy.copy(cluster)]]
    ctype="I"*(len(cluster)**2+len(cluster))    #Decision varibale type

    #Vector b of constraints Ax=b 
    rhs_vector=[0.0 for i in np.arange(len(cluster))]+[1 for i in np.arange(len(cluster))]+[0.0 for i in np.arange(len(cluster))]
    rownames=["r"+str(i) for i in cluster]+["p"+str(i) for i in cluster]+["q"+str(i) for i in cluster]
    sense="G"*len(cluster)+"E"*len(cluster)+"L"*len(cluster)
    
    try:
        model=cplex.Cplex()     #Creating a cplex model
        model.objective.set_sense(model.objective.sense.minimize)
        model.variables.add(obj=obj_coeff,lb=lower_bound,ub=upper_bound,
                            types=ctype,names=var_names)
        rows=[]
        #Adding constraints
        for i in np.arange(len(cluster)):
            const_var=var_names[i*len(cluster):i*len(cluster)+len(cluster)]
            coef=[k-0.8 for k in y[i*len(cluster):i*len(cluster)+len(cluster)]]
            const_coef=[a*b for a,b in zip(coef,dem)]
            rows.append([const_var,const_coef])
        for i in np.arange(len(cluster)):
            const_var=[]
            for j in np.arange(len(cluster)):
                const_var.append(var_names[j*len(cluster)+i])
            const_coef=[1 for x in np.arange(len(cluster))]
            rows.append([const_var,const_coef])
        for i in np.arange(len(cluster)):
            const_var=var_names[i*len(cluster):i*len(cluster)+len(cluster)]+[var_names[len(cluster)**2 +i]]
            const_coef=[1 for x in np.arange(len(cluster))]+[-len(cluster)]
            rows.append([const_var,const_coef])
        model.linear_constraints.add(lin_expr=rows,senses=sense,
                                     rhs=rhs_vector,names=rownames)
        model.solve()
        
    except CplexError as exc:
        print(exc)
        return
     
    print("Solution Status= ", model.solution.get_status())
    print(model.solution.status[model.solution.get_status()])
    print("Solution value= ",model.solution.get_objective_value())
    #print(obj_coeff)
    numcols = model.variables.get_num()
    numrows = model.linear_constraints.get_num()

    slack = model.solution.get_linear_slacks()
    x = model.solution.get_values()
    
    '''
    #Display value of slack variables
    for j in range(numrows):
        print("Row %d:  Slack = %10f" % (j, slack[j]))
    '''
    list1=[]
    for j in range(numcols):
        if x[j]==1:
            list1.append(var_names[j])
        print("Value of %s: = %d" % (var_names[j], x[j]))

    if (model.solution.get_status()==101):
        list2=[[int(list1[-1][2:])]]
        list2.append(cluster)
        return list2,model.solution.get_objective_value()   
    else:
        return [],0

'''
#============================================================
# Function to create a scatter plot of locations of customers
#============================================================
def scatter_plot(customers):
    cust=copy.copy(customers)
    cust=cust[['Latitude','Longitude']]
    gmap = gmplot.GoogleMapPlotter(cust['Latitude'][0], cust['Longitude'][0],10)
    #gmap.plot(cust['Latitude'], cust['Longitude'], 'cornflowerblue', edge_width=10)
    gmap.scatter(cust['Latitude'], cust['Longitude'], 'red',size=100,marker=True)
    gmap.draw("mymap.html")
'''

#===========================================================
# Function to create clusters of customers
#===========================================================
def cluster_cust(dist_mat):     #Requires a numpy array of distances between customers
    dist_mat_c=pairwise.pairwise_distances(copy.copy(dist_mat),metric='precomputed')
    clusterer=hdbscan.HDBSCAN(alpha=0.5,min_cluster_size=4,min_samples=3,metric='precomputed')  #Creating an object of hdbscan.HDBSCAN
    clusterer.fit(dist_mat)
    df_customers['Customer cluster']=clusterer.labels_.tolist()
    df_customers['Cluster Probabilities']=clusterer.probabilities_.tolist() 
    clusters=[[],[],[],[],[]]
    for index,row in df_customers.iterrows():
        if row['Customer cluster']==0:
            clusters[0].append(row['ID'])
        elif row['Customer cluster']==1:
            clusters[1].append(row['ID'])
        elif row['Customer cluster']==2:
            clusters[2].append(row['ID'])
        elif row['Customer cluster']==3:
            clusters[3].append(row['ID'])
        elif row['Customer cluster'] in [4,5,-1]:
            clusters[4].append(row['ID'])
    clusters_=[[int(x) for x in clusters[i]] for i in [0,1,2,3,4]]
    #print(len(clusterer.labels_.tolist()))
    #print(clusterer.probabilities_)
    '''for i in [0,1,2,3,4]:
        print(clusters[i])
        print(len(clusters[i]))
        print("\n")'''
    #print(clusters_)
    return clusters_    #Returning list of lists containing customer ID in clusters


#============================================================
# Function to return adjacency matrix of certain customers
#============================================================
def fetch_adj(dist_mat,cust):   #Requires adjacency matrix and list of customers
    dist_mat_c=copy.copy(dist_mat)
    cust=[c-1 for c in copy.copy(cust)]
    fetched_dist=np.array([])
    indices=[]
    flag=0
    for i in cust:
        if flag==1:
            fetched_dist=np.vstack((fetched_dist,np.take(dist_mat_c[i],cust)))
        if flag==0:
            fetched_dist=np.hstack((fetched_dist,np.take(dist_mat_c[i],cust)))
            flag=1

    return fetched_dist





np.set_printoptions(precision=4,suppress=True)      #Suppress scientific notation

scenario_one_trans_cost(df_annual_demand,df_distances,2014)

