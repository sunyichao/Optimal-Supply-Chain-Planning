''' SCENARIO 2: SWITCHING PRODUCTION '''

import pandas as pd
import numpy as np
import copy
import cplex
from cplex.exceptions import CplexError
import math
from gurobipy import *

#======================================================
# Reading worksheets form xlsx file
#======================================================
workbook=pd.ExcelFile( "Network Planning Case Study.xlsx")
df_plants=pd.read_excel(workbook,'Plants')
df_customers=pd.read_excel(workbook,'Customers')
df_product=pd.read_excel(workbook,'Product')
df_annual_demand=pd.read_excel(workbook,'Annual Demand')
df_distances=pd.read_excel(workbook,'Distances')
df_production_capacity=pd.read_excel(workbook,'Production Capacity')
df_setups=pd.read_excel(workbook,'Setups')



#=======================================================================================
# Function to find optimal sequence of changeover of products
# Arguments required- df_setups (DataFrame containing setup times
#                     List of products for which optimal production sequence is required 
#=======================================================================================
def opti_seq(setups,products):
    setup=copy.copy(setups)
    setup=setup.as_matrix()
    setup=setup[1:][:]      # Removed column names from matrix
    products.sort()
    prod=[p-1 for p in products]
    # Extracting setup times for products required
    temp=np.array([])
    flag=0
    for i in prod:
        if flag==1:
            temp=np.vstack((temp,np.take(setup[i],prod)))
        if flag==0:
            temp=np.hstack((temp,np.take(setup[i],prod)))
            flag=1
    setup=temp
    #print(setup)

    #Creating decision variables
    var=[]
    for i in products:
        for j in products:
            var.append('x_'+str(i)+'_'+str(j))

    #Creating list of objective coefficients
    obj_coeff=[i for j in setup.tolist() for i in j]
    for n,i in enumerate(obj_coeff):
        if i==0.0:
            obj_coeff[n]=cplex.infinity     # Making self loop cost infinity

    #Providing upper and lower bounds on decision variables
    upper_bound=[1 for x in np.arange(len(var))]
    lower_bound=[0 for x in np.arange(len(var))]
    ctype="I"*len(var)  #Setting variable type

    #Creating b vector of constraints Ax=b
    rhs_vector=[1 for x in np.arange(int(setup.shape[0])*2)]
    #Constraints reference names
    rownames=["p"+str(i) for i in np.arange(int(setup.shape[0]))]+["q"+str(i) for i in np.arange(int(setup.shape[0]))]
    #Constraints sense
    sense="E"*(int(setup.shape[0])*2)

    try:
        model=cplex.Cplex()     #Creating cplex model
        model.objective.set_sense(model.objective.sense.minimize)
        model.variables.add(obj=obj_coeff,lb=lower_bound,ub=upper_bound,
                            types=ctype,names=var)
        rows=[]
        #Adding constraints by rows
        for i in np.arange(int(setup.shape[0])):
            const_var=[]
            for j in np.arange(int(setup.shape[0])):
                const_var.append(var[j*int(setup.shape[0])+i])
            const_coeff=[1 for x in np.arange(int(setup.shape[0]))]
            rows.append([const_var,const_coeff])
        for i in np.arange(int(setup.shape[0])):
            const_var=var[i*int(setup.shape[0]):i*int(setup.shape[0])+int(setup.shape[0])]
            const_coeff=[1 for x in np.arange(int(setup.shape[0]))]
            rows.append([const_var,const_coeff])
        model.linear_constraints.add(lin_expr=rows,senses=sense,
                                     rhs=rhs_vector, names=rownames)
        model.solve()
    except CplexError as exc:
        print(exc)
        return
    
    print("Solution Status= ", model.solution.get_status())
    print(model.solution.status[model.solution.get_status()])
    print("Solution value= ",model.solution.get_objective_value())
    numcols = model.variables.get_num()
    numrows = model.linear_constraints.get_num()
    x = model.solution.get_values()

    # Displaying decision variables that are equal to 1
    lis=[]
    for j in range(numcols):
        if x[j]==1:
            lis.append(var[j])
            print(var[j],x[j])
    # From obtained solution, creating a list of sequence and total setup time
    count=0
    lis1=[]
    for items in lis:
        count+=1
        i=int(items[2])
        j=int(items[4])
        if count<=1:
            lis1.append(i)
            lis1.append(j)
        else:
            if (i not in lis1) and (j not in lis1):
               lis1.append(i)
               lis1.append(j)
            elif j in lis1:
                lis1.insert(lis1.index(j),i)
            elif i in lis1:
                lis1.insert(lis1.index(i)+1,j)
    #setup_days=model.solution.get_objective_value()-setup[products.index(lis1[-2])][products.index(lis1[-1])]
    setup_days=model.solution.get_objective_value()
    return (lis1,setup_days)    #Returning a tuple of list containing optimal sequence and corresponding setup days


#===================================================================
# Function to find quaterly demand of each product for each customer
#===================================================================
def ann_product_demand(annual_demand,year):     #Arguments- Dataframe of annual demand, year
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
    prod_demand=np.transpose(prod_demand)
    return prod_demand  #Returning numpy 2D array of demand as customer(rows)-product(columns)

#====================================================
# Function to find distance matrix for plant-customer
# ===================================================
def create_dist(distance):      #Argument- Dataframe of distances
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

#=========================================================
# Function to find production cost matrix as plant-product
#=========================================================
def prod_cost(prod_cap):        #Argument- Dataframe of production capacity
    prod_cap_c=copy.copy(prod_cap)
    prod_cost=np.array([])
    flag=0
    for item in np.split(prod_cap_c['Production Cost'],5):
        if flag==1:
            prod_cost=np.vstack((prod_cost,np.array(list(item))))
        if flag==0:
            prod_cost=np.hstack((prod_cost,np.array(list(item))))
            flag=1
    prod_cost=np.transpose(prod_cost)
    return prod_cost    #Returning numpy 2D array of production cost between plant(rows)-product(columns)


#====================================================================
# Function to find annual production capacity matrix as plant-product
#====================================================================
def prod_capacity(prod_cap):        #Argument- Dataframe of production capacity
    prod_cap_c=copy.copy(prod_cap)
    prod_capacity=np.array([])
    flag=0
    for item in np.split(prod_cap_c['Annual Production Capacity'],5):
        if flag==1:
            prod_capacity=np.vstack((prod_capacity,np.array(list(item))))
        if flag==0:
            prod_capacity=np.hstack((prod_capacity,np.array(list(item))))
            flag=1
    prod_capacity=np.transpose(prod_capacity)
    return prod_capacity    #Returning numpy 2D arrat of production capacity between plant(rows)-product(columns)

#===========================================================================
# Function to find optimal scheduling and allocation of demand at all plants
# Arguments - 1. Dataframe of distances
#             2. Dataframe of production capacity
#             3. Dataframe of annual demand
#===========================================================================
def optimize_cost(df_dist,df_prod_cap,df_ann_dem):
    # Creating copies of arguments
    ann_dem_c=copy.copy(df_ann_dem)
    dist_c=copy.copy(df_dist)
    prod_cap_c=copy.copy(df_prod_cap)
    rate=[100,50,50,50]     #Production rate of 4 plants

    #Getting required data as numpy 2D array
    prod_dem=ann_product_demand(ann_dem_c,2014)
    dist_mat=create_dist(dist_c)
    prod_cos=prod_cost(prod_cap_c)
    prod_cap=prod_capacity(prod_cap_c)
    
    try:
        
        #Creating a Gurobi Model
        model=Model("Scheduling_&_Assignment")
        
        #Creating dictionaries whose keys will be reference of decision variables and values will be corresponding obj coeff
        assignment={}
        for i in np.arange(1,5):
            for j in np.arange(1,51):
                for p in np.arange(1,6):
                    assignment[(i,j,p)]=prod_dem[j-1][p-1]*0.25*dist_mat[i-1][j-1]*0.2
        days={}
        for i in np.arange(1,5):
            for p in np.arange(1,6):
                days[(i,p)]=prod_cos[i-1][p-1]*8*rate[i-1]
        overtime={}
        for i in np.arange(1,5):
            for p in np.arange(1,6):
                overtime[(i,p)]=1.5*prod_cos[i-1][p-1]*rate[i-1]
        demand={}
        for i in np.arange(1,5):
            for p in np.arange(1,6):
                demand[(i,p)]=0

        #Adding decision variables with names Z,X,x,y
        Z=model.addVars(assignment.keys(),lb=0,ub=1,obj=assignment,vtype=GRB.BINARY,name="Z")
        X=model.addVars(days.keys(),lb=0,ub=90,obj=days,vtype=GRB.INTEGER,name="X")
        x=model.addVars(overtime.keys(),lb=0,ub=360,obj=overtime,vtype=GRB.INTEGER,name="x'")
        y=model.addVars(demand.keys(),lb=0,ub=GRB.INFINITY,obj=demand,vtype=GRB.CONTINUOUS,name="y_")

        model.ModelSense= GRB.MINIMIZE

        #Adding constraints
        model.addConstrs((X.sum(i,'*')<=90-31 for i in [1,2,3,4]),name="time_const_")       # Vary RHS value of this constraint for different set of products and total setup time
        model.addConstrs((8*rate[i-1]*X[i,p]+rate[i-1]*x[i,p]<=prod_cap[i-1][p-1]*0.25 for i in [1,2,3,4] for p in [1,2,3,4,5]),name="prod_cap_")
        model.addConstrs((x[i,p]-8*X[i,p]<=0 for i in [1,2,3,4] for p in [1,2,3,4,5]),name="Overtime_")
        model.addConstrs((y[i,p]==quicksum(prod_dem[j-1][p-1]*0.25*Z[i,j,p] for j in np.arange(1,51)) for i in [1,2,3,4] for p in [1,2,3,4,5]),name="demand")
        model.addConstrs((8*rate[i-1]*X[i,p]+rate[i-1]*x[i,p]-y[i,p]>=0 for i in [1,2,3,4] for p in [1,2,3,4,5]),name="demand_")
        model.addConstrs((Z.sum('*',j,p)==1 for j in np.arange(1,51) for p in [1,2,3,4,5]),name="service_")
        model.addConstrs((x.sum(i,'*')<=360 for i in [1,2,3,4]))
        
        model.optimize()
        #model.computeIIS()
        print('Obj: %g' % int(model.objVal))
        for v in model.getVars():
            if v.x>0:
                print('%s %g' % (v.varName, v.x))


        # Exporting results
        
        if model.status==GRB.Status.OPTIMAL:
            print('\nX:')
            
            X_value=model.getAttr('x',X)
            for i in [1,2,3,4]:
                for p in [1,2,3,4,5]:
                    print((i,p),X_value[i,p])
            print("\nx':")
            
            x_value=model.getAttr('x',x)
            for i in [1,2,3,4]:
                for p in [1,2,3,4,5]:
                    print((i,p),x_value[i,p])
            Z_value=model.getAttr('x',Z)

            # Creating DataFrame with index (Plant ID and Product ID), and a column of every customer.
            # Entry 1 shows that the customer is being served product ID=index[1] by plant ID=index[0] 
            cols=['Plant ID','Product ID']
            for i in np.arange(1,51):
                cols.append('Customer.'+str(i))
            arr=np.zeros((20,len(cols)),dtype=np.int)
            df_assignment=pd.DataFrame(arr,columns=cols)        
            df_assignment['Plant ID']=[x for i in [1,2,3,4] for x in [i]*5]
            df_assignment['Product ID']=[1,2,3,4,5]*4
            df_assignment=df_assignment.set_index(['Plant ID','Product ID'])
            
            #Filling entries in DataFrame using values of Z decision variables
            for i in np.arange(1,5):
                for p in np.arange(1,6):
                    for j in np.arange(1,51):
                        df_assignment.at[(i,p),'Customer.'+str(j)]=Z_value[i,j,p]
            df_assignment.to_csv('s2_assignment.csv')       #Saving DataFrame as csv file

            
            #Creating another DataFrame with index Plant ID and Product ID containing information of number of days of production and overtime hours required in a quarter
            cols=['Plant ID','Product ID','Number of Days of Production','Overtime Hours']
            arr=np.zeros((20,len(cols)),dtype=np.int)
            df_schedule=pd.DataFrame(arr,columns=cols)
            df_schedule['Plant ID']=[x for i in [1,2,3,4] for x in [i]*5]
            df_schedule['Product ID']=[1,2,3,4,5]*4
            df_schedule=df_schedule.set_index(['Plant ID','Product ID'])
            
            #Filling entries in DataFrame using values of X,x decision variables
            flag4=1
            for i in [1,2,3,4]:
                for p in [1,2,3,4,5]:
                    df_schedule.at[(i,p),'Number of Days of Production']=X_value[i,p]
                    df_schedule.at[(i,p),'Overtime Hours']=x_value[i,p]
            df_schedule.to_csv('s2_schedule.csv')       #Saving dataframe as csv file.

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError as a:
        print('Encountered an attribute error '+str(a))

    return model

#==================================================
# Function to find the total cost before scenario 2
#==================================================
def prev_cost(ann_dem,dist,year):
    df_dist=copy.copy(dist)
    df_dist=df_dist[['Plant Id','Customer ID','Distance']].dropna()
    df_dist[['Plant Id','Customer ID']]=df_dist[['Plant Id','Customer ID']].astype(int)
    df_ad=copy.copy(ann_dem)
    df_ad=df_ad.loc[df_ad['Time Period']==year]

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

    #Creating a dataframe containing information of current network
    df=pd.merge(df_ad,df_dist,on=['Customer ID','Plant Id'])

    # Calculating annual transportation cost of product to correspoding customer
    df['Transportation Cost']=df['Demand (in tonnes)']* df['Distance'] * 2 / 10
    # Calculating total annual revenue generated through a customer for a product
    df['Total Revenue']=df['Demand (in tonnes)']* df['Revenue ($)']
    # Calculating percentage transportation cost of total revenue for a product,customer pair
    df['Percentage Transportation Cost to Revenue']=df['Transportation Cost']*100/df['Total Revenue']
    

    prod_cost=[500,400,300,200,100]     #Product-wise cost starting from product 1 (same for all plants)
    df['Production Cost']=0
    df=df.set_index(['Customer ID','Product ID'])

    # Calculating production cost
    for p in np.arange(1,6):
        for c in np.arange(1,51):
            df.at[(c,p),'Production Cost']=df.at[(c,p),'Demand (in tonnes)']*prod_cost[p-1]
    df=df.reset_index()
    df['Percentage Production Cost to Revenue']=df['Production Cost']*100/df['Total Revenue']
    cost=df['Transportation Cost'].sum()*100/df['Total Revenue'].sum()
    print('Transportation cost of product 1: ', df.loc[df['Product ID']==1,'Transportation Cost'].sum())
    print('Transportation cost of product 2: ', df.loc[df['Product ID']==2,'Transportation Cost'].sum())
    print('Transportation cost of product 3: ', df.loc[df['Product ID']==3,'Transportation Cost'].sum())
    print('Transportation cost of product 4: ', df.loc[df['Product ID']==4,'Transportation Cost'].sum())
    print('Transportation cost of product 5: ', df.loc[df['Product ID']==5,'Transportation Cost'].sum())
    tc=df['Transportation Cost'].sum()
    pc=df['Production Cost'].sum()
    print('Total Transportation cost in year '+str(year)+'= '+str(tc))
    print('Total Production cost in year '+str(year)+'= '+str(pc))
    print('Total Cost in year '+str(year)+'= '+str(tc+pc))
    df=df.set_index(['Customer ID','Product ID'])
    
    # convert to csv file
    df.to_csv('df.csv')
    return df

    '''
    OUTPUT:
    ('Transportation cost of product 1: ', 72756687.09486438)
    ('Transportation cost of product 2: ', 15504520.239111)
    ('Transportation cost of product 3: ', 6197455.6810546)
    ('Transportation cost of product 4: ', 2895037.9993204)
    ('Transportation cost of product 5: ', 1352499.7484332)
    Total Transportation cost in year 2014= 98706200.7627836
    Total Production cost in year 2014= 188694005
    Total Cost in year 2014= 287400205.7627836
    '''

#=============================================
# Function to find total cost after scenario 2
#=============================================
def final_cost(dist,year):
    #Reading dataframe of information of current network
    df=pd.read_csv('df.csv')
    # Keeping basic columns and dropping other columns.
    df=df.drop(columns=['Production Cost','Transportation Cost','Percentage Transportation Cost to Revenue',
                'Percentage Production Cost to Revenue'])
    df=df.set_index(['Customer ID','Product ID'])

    #Reading csv file containing assignment of customer-product to plants generated by optimizing model.
    df_assign=pd.read_csv('s2_assignment.csv')
    #print(df_assign.head())
    for index in np.arange(20):
        for c in np.arange(1,51):
            if df_assign.at[index,'Customer.'+str(c)]==1:
                df.at[(c,df_assign.at[index,'Product ID']),'Plant Id']=df_assign.at[index,'Plant ID']       #Updating plant ID serving cust-prod pair resulted after optimizing model.

    #Updating distances in dataframe for new pair of (customer-product) and serving plant
    dist_mat=create_dist(copy.copy(dist))
    
    for p in [1,2,3,4]:
        lis=df.index[df['Plant Id']==p].tolist()
        for item in lis:
            c=int(item[0])
            df.at[item,'Distance']=dist_mat[p-1][c-1]

    df['Transportation Cost']=df['Demand (in tonnes)']* df['Distance'] *0.2
    df['Percentage Transportation Cost to Revenue']=df['Transportation Cost']*100/df['Total Revenue']
    
    df=df.reset_index()
    print('Transportation cost of product 1: ', df.loc[df['Product ID']==1,'Transportation Cost'].sum())
    print('Transportation cost of product 2: ', df.loc[df['Product ID']==2,'Transportation Cost'].sum())
    print('Transportation cost of product 3: ', df.loc[df['Product ID']==3,'Transportation Cost'].sum())
    print('Transportation cost of product 4: ', df.loc[df['Product ID']==4,'Transportation Cost'].sum())
    print('Transportation cost of product 5: ', df.loc[df['Product ID']==5,'Transportation Cost'].sum())
    tc=df['Transportation Cost'].sum()
    
    print('Total Transportation cost in year '+str(year)+'= '+str(tc))
    
    df=df.set_index(['Customer ID','Product ID'])
    df.to_csv('df_scenario2.csv')
    
    return df

    '''
    OUTPUT
    ('Transportation cost of product 1: ', 46611439.642578006)
    ('Transportation cost of product 2: ', 9294131.3390762)
    ('Transportation cost of product 3: ', 3240562.985777401)
    ('Transportation cost of product 4: ', 1497575.7535830003)
    ('Transportation cost of product 5: ', 939652.5086943998)
    Total Transportation cost in year 2014= 61583362.229709
    '''


#======================================================================================================================================
# Function to check solution of scenario 2 by comparing estimated annual production with annual demand of each product by each customer
#======================================================================================================================================
def sol_sce2():
    df_schedule=pd.read_csv('s2_schedule.csv')

    #Adjusting overtime hours for plant C
    #After optimizing model, plant C produced only 1,2,3,4 products.
    #Re-optimized sequence gives 4 -> 3 -> 1 -> 2 -> 4 requiring 27 Days of setup different from other plants which are producing all 5 products, with 31 days of setup time.
    #Overtime hours of production generated from optimizing model are being adjusted to available 4 days in order to minimize overtime production as much as possible.
    df_schedule.at[12,'Number of Days of Production']=7
    df_schedule.at[13,'Number of Days of Production']=2
    df_schedule.at[12,'Overtime Hours']=5
    df_schedule.at[13,'Overtime Hours']=0
    
    df_schedule['Quaterly Production']=0
    rate=[100,50,50,50]
    # Estimating quarterly production of each product of each plant from number of days of production and overtime hours
    for index,row in df_schedule.iterrows():
        row['Quaterly Production']=(row['Number of Days of Production']*8 + row['Overtime Hours'])*rate[row['Plant ID']-1]
    # Estimating projected annual production from quarterly production
    df_schedule['Projected Annual Production']=df_schedule['Quaterly Production']*4
    df_schedule['Required Annual Demand']=0

    #Reading customer demand data
    df=pd.read_csv('df_scenario2.csv')
    ind=0
    for plant in [1,2,3,4]:
        for product in [1,2,3,4,5]:
            dem=0
            for index,row in df.iterrows():
                if (row['Plant Id']==plant and row['Product ID']==product):
                    dem=dem+row['Demand (in tonnes)']
            df_schedule.at[ind,'Required Annual Demand']=dem
            ind=ind+1
    
    df_schedule=df_schedule.rename(columns={'Number of Days of Production':'Number of Days of Production in a Quarter',
                                            'Overtime Hours':'Overtime Hours in a Quarter'})
    df_schedule.set_index(['Plant ID','Product ID'],inplace=True)
    df_pc=copy.copy(df_production_capacity)
    df_pc.set_index(['Plant ID','Product ID'],inplace=True)
    # Including production capacity and production cost
    df_schedule=df_schedule.join(df_pc,how='outer')
    df_schedule['Overtime Production Cost']=df_schedule['Production Cost']*1.5
    df_schedule['Total Production Cost']=0
    for index,row in df_schedule.copy().iterrows():
        df_schedule.at[index,'Total Production Cost']=4*(row['Production Cost']*row['Number of Days of Production in a Quarter']*8*rate[index[0]-1] + row['Overtime Production Cost']*row['Overtime Hours in a Quarter']*rate[index[0]-1])
    df_schedule['Percentage Capacity Utilization']=(df_schedule['Projected Annual Production']/df_schedule['Annual Production Capacity'])*100
    df_schedule.to_csv('s2_schedule1.csv')
    




            
np.set_printoptions(precision=4,suppress=True)      #Suppress scientific notation




''' Find optimal sequence given products and their setup times '''
print(opti_seq(df_setups,[1,2,3,4,5]))
print(opti_seq(df_setups,[1,2,3,4]))

''' Solve the optimization problem after knowing optimal sequence of production of products '''
opti_model=(optimize_cost(df_distances,df_production_capacity,df_annual_demand))

''' Calculate previous cost and create a csv file '''
df1=prev_cost(df_annual_demand,df_distances,2014)

''' Calculate final cost '''
df2=final_cost(df_distances,2014)

''' Create csv file including production plan of each product at each plant '''
sol_sce2()
