''' DATA ANALYSIS
Uncomment required function (at the end) to call
'''

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#======================================================
# Reading worksheets form xlsx file
#======================================================
workbook=pd.ExcelFile( "Network Planning Case Study - Opex Analytics.xlsx")
df_plants=pd.read_excel(workbook,'Plants')
df_customers=pd.read_excel(workbook,'Customers')
df_product=pd.read_excel(workbook,'Product')
df_annual_demand=pd.read_excel(workbook,'Annual Demand')
df_distances=pd.read_excel(workbook,'Distances')

#====================================================================
# Function to calculate demand of each customer by product
#====================================================================
def ann_product_demand(annual_demand,year):
    ann_dem=copy.copy(annual_demand)
    ann_dem=ann_dem.loc[ann_dem['Time Period']==year]
    prod_demand=np.array([])
    flag=0
    for i in [1,2,3,4,5]:       #Creating a numpy array of demand of products (rows)- customers (columns)
        if flag==1:
            temp=ann_dem.loc[ann_dem['Product ID']==i]
            prod_demand=np.vstack((prod_demand,temp['Demand (in tonnes)']))
        if flag==0:
            temp=ann_dem.loc[ann_dem['Product ID']==i]
            prod_demand=np.hstack((prod_demand,temp['Demand (in tonnes)']))
            flag=1
    #print(prod_demand.shape)

    #Creating a pie chart of different product demands by all customers
    labels= 'Product 1','Product 2','Product 3','Product 4','Product 5'
    sizes=prod_demand.sum(axis=1)
    colors=['yellowgreen','lightcoral','lightskyblue','gold','red']
    plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
    plt.axis('equal')
   
    plt.show()

#======================================================
# Calculating current network performance metrics
#======================================================
def curr_metrics():
    df=pd.read_csv('df.csv')
    print('CURRENT NETWORK PERFORMANCE METRICS')
    print('Outbound Freight Cost per ton shipped=',
          df['Transportation Cost'].sum()/df['Demand (in tonnes)'].sum())
    print('Average Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].mean()))
    print('Minimum Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].min()))
    print('Maximum Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].max()))
    sum_=0
    net_demand=df['Demand (in tonnes)'].sum()
    for index,rows in df.copy().iterrows():
        if rows['Distance']>500:
            sum_=sum_+rows['Demand (in tonnes)']

    print('Percentage supply transported over 500 miles='+str(100*sum_/net_demand))
    print('Average distance travelled by a truck in a delivery='+str(df['Distance'].mean()))
    print('Maximum distance travelled in a delivery='+str(df['Distance'].max()))
    print('Total Annual Revenue='+str(df['Total Revenue'].sum()))
    print('Total Transport Cost='+str(df['Transportation Cost'].sum()))
    print('Total Production Cost='+str(df['Production Cost'].sum()))
    print('Percentage Net Transport Cost to Net Revenue='+str((df['Transportation Cost'].sum()/df['Total Revenue'].sum())*100))
    print('Percentage Net Production Cost to Net Revenue='+str((df['Production Cost'].sum()/df['Total Revenue'].sum())*100))


    #Histogram of distance travelled in delivery
    '''
    n,bins,patches=plt.hist(df['Distance'],10,facecolor='grey',alpha=0.5)
    plt.ylabel('Number of deliveries')
    plt.xlabel('Distance travelled')
    plt.title('Histogram of Distance Travelled in Deliveries')
    plt.show()
    '''
    
#===============================================================================
# Creating a histogram of distances between warehouse and customer in scenario 1
#===============================================================================
def s1_metrics():
    df=pd.read_csv('s1_assignment1.csv')
    print('SCENARIO 1 PERFORMANCE METRICS')
    print('Average Distance Travelled between Warehouse and Customer'+str(df['Warehouse-Customer Distance'].mean()))
    print('Maximum Distance between a warehouse and Customer'+str(df['Warehouse-Customer Distance'].max()))

    n,bins,patches=plt.hist(df['Warehouse-Customer Distance'],10,facecolor='grey',alpha=0.5)    #Creating a histogram of Warehouse-Customer Distance
    plt.ylabel('Number of customers')
    plt.xlabel('Warehouse-Customer Distance')
    plt.title('Histogram of Warehouse-Customer Distance')
    plt.show()

#=============================================
# calculating metrics from scenario 2 results
#=============================================
def s2_metrics():
    print('SCENARIO 2 PERFORMANCE METRICS')
    df=pd.read_csv('df_scenario2.csv')
    sum_=0
    net_demand=df['Demand (in tonnes)'].sum()
    for index,rows in df.copy().iterrows():
        if rows['Distance']>500:
            sum_=sum_+rows['Demand (in tonnes)']
    print('Percentage supply transported over 500 miles='+str(100*sum_/net_demand))
    print('Average distance travelled by a truck in a delivery='+
          str(df['Distance'].mean()))
    print('Maximum distance travelled in a delivery='+str(df['Distance'].max()))
    print('Total Annual Revenue='+str(df['Total Revenue'].sum()))
    print('Total Transport Cost='+str(df['Transportation Cost'].sum()))
    print('Outbound Freight Cost per ton shipped=',
          df['Transportation Cost'].sum()/df['Demand (in tonnes)'].sum())
    print('Average Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].mean()))
    print('Minimum Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].min()))
    print('Maximum Percentage Transport Cost to Revenue='+str(df['Percentage Transportation Cost to Revenue'].max()))

    #Histogram of distance travelled in delivery
    '''
    n,bins,patches=plt.hist(df['Distance'],10,facecolor='grey',alpha=0.5)
    plt.ylabel('Number of deliveries')
    plt.xlabel('Distance travelled')
    plt.title('Histogram of Distance Travelled in Deliveries')
    plt.show()
    '''
    
    s2_schedule=pd.read_csv('s2_schedule1.csv')
    print('Total Production Cost='+str(s2_schedule['Total Production Cost'].sum()))
    print('Average Percentage Capacity Utilization='+str(s2_schedule['Percentage Capacity Utilization'].mean()))
    


''' Calculate scenario 2 network metrics '''
#s2_metrics()

''' Calculate scenario 1 network metrics '''
#s1_metrics()

''' calculate current network metrics '''
#curr_metrics()

''' Create pie chart of demand '''
#ann_product_demand(df_annual_demand,2014)
