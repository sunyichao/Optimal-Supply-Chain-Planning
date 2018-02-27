# Optimal-Supply-Chain-Planning
CPLEX and Gurobi based LP and MIP models for network planning in supply chain

NOTE: *This repository is solely for demonstration purpose. Data has not been provided due to copyright restrictions.*

## Problem Statement
A company has 5 products and 4 plants. Each product is produced at only 1 plant except for product-4,5 which are produced at plant-4. The company serves 50 customers accross US, each having certain demand of each product.

## Scenario 1
In order to improve service, warehouses are to be built close to customer locations such that 80% of the supply from a warehouse is delivered within 500 miles of radius. The optimization model determines the minimum number of warehouses and their locations required to satisfy the constraints and at the same time minimize transportation cost.

## Scenario 2
Instead of building warehouses, investments are to be made at each plant so that each plant can produce all kinds of products. In addition to upgradation cost, setup from one product to another requires time. Setup times are different for different products. Setup times also affect the production capacity, though overtime hours are available but with higher production cost. The optimization model determines optimal sequence of production at each plant and optimal distribution of production of each product at each plant to minimize transportation cost, production cost and overtime cost. 
