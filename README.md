# EB-MDVSPTW INSTANCES DESCRIPTION

-----------------
# DX_SX_CX_X_trips.txt files

HEADER

0 column: number of vehicles	phi_max	phi_min	travel cost	charging rate per minute	energy consumption per km

1 column: number of trips

2 column: number of charging events

3 column: lambda value indicating the unit cost of passenger waiting

4 column: SOC of vehiclek when it is fully charged

5 column: minimum allowed SOC level 

6 column: travel cost per km

7 column: charging rate per minute

8 column: energy consumption per km

BODY

0 column: task node identification number

1 column: latitude of the origin of the task node

2 column: longitude of the origin of the task node

3 column: latitude of the destination of the task node

4 column: longitude of the destination of the task node

5 column: earliest possible starting time of service at the task node

6 column: latest possible starting time of service at the task node

-----------------

# DX_SX_CX_charging_event_sequence.txt files

HEADER

Provide a list of the identification numbers of the last charging events at all chargers

BODY

0 column: preceding charging event

1 column: following charging event

