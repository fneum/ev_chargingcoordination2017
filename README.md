# Optimal Scheduling of Electric Vehicle Charging in Distribution Networks

## Project Abstract
The transport sector accounts for a significant proportion of total energy consumption and is to date largely based on fossil fuels. Mitigation of greenhouse gas emissions via the large-scale electrification of road transport will likely deteriorate voltage profiles and overload network equipment in distribution networks. Controlling the charging schedule of electric vehicles in a centralised and coordinated manner provides a potential solution to mitigate the issues and could defer the investment on upgrading the network infrastructures.

In this work, a robust cost-minimising unidirectional day-ahead scheduling routine for charging electric vehicles overnight in residential low voltage distribution networks is presented that observes local network, equipment and charging demand constraints in a stochastic environment. To reduce the computational complexity, a linear power flow approximation is utilised. The modelled environment involves uncertain residential electricity demand, market prices, and the mobility behaviour of electric vehicle owners including stochastic daily trip distances, arrival and departure times. Knowledge about the probability distributions of these parameters is used to hedge risks regarding the cost of charging, network overloadings, voltage violation and charging reliability.

The results provide an insight into the impact of uncertainty and the effectiveness of addressing particular aspects of risk during optimisation. Particularly, consideration of temporally variable household-level demand peaks and planning with more conservative estimates of initial battery charge levels increased the reliability and technical feasibility of optimised schedules. It is further outlined that the introduction of dynamic grid levies, which amplify the effect of variable electricity prices, constitutes a key determinant of cost saving potential by demand side management that could incur only minor fiscal implications.

## Installation
Run `setup.py` to install all required packages. Alternatively `requirements.txt` provides a list of required packages.

## Running
To run, first, specify parameters in `parameters/evalParams.ini`, then run `/src/SimulationFramework.py`.

## Report
Dissertation and analyses are complementet in `/docs/`

## Results
Results are stored in `/log/` in a folder denoted by a unique date singleton. Results used for this thesis are available on request due to large file sizes.