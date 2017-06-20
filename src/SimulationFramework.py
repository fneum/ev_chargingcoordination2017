# standard imports
import win32com.client
import configparser
import numpy.random as rd
import numpy as np
import scipy as sp
import scipy.stats as sps
import csv
import fileinput
from statistics import mean
# class imports
from VehicleSpecifications import ElectricVehicle

# ****************************************************
# * General Framework Initialisation
# ****************************************************
# Administrative
rd.seed(191994)

# some utility functions
def read_timeseries(filename):
    with open(filename) as file:
        data = [float(line) for line in file]
    return data

# ****************************************************
# * Initialize OpenDSS
# ****************************************************

# Instantiate the OpenDSS Object
try:
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
    print("Unable to start the OpenDSS Engine")
    raise SystemExit

# Set up the Text, Circuit, and Solution Interfaces to manage OpenDSS
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# ****************************************************
# * Read Parameters
# ****************************************************
cfg = configparser.ConfigParser()
cfg.read('../parameters/evalParams.ini')

# assign parameter names
par_surcharge = cfg.getfloat('market_prices','surcharge')
par_spread = cfg.getfloat('market_prices','spread')
par_evpenetration = cfg.getfloat('electric_vehicles','penetration')

#print (cfg.getboolean('uncertainty', 'unc_ev',fallback='Request not in parameter set!'))

# ****************************************************
# * Generate Scenario
# ****************************************************

# assign residential load forecast in network
num_households = 55


# assign PV generation forecast in network

# assign EV behaviour in network
num_evs = round(par_evpenetration * num_households)
evs = [ElectricVehicle(cfg,sps.randint(1,num_households)) for i in range(1,num_evs)]

# generate EV availability forecast



# generate EV energy demand forecast


# generate electricity price forecast
day_id = rd.randint(0,1000)
price_ts = read_timeseries('../price_timeseries/15min/priceprofile_ukpx_15min'+format(day_id,'04d')+'.txt')
mean_price = mean(price_ts)
price_ts = [((item - mean_price) * par_spread + mean_price + par_surcharge) for item in price_ts]

# ****************************************************
# * Run Optimisation
# ****************************************************


# ****************************************************
# * Run Simulation
# ****************************************************

# Compile and solve circuit
DSSText.Command = r"Compile 'C:\Users\Fabian Neumann\OneDrive\Studium\UOE\Dissertation\Code\ev_chargingcoordination2017\network_details\Master.dss'"
DSSText.Command = "set mode=yearly number=1440 stepsize=1m"
#DSSText.Command = "solve"

# ****************************************************
# * Functions
# ****************************************************
