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
import measurement.measures as conv
# class imports
from VehicleSpecifications import ElectricVehicle

# ****************************************************
# * General Framework Initialisation
# ****************************************************
# Administrative
rd.seed(1932455)

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

start = conv.Time(hr=cfg.getfloat('general','starting')).min 
duration = conv.Time(hr=cfg.getint('general','duration')).min
resolution = cfg.getint('general','resolution')

# calculate further parameters from config
num_slots = int(duration/resolution)
dayswitch_slot = int((duration-start)/resolution) #first slot belonging to new day
start_slot = int(start/resolution)

# non-changeable final parameters
num_households = 55

# ****************************************************
# * Generate Scenario
# ****************************************************

# assign residential load forecast in network

# assign PV generation forecast in network


# assign EV behaviour in network
num_evs = round(par_evpenetration * num_households)
evs = [ElectricVehicle(cfg,sps.randint(1,num_households)) for i in range(1,num_evs)]
print(str(num_evs)+"/" +str(num_households)+" possible vehicles initialised and located.")

# generate EV availability and battery state of charge forecast
for ev in evs:
    ev.generateAvailabilityForecast()
    ev.generateBatterySOCForecast()
print("Vehicle availability and battery SOC forecasts generated.")

# generate electricity price forecast
day_id = rd.randint(0,999)

price_ts1 = read_timeseries('../price_timeseries/15min/priceprofile_ukpx_15min'+format(day_id,'04d')+'.txt')
price_ts2 = read_timeseries('../price_timeseries/15min/priceprofile_ukpx_15min'+format(day_id+1,'04d')+'.txt')
price_ts = []

for i in range(num_slots):
    if i<dayswitch_slot:
        price_ts.append(price_ts1[start_slot+i])
    else:
        price_ts.append(price_ts2[i-dayswitch_slot])
    
mean_price = mean(price_ts)
price_ts = [((item - mean_price) * par_spread + mean_price + par_surcharge) for item in price_ts]
print("Electricity market prices forecast generated.")

# ****************************************************
# * Run Optimisation
# ****************************************************



# ****************************************************
# * Run Simulation
# ****************************************************

# Compile and solve circuit
DSSText.Command = r"Compile '..\network_details\Master.dss'"
DSSText.Command = "set mode=yearly number=1440 stepsize=1m"
#DSSText.Command = "solve"

# ****************************************************
# * Functions
# ****************************************************
