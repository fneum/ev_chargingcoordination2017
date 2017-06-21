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

def merge_timeseries(x,y):
    z = []
    for i in range(num_slots):
        if i<dayswitch_slot:
            z.append(x[start_slot+i])
        else:
            z.append(y[i-dayswitch_slot])
    return z
# ****************************************************
# * Read Parameters
# ****************************************************
cfg = configparser.ConfigParser()
cfg.read("../parameters/evalParams.ini")

# assign parameter names
par_surcharge = cfg.getfloat("market_prices","surcharge")
par_spread = cfg.getfloat("market_prices","spread")
par_evpenetration = cfg.getfloat("electric_vehicles","penetration")

start = conv.Time(hr=cfg.getfloat("general","starting")).min 
duration = conv.Time(hr=cfg.getint("general","duration")).min
resolution = cfg.getint("general","resolution")
season = cfg.get("general", "season")

# calculate further parameters from config
num_slots = int(duration/resolution)
dayswitch_slot = int((duration-start)/resolution) #first slot belonging to new day
start_slot = int(start/resolution)

# non-changeable final parameters
num_households = 55

# ****************************************************
# * Initialize OpenDSS
# ****************************************************

# Instantiate the OpenDSS Object
print(">> Opening DSS Engine.")
try:
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
    print("Unable to start the OpenDSS Engine")
    raise SystemExit

# Set up the Text, Circuit, and Solution Interfaces to manage OpenDSS
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

print(">> Compiling distribution network.")
DSSText.Command = r"Compile '..\network_details\Master.dss'"
DSSText.Command = "set mode=daily number="+str(num_slots)+" stepsize="+str(resolution)+"m"

print(">> OpenDSS network instantiated and compiled.")
# ****************************************************
# * Generate Scenario
# ****************************************************

# assign residential load forecast in network

### number of inhabitants according to CREST model
x = rd.random()
num_inhabitants = 0
if x < 0.34:
    num_inhabitants = 1
    id_range = 340
elif x < 0.74:
    num_inhabitants = 2
    id_range = 400
elif x < 0.88:
    num_inhabitants = 3
    id_range = 140
elif x < 0.97:
    num_inhabitants = 4
    id_range = 90
else:
    num_inhabitants = 5
    id_range = 30

day_id = rd.randint(1,id_range)
demand_ts1 = read_timeseries("../demand_timeseries/loadprofile_"+season+"_inh"+\
                             str(num_inhabitants)+"_"+str(resolution)+"min"+format(day_id,"03d")+".txt")
day_id = rd.randint(1,id_range)
demand_ts2 = read_timeseries("../demand_timeseries/loadprofile_"+season+"_inh"+\
                             str(num_inhabitants)+"_"+str(resolution)+"min"+format(day_id,"03d")+".txt")
demand_ts = merge_timeseries(demand_ts1,demand_ts2)

dmd_dss = str(demand_ts).replace(',', '').replace('[', '').replace(']', '')
DSSText.Command = "New Loadshape.Shape_1 npts="+str(num_slots)+" minterval="+str(resolution)+" mult=("+dmd_dss+") useactual=true"
DSSText.Command = "Plot Loadshape object=Shape_1"


# assign PV generation forecast in network
# later

# assign EV behaviour in network
num_evs = round(par_evpenetration * num_households)
evs = [ElectricVehicle(cfg,sps.randint(1,num_households)) for i in range(1,num_evs)]
print(">> "+str(num_evs)+"/" +str(num_households)+" possible vehicles initialised and located.")

# generate EV availability and battery state of charge forecast
for ev in evs:
    ev.generateAvailabilityForecast()
    ev.generateBatterySOCForecast()
print(">> Vehicle availability and battery SOC forecasts generated.")

# generate electricity price forecast
day_id = rd.randint(0,999)

price_ts1 = read_timeseries("../price_timeseries/15min/priceprofile_ukpx_15min"+format(day_id,"04d")+".txt")
price_ts2 = read_timeseries("../price_timeseries/15min/priceprofile_ukpx_15min"+format(day_id+1,"04d")+".txt")
price_ts = merge_timeseries(price_ts1, price_ts2)
    
mean_price = mean(price_ts)
price_ts = [((item - mean_price) * par_spread + mean_price + par_surcharge) for item in price_ts]
print(">> Electricity market prices forecast generated.")

# ****************************************************
# * Run Optimisation
# ****************************************************



# ****************************************************
# * Run Simulation
# ****************************************************

# Compile and solve circuit

DSSSolution.Solve()
if DSSSolution.Converged:
    print (">> Simulation: The Circuit Solved Successfully")

#final DSS command: close demand interval files at end of run
#DSSText.Command = "closedi" 

# ****************************************************
# * Evaluation
# ****************************************************
