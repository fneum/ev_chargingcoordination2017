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
from operator import add

# class imports
from VehicleSpecifications import ElectricVehicle
from HouseholdSpecifications import Household

# ****************************************************
# * General Framework Initialisation
# ****************************************************

# Administrative
rd.seed(1932455)
np.set_printoptions(threshold=np.nan)

# Utility functions
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

def updateLoad(ts,id):
    dmd_dss = str(ts).replace(',', '').replace('[', '').replace(']', '')
    DSSText.Command = "Edit Loadshape.Shape_"+str(id)+" mult=("+dmd_dss+")"
    DSSText.Command = "Edit Load.LOAD"+str(id)+" daily=Shape_"+str(id)

def chargeAsFastAsPossible():
    schedules = np.zeros((num_households,num_slots))
    targetSOC = cfg.getfloat("electric_vehicles","targetSOC")
    for ev in evs:
        batterySOC = ev.batterySOC_simulated
        t = 0
        while not batterySOC == (targetSOC*ev.capacity):
            remainingEnergyDemand = targetSOC*ev.capacity-batterySOC
            chargingrate_need = remainingEnergyDemand/(ev.charging_efficiency*conv.Time(min=resolution).hr)
            chargingrate = ev.availability_simulated[t]*min(ev.chargingrate_max, chargingrate_need)
            schedules[ev.position-1][t] = chargingrate
            batterySOC+=(chargingrate*ev.charging_efficiency*conv.Time(min=resolution).hr)
            t+=1
            if t == num_slots:
                break
    return schedules

def runOptGreedy():
    return 0

def runOptParticleSwarm():
    return 0

def runOptGenetic():
    return 0

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
try:
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
    print("<< @Init: Unable to start the OpenDSS Engine")
    raise SystemExit

print(">> @Init: OpenDSS engine opened.")

# Set up the Text, Circuit, and Solution Interfaces to manage OpenDSS
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# Compile and set major parameters
DSSText.Command = r"Compile '..\network_details\Master.dss'"
DSSText.Command = "set mode=daily number="+str(num_slots)+" stepsize="+str(resolution)+"m"

for i in range(num_households):
    DSSText.Command = "New Loadshape.Shape_"+str(i+1)
    DSSText.Command = "~ npts="+str(num_slots)
    DSSText.Command = "~ minterval="+str(resolution)
    DSSText.Command = "~ useactual=true"
    
print(">> @Init: Network instantiated and compiled.")

# ****************************************************
# * Generate Scenario
# ****************************************************
print("-------------------------------------------------")

# assign residential load forecast in network
households = [Household() for i in range(num_households)]
counter = 1
for hd in households:
    hd.day_id_1 = rd.randint(1,hd.id_range)
    demand_ts1 = read_timeseries("../demand_timeseries/loadprofile_"+season+"_inh"+\
                                 str(hd.inhabitants)+"_"+str(resolution)+"min"+format(hd.day_id_1,"03d")+".txt")
    hd.day_id_2 = rd.randint(1,hd.id_range)
    demand_ts2 = read_timeseries("../demand_timeseries/loadprofile_"+season+"_inh"+\
                                 str(hd.inhabitants)+"_"+str(resolution)+"min"+format(hd.day_id_2,"03d")+".txt")
    hd.demandForecast = merge_timeseries(demand_ts1,demand_ts2)
    updateLoad(hd.demandForecast,counter)
    counter+=1
print(">> @Scen: "+str(num_households)+" households initialised and demand forecasts generated.")

# assign PV generation forecast in network
# LATER

# assign EV behaviour in network
num_evs = round(par_evpenetration * num_households)
evs = [ElectricVehicle(cfg,rd.randint(1,num_households)) for i in range(1,num_evs)]
print(">> @Scen: "+str(num_evs)+"/" +str(num_households)+" possible vehicles initialised and located.")

# generate EV availability and battery state of charge forecast
for ev in evs:
    ev.generateAvailabilityForecast()
    ev.generateBatterySOCForecast()
    
print(">> @Scen: Vehicle availability and battery SOC forecasts generated.")

# generate electricity price forecast
day_id = rd.randint(0,999)
price_ts1 = read_timeseries("../price_timeseries/15min/priceprofile_ukpx_"+str(resolution)+"min"+format(day_id,"04d")+".txt")
price_ts2 = read_timeseries("../price_timeseries/15min/priceprofile_ukpx_"+str(resolution)+"min"+format(day_id+1,"04d")+".txt")
price_ts = merge_timeseries(price_ts1, price_ts2)
mean_price = mean(price_ts)
price_ts = [((item - mean_price) * par_spread + mean_price + par_surcharge) for item in price_ts]
print(">> @Scen: Electricity market prices forecast generated.")

DSSText.Command = "set year=1"
DSSSolution.Solve()

# ****************************************************
# * Run Optimisation
# ****************************************************
print("-------------------------------------------------")

alg = cfg.get("general","algorithm") 
print(">> @Opt: "+alg+" selected as optimisation algorithm.")

if alg is "greedy":
    schedules = runOptGreedy()
elif alg is "ga":
    schedules = runOptGenetic()
elif alg is "pso":
    schedules = runOptParticleSwarm()
else:
    schedules = []

print(">> @Opt: Optimisation cycle complete.")

# ****************************************************
# * Run Simulation
# ****************************************************
print("-------------------------------------------------")

# generate actual EV behaviour
if cfg.getboolean("uncertainty","unc_ev"):
    for ev in evs:
        ev.simulateAvailability()
        ev.simulateBatterySOC()
    print(">> @Sim: Vehicle uncertainty realised.")
else:
    for ev in evs:
        ev.availability_simulated = ev.availability_forecast
        ev.batterySOC_simulated = ev.batterySOC_forecast
    print(">> @Sim: Vehicle uncertainty not realised.")

# generate actual demand behaviour
if cfg.getboolean("uncertainty", "unc_dem"):
    for hd in households:
        hd.simulateDemand()
    print(">> @Sim: Demand uncertainty realised.")
else:
    for hd in households:
        hd.demandSimulated = hd.demandForecast
    print(">> @Sim: Demand uncertainty not realised.")

# generate actual electricity prices    
if cfg.getboolean("uncertainty", "unc_pri"):
    price_ts_sim = [item + norm.rvs(0,1) for item in price_ts]
    print(">> @Sim: Price uncertainty realised.")
else:
    price_ts_sim = list(price_ts)
    print(">> @Sim: Price uncertainty not realised.")

# if no optimised schedule available -> uncontrolled charging
if not schedules:
    schedules = chargeAsFastAsPossible()

# include schedule in residential net load
for i in range(num_households):
    net_load = list(map(add,households[i].demandSimulated, schedules[i].tolist()))
    updateLoad(net_load,i+1)

# Compile and solve circuit
DSSText.Command = "set year=2"
DSSSolution.Solve()
if DSSSolution.Converged:
    print (">> @Sim: Circuit solved successfully.")

#final DSS command: close demand interval files at end of run
#DSSText.Command = "closedi" 

# ****************************************************
# * Evaluation
# ****************************************************
print("-------------------------------------------------")
print(">> @Eval: ")

# DSSText.Command = "export voltages"
# DSSText.Command = "export seqvoltages"
# DSSText.Command = "export powers"
# DSSText.Command = "export seqpowers"
# DSSText.Command = "export loads"
# DSSText.Command = "export summary"
