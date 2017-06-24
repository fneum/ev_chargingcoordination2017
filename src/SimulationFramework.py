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
        ev.schedule = schedules[ev.position-1].tolist()
#         print(ev.schedule)
#         print(schedules[ev.position-1].tolist())
#         print("------")
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
surcharge = cfg.getfloat("market_prices","surcharge")
spread = cfg.getfloat("market_prices","spread")
evpenetration = cfg.getfloat("electric_vehicles","penetration")
start = conv.Time(hr=cfg.getfloat("general","starting")).min 
duration = conv.Time(hr=cfg.getint("general","duration")).min
resolution = cfg.getint("general","resolution")
season = cfg.get("general", "season")
reg_price = cfg.get("market_prices", "regulation_price")

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
DSSBus = DSSCircuit.ActiveBus
DSSMonitors = DSSCircuit.Monitors

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
num_evs = round(evpenetration * num_households)
deck = list(range(num_households))
rd.shuffle(deck)
evs = []
for i in range(num_evs):
    ev_household_id = deck.pop()
    households[ev_household_id].ev = ElectricVehicle(cfg,ev_household_id)
    evs.append(households[ev_household_id].ev)

print(">> @Scen: "+str(len(evs))+"/" +str(num_households)+" possible vehicles initialised and located.")

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
price_ts = [((item - mean_price) * spread + mean_price + surcharge) for item in price_ts]
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
netloads = []
for i in range(num_households):
    netload = list(map(add,households[i].demandSimulated, households[i].ev.schedule))
    netloads.append(netload)
    updateLoad(netload,i+1)

# Solve circuit
household_voltages = []
DSSText.Command = "reset"
DSSText.Command = "set year=2"
DSSSolution.Solve()
if DSSSolution.Converged:
    print (">> @Sim: Circuit solved successfully.")
    
# export voltage profiles and link to household
DSSMonitors.SaveAll
for i in range(1,num_households+1):
    #DSSMonitors.Name = "VI_MON"+str(i)
    household_voltages.append(list(DSSMonitors.Channel(1)))
    households[i-1].voltages = household_voltages[i-1]

#final DSS command: close demand interval files at end of run
#DSSText.Command = "closedi" 

# ****************************************************
# * Evaluation
# ****************************************************
print("-------------------------------------------------")
print(">> @Eval: ")

log_hd = open("../log/simResults_HouseholdAggregate.csv", 'w', newline='')
hdlog_writer = csv.writer(log_hd,delimiter=',')
hdlog_writer.writerow( ( 'id', 'inhabitants', 'withEV', 'chCostTotal', 'regRevTotal', 'netChCostTotal','resCostTotal','totalCostTotal',\
                      'netDemandTotal', 'evDemandTotal', 'resDemandTotal', 'pvGenTotal', 'minVoltageV','minVoltagePU' ) )

# instantiate arrays for postcalculations
chCost = np.zeros((num_households,num_slots))
netChCost = np.zeros((num_households,num_slots))
totalCost = np.zeros((num_households,num_slots))
resCost = np.zeros((num_households,num_slots))
regAv  = np.zeros((num_households,num_slots))
regRev = np.zeros((num_households,num_slots))
eCharged = np.zeros((num_households,num_slots))
batterySOC = np.zeros((num_households,num_slots))
av = np.zeros((num_households,num_slots))
resDemand = []

j = 0
for hd in households:
     
    resDemand.append(hd.demandSimulated)
     
    for i in range(0,num_slots):
        if hd.ev is None:
            av[j][i] = 0
        else:
            av[j][i] = hd.ev.availability_simulated[i]
        chCost[j][i] = hd.ev.schedule[i]*conv.Time(min=resolution).hr*price_ts_sim[i]/100
        regAv[j][i] = 0 
        regRev[j][i] = 0
        netChCost[j][i] = chCost[j][i]-regRev[j][i]
        eCharged[j][i] = hd.ev.schedule[i]*hd.ev.charging_efficiency*conv.Time(min=resolution).hr
        resCost[j][i] = hd.demandSimulated[i]*conv.Time(min=resolution).hr*price_ts_sim[i]/100
        totalCost[j][i] = resCost[j][i]+netChCost[j][i]
        if i == 0:
            batterySOC[j][i] = av[j][i]*(hd.ev.batterySOC_simulated+eCharged[j][i])
        else:
            batterySOC[j][i] = av[j][i]*(max(hd.ev.batterySOC_simulated+eCharged[j][i],batterySOC[j][i-1]+eCharged[j][i]))
     
    # WRITE individual household solutions to CSV
    with open("../log/simResults_household"+format(j+1,"02d")+".csv", 'w', newline='') as f:
        try:
            solution_writer = csv.writer(f,delimiter=',')
            solution_writer.writerow( ( 'slot', 'netLoad', 'resLoad', 'pvGen','evSchedule',\
                               'evAvailability', 'regAvailability', 'energyCharged','batterySOC',\
                               'voltageV','voltagePU', 'elPrice', 'chCost', 'regRev', 'netChCost','resCost','totalCost' ) )
            for i in range(num_slots):
                solution_writer.writerow( ( (i+1), netloads[j][i], hd.demandSimulated[i], 0, hd.ev.schedule[i],\
                                    av[j][i], 0, eCharged[j][i],batterySOC[j][i], hd.voltages[i], hd.voltages[i]/230, price_ts_sim[i],chCost[j][i],\
                                    regRev[j][i],netChCost[j][i],resCost[j][i],totalCost[j][i]) )
        finally:
            f.close()
             
    hdlog_writer.writerow( ( (j+1), hd.inhabitants, max(av[j]), sum(chCost[j]), sum(regRev[j]), sum(netChCost[j]), sum(resCost[j]),\
                          sum(totalCost[j]), conv.Time(min=duration).hr*sum(netloads[j])/len(netloads[j]),\
                          conv.Time(min=duration).hr*sum(hd.ev.schedule)/len(hd.ev.schedule),\
                          conv.Time(min=duration).hr*sum(hd.demandSimulated)/len(hd.demandSimulated), 0, min(hd.voltages), min(hd.voltages)/230 ) )
    j+=1

log_hd.close()

# SLOTWISE AGGREGATE LOG
log_slot = open("../log/simResults_SlotwiseAggregate.csv", 'w', newline='')
slotlog_writer = csv.writer(log_slot,delimiter=',')
slotlog_writer.writerow( ( 'slot', 'netLoad', 'resLoad', 'pvGen','evSchedule',\
                   'evAvailability', 'regAvailability', 'batterySOC',\
                   'minVoltageV','minVoltagePU', 'elPrice', 'chCost', 'regRev', 'netChCost','resCost','totalCost' ) )

for i in range(num_slots):
    slotlog_writer.writerow( ( (i+1), sum(np.asarray(netloads).T[i]), sum(np.asarray(resDemand).T[i]), 0,\
                               sum(np.asarray(schedules).T[i]), sum(av.T[i]),sum(regAv.T[i]),\
                               sum(batterySOC.T[i])/(sum(av.T[i])*ev.capacity),min(np.asarray(household_voltages).T[i]),\
                               min(np.asarray(household_voltages).T[i])/230,price_ts_sim[i],sum(chCost.T[i]),\
                               sum(regRev.T[i]),sum(netChCost.T[i]),sum(resCost.T[i]),sum(totalCost.T[i]) ) )
log_slot.close()

# WRITE SOME MORE FILES
# WRITE FILES
np.savetxt("../log/simResults_Voltages.csv", np.asarray(household_voltages), delimiter=",")
np.savetxt("../log/simResults_Schedules.csv", np.asarray(schedules), delimiter=",")
np.savetxt("../log/simResults_NetLoads.csv", np.asarray(netloads), delimiter=",")
np.savetxt("../log/simResults_ResLoads.csv", np.asarray(resDemand), delimiter=",")
np.savetxt("../log/simResults_EVAvailability.csv", np.asarray(av), delimiter=",")
np.savetxt("../log/simResults_NetChCost.csv", np.asarray(netChCost), delimiter=",")
np.savetxt("../log/simResults_TotalCost.csv", np.asarray(totalCost), delimiter=",")
np.savetxt("../log/simResults_ResCost.csv", np.asarray(resCost), delimiter=",")
np.savetxt("../log/simResults_BatterySOC.csv", np.asarray(batterySOC), delimiter=",")
np.savetxt("../log/simResults_RegAvailability.csv", np.asarray(regAv), delimiter=",")

# chCost = np.zeros((num_households,num_slots))
# netChCost = np.zeros((num_households,num_slots))
# totalCost = np.zeros((num_households,num_slots))
# resCost = np.zeros((num_households,num_slots))
# regAv  = np.zeros((num_households,num_slots))
# regRev = np.zeros((num_households,num_slots))
# eCharged = np.zeros((num_households,num_slots))
# batterySOC = np.zeros((num_households,num_slots))
# av = np.zeros((num_households,num_slots))

# DSSText.Command = "export voltages"
# DSSText.Command = "export seqvoltages"
# DSSText.Command = "export powers"
# DSSText.Command = "export seqpowers"
# DSSText.Command = "export loads"
# DSSText.Command = "export summary"

print(">>>  Programme terminated.")