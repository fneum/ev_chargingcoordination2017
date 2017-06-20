import win32com.client
import configparser
import numpy.random as rd
import numpy as np
import scipy as sp
import scipy.stats as sps
import csv
import fileinput
from statistics import mean

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
config = configparser.ConfigParser()
config.read('../parameters/evalParams.ini')

# assign parameter names
par_surcharge = config.getfloat('market_prices','surcharge')
par_spread = config.getfloat('market_prices','spread')
par_evpenetration = config.getfloat('electric_vehicles','penetration')
par_tripend_av_dist = 'loglogistic'
par_tripend_av_mu = config.getfloat('travel_patterns','tripend_av_mu')
par_tripend_av_sigma = config.getfloat('travel_patterns','tripend_av_sigma')
par_tripend_sd_dist = 'halfnormal'
par_tripend_sd_mu = config.getfloat('travel_patterns','tripend_sd_mu')
par_tripend_sd_sigma = config.getfloat('travel_patterns','tripstart_av_mu')
par_tripstart_av_dist = 'tlocationscale'
par_tripstart_av_mu = config.getfloat('travel_patterns','tripstart_av_mu')
par_tripstart_av_sigma = config.getfloat('travel_patterns','tripstart_av_sigma')
par_tripstart_av_nu = config.getfloat('travel_patterns','tripstart_av_nu')
par_tripstart_sd_dist = 'logistic'
par_tripstart_sd_mu = config.getfloat('travel_patterns','tripstart_sd_sigma')
par_tripstart_sd_sigma = config.getfloat('travel_patterns','tripstart_sd_sigma')
par_tripstart_corcoeff = config.getfloat('travel_patterns','tripstart_corcoeff')
par_mileage_av_dist = 'gamma'
par_mileage_av_a = config.getfloat('travel_patterns','mileage_av_a')
par_mileage_av_b = config.getfloat('travel_patterns','mileage_av_b')
par_mileage_sd_dist = 'exponential'
par_mileage_sd_mu = config.getfloat('travel_patterns','mileage_sd_mu')
par_mileage_corcoeff = config.getfloat('travel_patterns','mileage_corcoeff')

#print (config.getboolean('uncertainty', 'unc_ev',fallback='Request not in parameter set!'))

# ****************************************************
# * Generate Scenario
# ****************************************************

# assign residential load forecast in network
num_households = 55

# assign PV generation forecast in network

# assign EV behaviour in network
num_evs = round(par_evpenetration * num_households)

Z = np.random.multivariate_normal([0, 0], [[1, par_mileage_corcoeff], [par_mileage_corcoeff, 1]],num_evs)
U = sps.norm.cdf(Z)
print(np.corrcoef(Z.T)) 




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
