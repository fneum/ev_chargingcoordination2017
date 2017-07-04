import scipy as sp
import scipy.stats as sps
import numpy.random as rd
from scipy.stats._continuous_distns import norm, triang
import math
import measurement.measures as conv
from builtins import int

class ElectricVehicle:
    
    def __init__(self, cfg, pos):
        
        # config file
        self.cfg = cfg
        
        # vehicle specifications
        self.consumption = cfg.getfloat('electric_vehicles', 'consumption')
        self.capacity = cfg.getfloat('electric_vehicles', 'capacity')
        self.charging_efficiency = cfg.getfloat("electric_vehicles","charging_efficiency")
        self.chargingrate_max = cfg.getfloat("electric_vehicles", "chargingrate_max")
        self.position = pos
        
        # arrival time behaviour
        while True:
            self.tripend_mu = sps.genlogistic.rvs(cfg.getfloat('travel_patterns','c_tem'),\
                                                  loc=cfg.getfloat('travel_patterns','loc_tem'),\
                                                  scale=cfg.getfloat('travel_patterns','scale_tem') )
            if self.tripend_mu > 0:
                break
        while True:
             self.tripend_sig = sps.genlogistic.rvs(cfg.getfloat('travel_patterns','c_tes'),\
                                                  loc=cfg.getfloat('travel_patterns','loc_tes'),\
                                                  scale=cfg.getfloat('travel_patterns','scale_tes') )
             if self.tripend_sig > 0:
                break
        
        # departure time behaviour
        corcoeff = cfg.getfloat('travel_patterns','tripstart_corcoeff')
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        while True:
            self.tripstart_mu = sps.genlogistic.ppf(U[0],\
                                      cfg.getfloat('travel_patterns','c_tsm'),\
                                      loc=cfg.getfloat('travel_patterns','loc_tsm'),\
                                      scale=cfg.getfloat('travel_patterns','scale_tsm') )
            if self.tripstart_mu > 0:
                break
        while True:
            self.tripstart_sig = sps.halfnorm.ppf(U[1],\
                                      loc=cfg.getfloat('travel_patterns','loc_tss'),\
                                      scale=cfg.getfloat('travel_patterns','scale_tss') )
            if self.tripstart_sig > 0:
                break
            
        # daily mileage behaviour
        corcoeff = cfg.getfloat('travel_patterns','mileage_corcoeff')
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        while True:
            self.mileage_mu = sps.gamma.ppf(U[0],\
                                    cfg.getfloat('travel_patterns','a_mim'),\
                                    loc=cfg.getfloat('travel_patterns','loc_mim'),\
                                    scale=cfg.getfloat('travel_patterns','scale_mim') )
            if self.mileage_mu > 0:
                break
        while True:
            self.mileage_sig = sps.expon.ppf(U[1],\
                                     loc=cfg.getfloat('travel_patterns','loc_mis'),\
                                     scale=cfg.getfloat('travel_patterns','scale_mis') )
            if self.mileage_sig > 0:
                break
        self.availability_forecast = []
        self.availability_simulated = []
        
        self.batterySOC_forecast = 0
        self.batterySOC_simulated = 0
        
        self.schedule = []
        
    def generateAvailabilityForecast(self):
        start = conv.Time(hr=self.cfg.getfloat('general','starting')).min 
        duration = conv.Time(hr=self.cfg.getint('general','duration')).min
        resolution = self.cfg.getint('general','resolution')
        num_slots = int(duration/resolution)
        availability_start = math.floor(max(0,self.tripend_mu-start)/resolution)
        availability_end = math.floor(min(duration,(duration-start)+self.tripstart_mu)/resolution)
        for i in range(num_slots):
            if i<availability_start:
                self.availability_forecast.append(0)
            elif i<availability_end:
                self.availability_forecast.append(1)
            else:
                self.availability_forecast.append(0)
        return self.availability_forecast
    
    def generateBatterySOCForecast(self):
        self.batterySOC_forecast = max(0,self.capacity-conv.Distance(mi=self.mileage_mu).km*self.consumption)
        return self.batterySOC_forecast
        
    def simulateAvailability(self):
        start = conv.Time(hr=self.cfg.getfloat('general','starting')).min 
        duration = conv.Time(hr=self.cfg.getint('general','duration')).min
        resolution = self.cfg.getint('general','resolution')
        num_slots = int(duration/resolution)
        # TODO choose between triangular and normal distribution
        actual_end = triang.rvs(0.5,\
                                loc=(self.tripend_mu-self.tripend_sig),\
                                scale=(2*self.tripend_sig) )
                                #scale=(4*self.tripend_sig) )
        actual_start = triang.rvs(0.5,\
                                loc=(self.tripstart_mu-self.tripstart_sig),\
                                scale=(2*self.tripstart_sig) )
                                #scale=(4*self.tripstart_sig) )
        availability_start = math.floor(max(0,actual_end-start)/resolution)
        availability_end = math.floor(min(duration,(duration-start)+actual_start)/resolution)
        for i in range(num_slots):
            if i<availability_start:
                self.availability_simulated.append(0)
            elif i<availability_end:
                self.availability_simulated.append(1)
            else:
                self.availability_simulated.append(0)
        return self.availability_simulated
    
    def simulateBatterySOC(self):
        self.batterySOC_simulated = max(0,self.capacity-conv.Distance(mi=norm.rvs(self.mileage_mu,self.mileage_sig)).km*self.consumption)
        return self.batterySOC_simulated