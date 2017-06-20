import scipy as sp
import scipy.stats as sps
import numpy.random as rd

class ElectricVehicle:
    
    def __init__(self, cfg, pos):
        
        # vehicle specifications
        self.consumption = cfg.getfloat('electric_vehicles', 'consumption')
        self.capacity = cfg.getfloat('electric_vehicles', 'capacity')
        self.position = pos
        
        # arrival time behaviour
        self.tripend_mu = sps.genlogistic.rvs(cfg.getfloat('travel_patterns','c_tem'),\
                                              loc=cfg.getfloat('travel_patterns','loc_tem'),\
                                              scale=cfg.getfloat('travel_patterns','scale_tem') )
        self.tripend_sig = sps.genlogistic.rvs(cfg.getfloat('travel_patterns','c_tes'),\
                                              loc=cfg.getfloat('travel_patterns','loc_tes'),\
                                              scale=cfg.getfloat('travel_patterns','scale_tes') )
        
        # departure time behaviour
        corcoeff = cfg.getfloat('travel_patterns','tripstart_corcoeff')
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        self.tripstart_mu = sps.gamma.ppf(U[0],\
                                      cfg.getfloat('travel_patterns','a_tsm'),\
                                      loc=cfg.getfloat('travel_patterns','loc_tsm'),\
                                      scale=cfg.getfloat('travel_patterns','scale_tsm') )
        self.tripstart_sig = sps.halfnorm.ppf(U[1],\
                                      loc=cfg.getfloat('travel_patterns','loc_tss'),\
                                      scale=cfg.getfloat('travel_patterns','scale_tss') )
        
        # daily mileage behaviour
        corcoeff = cfg.getfloat('travel_patterns','mileage_corcoeff')
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        self.mileage_mu = sps.gamma.ppf(U[0],\
                                    cfg.getfloat('travel_patterns','a_mim'),\
                                    loc=cfg.getfloat('travel_patterns','loc_mim'),\
                                    scale=cfg.getfloat('travel_patterns','scale_mim') )
        self.mileage_sig = sps.expon.ppf(U[1],\
                                     loc=cfg.getfloat('travel_patterns','loc_mis'),\
                                     scale=cfg.getfloat('travel_patterns','loc_mis') )
        
    def generateAvailabilityForecast(self):
        return 0
    
    def generateDemandForecast(self):
        return 0
    
    def simulateAvailability(self):
        return 0
    
    def simulateDemand(self):
        return 0