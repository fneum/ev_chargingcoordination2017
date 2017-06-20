import scipy as sp
import scipy.stats as sps
import numpy.random as rd

class ElectricVehicle:
    
   
    
    def __init__(self, cfg, pos):
        
        eng = matlab.engine.start_matlab()
        self.consumption = cfg.getfloat('electric_vehicles', 'consumption')
        self.capacity = cfg.getfloat('electric_vehicles', 'capacity')
        self.position = pos
        
        self.tripend_mu = 
        print(self.tripend_mu)
        #self.tripend_sig = 0
        
        corcoeff = cfg.getfloat('travel_patterns','tripstart_corcoeff')
        
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        #self.tripstart_mu = 
        #self.tripstart_sig = 
        
        corcoeff = cfg.getfloat('travel_patterns','mileage_corcoeff')
        Z = rd.multivariate_normal([0, 0], [[1, corcoeff], [corcoeff, 1]])
        U = sps.norm.cdf(Z)
        #self.mileage_mu = 
        #self.mileage_sig = 
        
        #Z = np.random.multivariate_normal([0, 0], [[1, par_mileage_corcoeff], [par_mileage_corcoeff, 1]])
        #U = sps.norm.cdf(Z)
        
    def generateAvailabilityForecast(self):
        return 0
    
    def generateDemandForecast(self):
        return 0
    
    def simulateAvailability(self):
        return 0
    
    def simulateDemand(self):
        return 0