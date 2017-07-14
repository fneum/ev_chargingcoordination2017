from VehicleSpecifications import ElectricVehicle
import numpy.random as rd

class Household:

    def __init__(self):
        x = rd.random()
        self.inhabitants = 0
        self.id_range = 0
        if x < 0.34:
            self.inhabitants = 1
            self.id_range = 340
        elif x < 0.74:
            self.inhabitants = 2
            self.id_range = 400
        elif x < 0.88:
            self.inhabitants = 3
            self.id_range = 140
        elif x < 0.97:
            self.inhabitants = 4
            self.id_range = 90
        else:
            self.inhabitants = 5
            self.id_range = 30
        
        self.demandForecast = []
        self.demandSimulated = []
        
        self.day_id_1 = 0
        self.day_id_2 = 0
        
        self.ev = None
        self.voltages = None
        
