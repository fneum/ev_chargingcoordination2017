'''
Created on 31 May 2017

@author: Fabian Neumann
'''

import win32com.client

#****************************************************
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

# Compile and solve circuit
DSSText.Command = r"Compile 'C:\Users\Fabian Neumann\OneDrive\Studium\UOE\Dissertation\Code\ev_chargingcoordination2017\network_details\Master.dss'"
DSSText.Command = "set mode=yearly number=1440 stepsize=1m"
DSSText.Command = "solve"

DSSText.Command = "Show mon LINE558_VI_vs_Time "
DSSText.Command = "Plot monitor object= line558_vi_vs_time channels=(1 3 5 ) bases=[230 230 230]"