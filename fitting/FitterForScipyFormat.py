import scipy.io as sio
from scipy.stats import genlogistic, halfnorm, fisk, expon, gamma
import matplotlib.pyplot as plt
import scipy.stats as sps

tripend = sio.loadmat('tripend_sigmu.mat')['tripend_sigmu'].T
tripstart = sio.loadmat('tripstart_sigmu.mat')['tripstart_sigmu'].T
mileage = sio.loadmat('mileage_sigmu.mat')['mileage_sigmu'].T

c_tes,loc_tes,scale_tes = genlogistic.fit(tripend[0,:])
c_tem,loc_tem,scale_tem = genlogistic.fit(tripend[1,:])

loc_tss,scale_tss = halfnorm.fit(tripstart[0,:])
c_tsm,loc_tsm,scale_tsm = genlogistic.fit(tripstart[1,:])

loc_mis,scale_mis = expon.fit(mileage[0,:])
a_mim,loc_mim,scale_mim = gamma.fit(mileage[1,:])

print(c_tes,loc_tes,scale_tes)
print(c_tem,loc_tem,scale_tem)
print(loc_tss,scale_tss)
print(c_tsm,loc_tsm,scale_tsm )
print(loc_mis,scale_mis )
print(a_mim,loc_mim,scale_mim)


#plt.hist(sps.genlogistic.rvs(c_tes,loc=loc_tes,scale=scale_tes, size=2000),bins=30)
plt.hist(sps.genlogistic.rvs(c_tsm,loc=loc_tsm,scale=scale_tsm, size=2000),bins=96)
plt.show()
plt.hist(tripstart[1,:],bins=96)
plt.show()
