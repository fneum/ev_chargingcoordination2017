import scipy.io as sio
from scipy.stats import genlogistic, halfnorm, fisk, expon, gamma

tripend = sio.loadmat('tripend_sigmu.mat')['tripend_sigmu'].T
tripstart = sio.loadmat('tripstart_sigmu.mat')['tripstart_sigmu'].T
mileage = sio.loadmat('mileage_sigmu.mat')['mileage_sigmu'].T

c_tes,loc_tes,scale_tes = genlogistic.fit(tripend[0,:])
c_tem,loc_tem,scale_tem = genlogistic.fit(tripend[1,:])

loc_tss,scale_tss = halfnorm.fit(tripstart[0,:])
a_tsm,loc_tsm,scale_tsm = gamma.fit(tripstart[1,:])

loc_mis,scale_mis = expon.fit(mileage[0,:])
a_mim,loc_mim,scale_mim = gamma.fit(mileage[1,:])

print(c_tes,loc_tes,scale_tes)
print(c_tem,loc_tem,scale_tem)
print(loc_tss,scale_tss)
print(a_tsm,loc_tsm,scale_tsm )
print(loc_mis,scale_mis )
print(a_mim,loc_mim,scale_mim)