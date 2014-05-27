# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 16:05:19 2014

@author: waffles
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter



if __name__ == "__main__" :

from scipy.signal import freqz
from scipy.optimize import leastsq

signal=np.loadtxt('signal5.dat')
plt.plot(signal[:,0],signal[:,1])


t=signal[:,0]
x=signal[:,1]

#Fourier filtering

signal=x.copy()
W=scipy.fftpack.fftfreq(signal.size, d=t[1]-t[2])
fsignal=scipy.fftpack.rfft(signal)

cut=fsignal.copy()
for i in range(len(fsignal)):
    if i>1103 or i<976: cut[i]=0

csignal=scipy.fftpack.irfft(cut)

plt.subplot(221)
plt.plot(t,signal)
plt.subplot(222)
plt.plot(fsignal)
plt.xlim(0,5000)
plt.subplot(223)
plt.plot(cut)
plt.xlim(0,5000)
plt.subplot(224)
plt.plot(t,csignal)
plt.show()


plt.plot(smoothsignal)
#plotting both signals
plt.plot(t,x)
plt.plot(t,smoothsignal)

###Technique 2
from scipy.optimize import curve_fit

def func(x,a,b,c,d):
    return -a*np.sin(b*x+c)+d


x=t[250:2000]

yn=csignal[250:2000]
popt, pcov=curve_fit(func,x,yn,(.27,3300,1,0))

ynew=func(x,*popt)

"""Plotting The Curve Fitting"""

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Curve Fitting Results')
ax.set_xlabel('time(s)')
ax.set_ylabel('amplitude')

ax.plot(x,signal[250:2000],'g', label="original signal")
ax.plot(x,csignal[250:2000],'ro',lw=3, label="smoothed signal")
ax.plot(x,ynew, 'b--', label="fit sin wave")

ax.text(0.95, 0.01, "freq: %.3s amp: %.6s phase: %.3s" % (popt[1]/(2*np.pi), -popt[0],popt[2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)
plt.show()
plt.legend()


####ODE PART######


from scipy.integrate import odeint
omega=523*np.pi*2

def deriv(y,t):
    uprime=y[1]
    wprime=-y[0]*(omega**2)
    yprime=np.array([uprime,wprime])
    return yprime
    
start=0
end=1
numsteps=10000
time=np.linspace(start,end,numsteps)
y0=np.array([0.244,0])
y=scipy.integrate.odeint(deriv,y0,time)  

plt.plot(time,y[:,0])

##ploting ode results with original smoothed signal.
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('ODE Results vs. Original Smoothed Signal')
ax.set_xlabel('time(s)')
ax.set_ylabel('amplitude')

ax.plot(x,csignal[250:2000],'g', label="smoothed signal")
ax.plot(time,y[:,0],'r--',lw=3, label="ODE Results")

ax.text(0.95, 0.01, "freq: %.3s hz amp: %.6s phase: %.3s" % (523, -popt[0],0),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)
plt.show()
plt.legend()

##ODE with restoring force.

omega=523*np.pi*2
omega2=omega + 100
F=omega**2 - 100

def deriv(y,t):
    uprime=y[1]
    wprime=-y[0]*(omega**2)+F*np.sin(omega2*t)
    yprime=np.array([uprime,wprime])
    return yprime
    
start=0
end=.5
numsteps=1000000
time=np.linspace(start,end,numsteps)
y0=np.array([0.244,0])
y=scipy.integrate.odeint(deriv,y0,time)  

omega3=omega
F2=-1000000

def deriv2(y,t):
    uprime=y[1]
    wprime=-y[0]*(omega**2)+F2*np.sin(omega3*t)
    yprime=np.array([uprime,wprime])
    return yprime

time=np.linspace(start,end,numsteps)
y0=np.array([0.244,0])
y2=scipy.integrate.odeint(deriv2,y0,time)  


##ODE Solution with beats.

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Beats from restoring force')
ax.set_xlabel('time(s)')
ax.set_ylabel('amplitude')

ax.plot(time,y[:,0],'g', label="Beats signal")
ax.plot(time,y2[:,0],'b--', label="Powerful external force")

ax.text(0.95, 0.01, "freq. ext. force: %.3s F_0: %.6s" % (omega2, F),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.text(0.95, 0.08, "freq. ext. force: %.3s F_0: %.6s" % (omega3, F2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)
plt.show()
plt.legend()

#results from different omegas, data gathered from ipython console
omega=523*np.pi*2
l=np.zeros((18,2)) 
l[0]=(omega + 100)/10, 1.27
l[1]=(omega + 100)/7, 1.3
l[2]=(omega + 100)/5, 1.37
l[3]=(omega + 100)/3, 1.61
l[4]=(omega + 100)/2, 2.1
l[5]=(omega + 100)/1.5, 3.21
l[6]=(omega + 100)/1.3, 4.8
l[7]=(omega + 100)/1.1, 15.7
l[8]=(omega + 100)/1, 33
l[9]=(omega + 100)*1.1, 7.5
l[10]=(omega + 100)*1.2, 4.26
l[11]=(omega + 100)*1.3, 3
l[12]=(omega + 100)*1.5, 1.85
l[13]=(omega + 100)*2, .98
l[14]=(omega + 100)*3, .55
l[15]=(omega + 100)*5, .35
l[16]=(omega + 100)*7, .30
l[17]=(omega + 100)*10, .27

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Effects of Driving Force Angular Frequency on Max Amplitude')
ax.set_xlabel('frequency of driving force')
ax.set_ylabel('maximum amplitude')

ax.plot(l[:,0],l[:,1],'g', label="Beats signal")


