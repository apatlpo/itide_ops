import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def generate_slow(t=None,T=365.*3,dt=1./24.,Tcut=50,rms=0.2):
    """ generate a sea level corresponding to QG turbulence, i.e.
    a flat spectrum at frequencies lower that Tcut
        
    Parameters
    ----------
        t:
            time line in days
        T:
            time period in days, default corresponds to 3 years
        dt:
            time interval in days, default corresponds to 1h
        Tcut:
            period cutoff for the spectrum in days, default is 50 days
        rms:
            signal rms, default is 0.2
    """
    
    if t is None:
        t=np.arange(0.,T,dt)
    N=t.size
    
    # build frequency line
    omega = 2.*np.pi*np.fft.fftfreq(N,dt)
    omega_cut = 2.*np.pi/Tcut
    
    # create a distribution with right spectral properties
    eta_f = np.exp(1j*np.random.uniform(high=1.,size=N)*2.*np.pi)
    eta_f[np.where(np.abs(omega)>omega_cut)] = eta_f[np.where(np.abs(omega)>omega_cut)] \
                                * omega_cut/omega[np.where(np.abs(omega)>omega_cut)]
    
    # inverse fft
    eta=np.real(np.fft.ifft(eta_f))
    
    # renormalize
    eta = rms * eta / np.std(eta)
    
    return eta, t

def generate_tide(t=None,T=365.*3,dt=1./24.,
                  omega=2.,rms=0.01, omega2=None, 
                  Tmod=20., Amod=0.001,plot=False):
    """ generate a tidal (time periodic) signal
    The signal is by default the sum of a stationary signal and a nonstationary signal (time modulated
    noise with cutoff timescale and amplitude prescribed).
    It can also be a fully stationary signal with one or two frequencies
    
    Parameters
    ----------
        t:
            time line in days
        T:
            time period in days, default corresponds to 3 years
        dt:
            time interval in days, default corresponds to 1h
        omega:
            iwave frequency in cycle per days, default is 2 cpd
        rms:
            iwave rms
        omega2:
            second iwave frequency in cycle per days
        Tmod:
            modulation cutoff time scale in days
        Amod:
            modulation amplitude in rms units
        
    """
    
    if t is None:
        t=np.arange(0.,T,dt)
    N=t.size
    dt=t[1]-t[0]
    
    #
    eta = np.cos(2.*np.pi*omega*t)

    # spring neap type modulations
    if omega2 is not None:
        eta += 0.5*np.cos(2.*np.pi*omega2*t)

    # noisy/unstationary modulations
    if Tmod is not None:
        # start from normal random variables
        Ar = np.random.normal(scale=Amod/rms,size=2*t.size)
        Ai = np.random.normal(scale=Amod/rms,size=2*t.size)
        # low pass filter time series
        Ar = low_pass(Ar,t,Tmod,plot=False)[np.int(0.5*t.size):][:t.size]
        Ai = low_pass(Ai,t,Tmod,plot=False)[np.int(0.5*t.size):][:t.size]
        # renormalize rms
        Ar = 1. + Ar/np.std(Ar) * Amod/rms
        Ai = 0. + Ai/np.std(Ai) * Amod/rms
        #
        A = Ar+1j*Ai
        # = 
        #phi0 = np.pi*np.random.uniform(low=-1.,high=1.,size=2*t.size)
        #phi = low_pass(phi0,t,Tmod,plot=False)
        #phi = phi[np.int(0.5*t.size):]; phi = phi[:t.size]
        #phi = np.pi*phi/np.std(phi)
        #
        # A = 1 # could it be random too?
        #
        if plot:
            plt.figure()
            #phi0 = phi0[np.int(0.5*t.size):]; phi0 = phi0[:t.size]
            #plt.plot(t,phi0,'k', label())
            #plt.plot(t,phi,'k')
            plt.plot(t,Ar,'k',label='real(A)')
            plt.plot(t,Ai,'r',label='imag(A)')
            plt.legend(loc=0)
            plt.grid()
            plt.title('Phase of the tidal signal')
            plt.show()
        #eta = np.cos(2.*np.pi*omega*t+phi)
        eta = np.real(A * np.exp(1j*2.*np.pi*omega*t))

    # renormalize
    eta = rms * eta / np.std(eta)
    
    return eta, t