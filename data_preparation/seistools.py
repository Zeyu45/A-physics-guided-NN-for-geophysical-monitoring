import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import toeplitz

def gen_ricker(nt, dt, f0, t0):
    """
    Generate a Ricker wavelet as a 1D numpy vector
    nt - number of samples in the signal
    dt - time sampling of the signal
    f0 - central frequency of the Rocker wavelet
    t0 - time location of the peak of the Ricker wavelet
         to avoid wrap around of side lobe, choose t0>1/f0.
    For info on Ricker wavelet: https://wiki.seg.org/wiki/Ricker_wavelet
    """
    # Start of actual function
    tax=np.arange(0,nt*dt,dt)
    tmp= np.pi ** 2 * f0 ** 2 * (tax - t0) ** 2
    wav=(1-2*tmp)*np.exp(-tmp)

    return wav

def wiggle_plot(data, dt=0.004, scale=1.0, fill_positive=False, figsize=(12,6), title='Wiggle plot', ax=None):
    """
    Make a wiggle plot of seismic data.
    
    Parameters:
    - data: 2D numpy array with shape (n_samples, n_traces)
    - dt: sample interval in seconds
    - scale: scale factor for amplitudes
    - fill_positive: True to fill positive wiggles
    - figsize: size of the figure
    - title: title displayed above the wiggle plot
    - ax: existing matplotlib axes object (optional)
    """
    n_samples, n_traces = data.shape
    t = np.arange(n_samples) * dt
    offset=np.max(np.abs(data))
    if offset < 1e-33:
        offset = 1.0
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # make normalized plot
    for i in range(n_traces):
        trace = data[:, i] * scale / offset
        x = i + trace
        y = t
        
        ax.plot(x, y, color='black', linewidth=0.8)
        
        if fill_positive:
            ax.fill_betweenx(y, i, x, where=(trace > 0), facecolor='black', interpolate=True, alpha=0.6)

    ax.invert_yaxis()
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time [s]')
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    return ax

##############################################################
# function calc_2way_time
# calculate tt for possible src to rcv reflection path
# Author: Eric Verschuur, Delft University of Technology
# Date  : April 2025 (adapted from one-way version)
##############################################################

def calc_2way_time(nrefl,z0,a,b,v,vgrad,xpath):
#
# calculate the 2-way traveltime of a proposed xpath
# within a medium with nrefl reflectors
# per reflector k we have interface:
# z(x)=z0[k]+a[k]*x+b[k]*x^2
# xpath should have size 2*nrefl+1
# xpath[0] is the source, xpath[end]=recevier x-location
# along the path we have velocity values v[k] and 
# velocity-gradient values vgrad[k]
# the local velocity is based on average x-value
# the function returns the travel time and the zpath
#

    # determine lengths of the various arrays
    lenv=len(v)
    lenvgrad=len(vgrad)
    lenz0=len(z0)
    lena=len(a)
    lenb=len(b)
    lenx=len(xpath)
    
    # check the size of the arrays compared to the number of reflectors
    if lenv < nrefl:
        print('error in calc_2way_time: too small number of v-values')
        return -1,0.0*xpath
    if lenvgrad < nrefl:
        print('error in calc_2way_time: too small number of vgrad-values')
        return -1,0.0*xpath
    if lenz0 < nrefl:
        print('error in calc_2way_time: too small number of zo-values')
        return -1,0.0*xpath
    if lena < nrefl:
        print('error in calc_2way_time: too small number of a-values')
        return -1,0.0*xpath
    if lenb < nrefl:
        print('error in calc_2way_time: too small number of b-values')
        return -1,0.0*xpath
    if lenx != 2*nrefl+1:
        print('error in calc_2way_time: size xpath not consistent with nrefl')
        return -1,0.0*xpath

    # Calculate the zpath array and the traveltime
    # first half is down, send half is up
    tt=0
    zpath=0.0*xpath
    for k in range(lenx-1):
        # determine z-location of path at the corresponding reflector
        if k < nrefl:
            zpath[k+1]=z0[k]+a[k]*xpath[k+1]+b[k]*xpath[k+1]**2
        elif k < (lenx-2):
            z00=z0[2*nrefl-k-2]
            aa=a[2*nrefl-k-2]
            bb=b[2*nrefl-k-2]
            zpath[k+1]=z00+aa*xpath[k+1]+bb*xpath[k+1]**2
        # determine corresponding velocity by averaging the slowness
        xmid=(xpath[k+1]+xpath[k])/2.0
        if k < nrefl:
            vloc=v[k]+vgrad[k]*xmid
        else:
            vloc=v[2*nrefl-k-1]+vgrad[2*nrefl-k-1]*xmid
        #if k ==1:
        #    print('xmid=%f vloc=%f' %(xmid,vloc))
        #if k ==2:
        #    print('xmid=%f vloc=%f' %(xmid,vloc))
        tt=tt+np.sqrt((xpath[k+1]-xpath[k])**2+(zpath[k+1]-zpath[k])**2)/vloc
    
    return tt,zpath

# just a test of this function
# z0=[200,500,800]
# a=[0.1,-0.1,0.2]
# b=[0.0,0.0001,0.0]
# v=[1500,2000,2500]
# vgrad=[0.0,0.1,0.0]
# xpath=np.arange(100,701,100)
# nrefl=len(z0)
# tt,zpath=calc_2way_time(nrefl,z0,a,b,v,vgrad,xpath)
# print('tt is %f' %(tt))


##############################################################
# function calc_2way_ray_path
# determine the src to rcv optimum reflection path
# in a layered medium with parabola-shaped reflectors
# Author: Eric Verschuur, Delft University of Technology
# Date  : April 2025 (adapted from one-way version)
##############################################################

def calc_2way_ray_path(nrefl,z0,a,b,v,vgrad,xs,xr):
    """
# Determine the travel time and ray path from (xs,zs) via a reflector to (xr,zr) 
# where zs and zr are at the surface (z=0), in a 2D model with dipping reflectors:
# reflector k+1 with pivot point (x=0,z0[k]) and directional coeff a[k]=dz/dx=tan(alpha)
# and also a curvature b[k], so the interface is described by
#       z(x)=z0[k]+a[k]*x+b[k]*x^2
# Within the layer above each reflector, we have velocity v[k] in m/s
# and a lateral gradient vgrad[k] in (m/s)/m, such that the lateral varying
# local velocity is v_loc(x)=v[k]+vgrad[k]*x
#
# It will return the travel time from source via the reflectors to the rcv,
# and the nrefl parameter determines which reflector is reflecting the wave
#
# nrefl   = number of reflectors to consider in the model, reflection at nrefl
# z0[]    = pivot points in depth for each reflector (assumed ad x=0)
# a[]     = slope of the reflector
# b[]     = curvature of the reflector
# v[]     = velocity in each layer
# vgrad[] = lateral gradient within layer
# xs, xr  = source and receiver x-location; depth is at z=0
#
# Author: Eric Verschuur, Delft University of Technology
# Date  : April 2025 (adapted from one-way version)
    """ 

    # define some parameters: perturbation for gradient and initial step
    dx=0.1
    step=200
    maxiter=100

    # check sizes of input arrays
    lenv=len(v)
    lenvgrad=len(vgrad)
    lenz0=len(z0)
    lena=len(a)
    lenb=len(b)
    
    #-------------------------------------------------------------------------
    # check the size of arrays compared to the deepest reflector of interest
    #-------------------------------------------------------------------------
    if lenv < nrefl:
        print('error in calc_2way_ray_path: too small number of v-values')
        return -1
    if lenvgrad < nrefl:
        print('error in calc_2way_ray_path: too small number of vgrad-values')
        return -1
    if lenz0 < nrefl:
        print('error in calc_2way_ray_path: too small number of zo-values')
        return -1
    if lena < nrefl:
        print('error in calc_2way_ray_path: too small number of a-values')
        return -1
    if lenb < nrefl:
        print('error in calc_2way_ray_path: too small number of b-values')
        return -1

    # define the arrays for x-locations and fill it with an initial path for x
    xpath=np.linspace(start=xs,stop=xr,num=2*nrefl+1)
    lenx=len(xpath)

    # define initial travel time and zpath related to these xpath values
    tt,zpath=calc_2way_time(nrefl,z0,a,b,v,vgrad,xpath)

    #-------------------------------------------------------------------------
    # now do the updating loop to find the optimum path for this src/rcv combi
    #-------------------------------------------------------------------------
    iter=0
    finish=0
    while finish < 1 and iter < maxiter:
        # calculate the derivative by perturbing each reflection x-point
        # note that grad[0] and grad[-1] remain zero, at src/rcv location
        xpathpert=np.copy(xpath)
        grad=np.zeros(lenx)
        # only perturb reflection points in the subsurface
        for k in range(1,lenx-1):
            xpathpert[k]=xpath[k]+dx
            ttpert,zpathpert=calc_2way_time(nrefl,z0,a,b,v,vgrad,xpathpert)
            grad[k]=ttpert-tt
            # restore perturbation for next step
            xpathpert[k]=xpath[k]

        # normalize the gradient and include minus sign
        # in this way the step size has a real meaning
        grad=-grad/np.sqrt(sum(grad*grad))
                
        # test with current step size and reduce step until smaller tt value
        ttnew=tt+1.0
        while ttnew > tt and step > dx:
            # define new xpath with current step size
            xpathnew=np.copy(xpath)
            for k in range(1,lenx-1):
                xpathnew[k]=xpath[k]+grad[k]*step

            # calculate the corresponding traveltime
            ttnew,zpathnew=calc_2way_time(nrefl,z0,a,b,v,vgrad,xpathnew)
            # reduce step size after each trial until smaller tt found
            step=step/2

        # if step is too small, we reached convergence
        # otherwise, increase step size for next iteration
        if step > dx:
            step=step*2
            tt=ttnew
            xpath=np.copy(xpathnew)
            zpath=np.copy(zpathnew)
            finish=0
        else:
            # no new path that has smaller tt, so keep current one
            finish=1
        iter=iter+1
        #print('tt=%f at iter=%d step=%f' %(tt,iter,step))
            
    # we return the traveltime and the path description
    return tt,xpath,zpath

def convolve2d(mat, sigma):
    kernel_size = int(2*sigma)
    kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
    kernel2D = np.outer(kernel1D.T, kernel1D.T)
    return cv2.filter2D(mat, -1, kernel2D)

#---------------------------------------------------------------
# function to create well-logs from layer definition
# zlayer describes the thicknes of each layer and vp/cs/rho
# the elastic properties in each layer.
# nz and dz is the desired regular output sampling of the log
# it will return the zlog, vplog, vslog, rholog with length nz
#---------------------------------------------------------------

def make_log(nz,dz,zlayer,vplayer,vslayer,rholayer):
    # create empty output logs
    zlog=np.arange(0.0,nz*dz,dz)
    vplog=np.zeros(nz)
    vslog=np.zeros(nz)
    rholog=np.zeros(nz)
    # initialize
    ilayer=0
    zcurrent=0
    nlayer=len(zlayer)
    # loop over depth levels
    for iz in range(nz):
        vplog[iz]=vplayer[ilayer]
        vslog[iz]=vslayer[ilayer]
        rholog[iz]=rholayer[ilayer]
        #print("iz=%d zcurrent=%g ilayer=%d" %(iz,zcurrent,ilayer))
        # check if next sample is in new layer
        # if last layer, continue this layer with last values
        if iz<nz-1 and ilayer<nlayer-1:
            znext=zcurrent+zlayer[ilayer]
            if (iz+1)*dz>=znext:
                zcurrent=znext
                ilayer=ilayer+1
    # return the logs
    return zlog,vplog,vslog,rholog


#------------------------------------------------------------------------
# Function for forward Taup transform (ChatGPT translated this from Matlab)
#------------------------------------------------------------------------

def taup(data_xt, xmid, dt, dx, pmin, pmax, np_out, method='linear', plot=False):
    
    from scipy.fft import fft, ifft
    from scipy.interpolate import interp1d

    """
    Transforms a shot record in the x-t domain to the tau-px domain.

    Parameters:
        data_xt : ndarray
            Shot record in the x-t domain, shape (nt, nx)
        xmid : float
            Offset of the leftmost trace from the zero offset trace
        dt : float
            Temporal sampling rate
        dx : float
            Spatial sampling rate
        pmin : float
            Minimum slowness value
        pmax : float
            Maximum slowness value
        np_out : int
            Number of slowness values to output
        method : str
            Interpolation method: 'linear', 'spline', etc.
        plot : bool
            Whether to plot intermediate results

    Returns:
        data_tp : ndarray
            Tau-p transformed data, shape (nt, np_out)
            
    Author original MAtlab code: Xander Staal, 2012
    Converted to Python by ChatGPT
    """
    nt, nx = data_xt.shape
    ixmid = round(xmid / dx)

    nf = nt // 2 + 1
    nx2 = 2 ** int(np.ceil(np.log2(nx)))

    # Plot original data
    if plot:
        xplot = np.arange(0, nx) * dx
        tplot = np.arange(0, nt) * dt
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(data_xt, extent=[xplot[0], xplot[-1], tplot[-1], tplot[0]], aspect='auto')
        plt.title('x-t domain')
        plt.xlabel('x')
        plt.ylabel('t')

    # Pad in x-direction
    data_xt_int = np.zeros((nt, nx2), dtype=complex)
    data_xt_int[:, :nx] = data_xt

    # Shift data to zero-offset trace
    data_xt_int = np.roll(data_xt_int, -ixmid, axis=1)

    # FFT in time (x-w domain)
    data_xw = fft(data_xt_int, axis=0)

    # IFFT in space (k-w domain)
    data_fk = ifft(data_xw[:nf, :], axis=1) * nx2

    # Shift zero spatial frequency to center
    data_fk = np.roll(data_fk, shift=nx2 // 2, axis=1)

    # Frequency and wavenumber vectors
    dom = 2 * np.pi / (nt * dt)
    omega = np.arange(nf) * dom

    dkx = 2 * np.pi / (nx2 * dx)
    kx = np.linspace(-np.pi/dx, np.pi/dx, nx2)

    if plot:
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(data_fk), extent=[kx[0], kx[-1], omega[-1]/(2*np.pi), omega[0]/(2*np.pi)], aspect='auto')
        plt.title('f-kx domain')
        plt.xlabel('k_x')
        plt.ylabel('f')

    # Interpolation to p-omega domain
    dp = (pmax - pmin) / (np_out - 1)
    p = np.linspace(pmin, pmax, np_out)

    data_fp = np.zeros((nf, np_out), dtype=complex)

    for ifr in range(1, nf):  # Start from 1 to skip DC component
        om = omega[ifr]
        kx_out = p * om

        # Interpolator with extrapolation filled as 0
        interp_func = interp1d(kx, data_fk[ifr, :], kind=method, bounds_error=False, fill_value=0)
        data_fp[ifr, :] = interp_func(kx_out)

    if plot:
        plt.subplot(2, 2, 4)
        plt.imshow(np.abs(data_fp), extent=[p[0], p[-1], omega[-1]/(2*np.pi), omega[0]/(2*np.pi)], aspect='auto')
        plt.title('f-px domain')
        plt.xlabel('p')
        plt.ylabel('f')

    # Inverse FFT to tau-p domain
    data_tp = np.zeros((nt, np_out), dtype=complex)
    data_tp[:nf, :] = data_fp
    data_tp[nf:, :] = np.conj(data_fp[nf-2:0:-1, :])
    data_tp = ifft(data_tp, axis=0)

    if plot:
        plt.subplot(2, 2, 3)
        plt.imshow(np.real(data_tp), extent=[p[0], p[-1], tplot[-1], tplot[0]], aspect='auto')
        plt.title(r'$\tau$-$p_x$ domain')
        plt.xlabel('p')
        plt.ylabel(r'$\tau$')
        plt.tight_layout()
        plt.show()

    return np.real(data_tp)


#-----------------------------------------------------------------------
# Define function to do LS subtraction of two panels
#-----------------------------------------------------------------------

def LSsub(ref,mul,lfilt,eps):
    
    # Input parameters:
    # ref  : 2D array with reference data
    # mul  : 2D array with predicted multiples
    # lfilt: length of the subtraction filter (odd)
    # eps  : relative stabilization factor
    # Return parameters"
    # out  : 2D array with subtraction result
    # filt : 1D array with estimated filter (lfilt length)
        
    # get size of data
    nx=ref.shape[0]
    nt=ref.shape[1]

    # define half filter length, lfilt should be odd
    lfilth=int(lfilt/2)
    if 2*lfilth+1 != lfilt:
        print('lfilt should be odd')
        exit
        
    # calculate inner product over rotated versions of mul with itself
    rauto=np.zeros(lfilt)
    for k in range(lfilt):
        roll=k
        mulroll=np.roll(mul,roll,axis=1)
        rauto[k]=np.sum(mul*mulroll)

    # calculate inner product over rotated versions of mul with ref
    rcorr=np.zeros(lfilt)
    for k in range(lfilt):
        roll=k-lfilth
        mulroll=np.roll(mul,roll,axis=1)
        rcorr[k]=np.sum(ref*mulroll)
        
    # create Toeplitz matrix and stabilize it
    R=toeplitz(rauto, rauto)
    R=R+eps*np.max(R)*np.eye(lfilt)        
    filt=np.linalg.inv(R)@rcorr

    # apply this filter as convolution
    con=0*ref
    for ix in range(nx):
        tmp=np.convolve(mul[ix,:],filt,mode='full')
        con[ix,:]=tmp[lfilth:nt+lfilth]

    # and do the subtraction
    out=ref-con
    
    # end of function; return the variable
    return out, filt; 