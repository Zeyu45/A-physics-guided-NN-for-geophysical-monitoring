import numpy as np
import math as m
import time


#-------------------------------------------------------------------------
# function to merge layers within the density/velocity log when they have 
# the same values (within eps difference) to speed up the modeling
# In fact, this is the reverse of function make_log
#-------------------------------------------------------------------------

def merge_layers(vp,vs,rho,dz,eps=0.001):
    _z=[0]
    _vp=[vp[0]]
    _vs=[vs[0]]
    _rho=[rho[0]]

    for iz in range(1,len(vp)):
        dvp=abs(vp[iz-1]-vp[iz])
        dvs=abs(vs[iz-1]-vs[iz])
        drho=abs(rho[iz-1]-rho[iz])
        if (dvp<eps and dvs<eps and drho<eps):
            pass
        else:
            _z.append(iz*dz)
            _vp.append(vp[iz])
            _vs.append(vs[iz])
            _rho.append(rho[iz])
    return _z,_vp,_vs,_rho
    
#------------------------------------------------------------------------
# Class with Kennett solver for the elastic case
#
# Initialize the class with:
# vp  - 1D array of P-wave velocity values with depth steps dz
# vs  - 1D array of S-wave velocity values with depth steps dz
# rho - 1D array of density values with depth steps dz
# dz  - depth step in the above functions
#
# Use "solve" to get the acoustic reflection response:
# nt      - number of time samples in the response
# nx      - number of traces in the response
# dt      - time sampling interval
# dx      - spatial sampling interval
# surfmul - if >0: include surface multiples
# intmul  - if >0: include internal multiples
# alpha   - exponential factor to suppress wrap aroudn in time (alpha<0)
# wavelet - source signal in time with length nt
# ntrot   - optional time rotation before applying the exponential factor
#           (this can be useful if the wavelet has a shifted peak)
#------------------------------------------------------------------------

class KennettSolver:
    def __init__(self,vp,vs,rho,dz):
        z,vp,vs,rho=merge_layers(vp,vs,rho,dz)
        self.z=z
        self.vp=vp
        self.vs=vs
        self.rho=rho

    def solve(self,nt=512,nx=100,dt=0.004,dx=10,surfmul=0,intmul=0,alpha=-1.0,wavelet=None,ntrot=0):
        # debug option to display reflecitivty operators
        debug=0
        print('Start Kennett modeling')
        # define parameters
        nf = nt
        df = 1 / (dt * nt)
        dkx = 1 / (dx * nx)

        # initialization
        Rpp = np.zeros((nt, nx))
        Rsp = np.zeros((nt, nx))
        Rps = np.zeros((nt, nx))
        Rss = np.zeros((nt, nx))

        Rpp0 = np.zeros((nt, nx))
        Rsp0 = np.zeros((nt, nx))
        Rps0 = np.zeros((nt, nx))
        Rss0 = np.zeros((nt, nx))

        dz = np.diff(self.z)

        w = (1+0j)*np.hstack((np.arange(0, (nt / 2)), np.arange(-(nt / 2), 0))) * 2 * m.pi*df
        w = w.reshape((-1,1))
        w += 1j*alpha

        kx = np.hstack((np.arange(0, (nx / 2)), np.arange(-(nx / 2), 0))) * 2 * m.pi * dkx
        kx2 = kx**2

        # initialize timings
        tkxax=0.0
        toper=0.0
        tcalc=0.0

        # loop over all depth levels, from bottom to surface
        nz=len(self.z)
        for iz in range(nz - 1, 0, -1):
            Rpp0=Rpp
            Rsp0=Rsp
            Rps0=Rps
            Rss0=Rss

            # get velocities
            cpup = self.vp[iz - 1]
            csup = self.vs[iz - 1]
            cplow = self.vp[iz]
            cslow = self.vs[iz]
            
            t0=time.time()
                    
            # define kz(kx,w) lower layer for P-waves (copy from previous layer)
            if iz == nz-1:
                k = w / cplow
                kplow2 = (k ** 2) * np.ones((1, nx))
                kzlow = np.conj(np.sqrt(kplow2 - kx ** 2))
                kzlow[int(nf / 2) + 1:, ...] = -kzlow[int(nf / 2) + 1:, ...]
                kzplow = np.real(kzlow) - 1j * np.abs(np.imag(kzlow))
            else:
                # due to recursion, this will be taken from previous depth
                kplow2=kpup2
                kzplow=kzpup
                
            # define kz(kx,w) lower layer for S-waves (copy from previous layer)
            if iz == nz-1:
                k = w / cslow
                kslow2 = (k ** 2) * np.ones((1, nx))
                kzlow = np.conj(np.sqrt(kslow2 - kx ** 2))
                kzlow[int(nf / 2) + 1:, ...] = -kzlow[int(nf / 2) + 1:, ...]
                kzslow = np.real(kzlow) - 1j * np.abs(np.imag(kzlow))
            else:
                # due to recursion, this will be taken from previous depth
                kslow2=ksup2
                kzslow=kzsup
            
            # define kz(kx,w) upper layer for P-waves
            k = w / cpup
            kpup2 = (k**2) * np.ones((1, nx))
            kzupp = np.conj(np.sqrt(kpup2 - kx**2))
            kzupp[int(nf/2)+1:,...]=-kzupp[int(nf/2)+1:,...]
            kzpup = np.real(kzupp) - 1j * np.abs(np.imag(kzupp))

            # define kz(kx,w) upper layer for S-waves
            k = w / csup
            ksup2 = (k**2) * np.ones((1, nx))
            kzupp = np.conj(np.sqrt(ksup2 - kx**2))
            kzupp[int(nf/2)+1:,...]=-kzupp[int(nf/2)+1:,...]
            kzsup = np.real(kzupp) - 1j * np.abs(np.imag(kzupp))

            tkxax+=time.time()-t0
            t0=time.time()

            # get densities
            rhoup=self.rho[iz-1]
            rholow=self.rho[iz]

            # define intermediate variables, stabilize inversion for w=0
            a = rhoup * (1 - 2 * kx2 / ksup2) - rholow * (1 - 2 * kx2 / kslow2)
            b = rhoup * (1 - 2 * kx2 / ksup2) + 2 * rholow * kx2 / kslow2
            c = rholow * (1 - 2 * kx2 / kslow2) + 2 * rhoup * kx2 / ksup2
            d = 2 * (rhoup / ksup2 - rholow / kslow2)
            a[0,...]=0.0
            b[0,...]=0.0
            c[0,...]=0.0
            d[0,...]=0.0
            e = kzplow * b + kzpup * c
            f = kzslow * b + kzsup * c
            g = a - d * kzplow * kzsup
            h = a - d * kzpup * kzslow

            # determinant and its stabilized inverse
            det = e * f + g * h * kx2
            detinv = 0.0 * det
            detinv[abs(det) > 500] = 1 / det[abs(det) > 500]

            # create layer reflectivity and transmission operators
            rdpp = ((-kzplow * b + kzpup * c) * f - (a + kzpup * kzslow * d) * g * kx2) * detinv
            rdsp = 2 * kzpup * cpup / csup * kx * (a * c + b * d * kzplow * kzslow) * detinv
            rdps = 2 * kzsup * csup / cpup * kx * (a * c + b * d * kzslow * kzplow) * detinv
            rdss = -((-kzslow * b + kzsup * c) * e - kx2 * (kzsup * kzplow * d + a) * h) * detinv

            rupp = ((-kzpup * c + kzplow * b) * f - kx2 * (a + kzsup * kzplow * d) * h) * detinv
            rusp = -2 * kzplow * cplow / cslow * kx * (a * b + c * d * kzsup * kzpup) * detinv
            rups = -2 * kzslow * cslow / cplow * kx * (a * b + c * d * kzsup * kzpup) * detinv
            russ = -((kzslow * b - kzsup * c) * e - kx2 * (a + kzslow * kzpup * d) * h) * detinv

            tdpp = 2 * rhoup * cpup / cplow * kzpup * f * detinv
            tdsp = -2 * rhoup * cpup / cslow * kx * kzpup * g * detinv
            tdps = 2 * rhoup * csup / cplow * kx * kzsup * h * detinv
            tdss = 2 * rhoup * csup / cslow * kzsup * e * detinv

            tupp = 2 * rholow * cplow / cpup * kzplow * f * detinv
            tusp = 2 * rholow * cplow / csup * kx * kzplow * h * detinv
            tups = -2 * rholow * cslow / cpup * kx * kzslow * g * detinv
            tuss = 2 * rholow * cslow / csup * kzslow * e * detinv

            toper+=time.time()-t0
            t0=time.time()

            # for debugging we can plot some reflectivity functions
            if debug>0:
                plt.figure();plt.imshow(np.real(det[0:257,:]),cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(detinv[0:257,:]),cmap='bwr');plt.colorbar();plt.show()
                cc=5
                plt.figure();plt.imshow(np.real(rupp[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(rups[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(rusp[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(russ[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()

            if iz==nz-1:
                Rpp = rdpp
                Rps = rdps
                Rsp = rdsp
                Rss = rdss
            else:
                wpoper = np.exp(-1j * kzplow * abs(dz[iz]))
                wsoper = np.exp(-1j * kzslow * abs(dz[iz]))

                if intmul:
                    Wru_11 = wpoper * rupp
                    Wru_12 = wpoper * rups
                    Wru_21 = wsoper * rusp
                    Wru_22 = wsoper * russ

                    WRd_11 = wpoper * Rpp0
                    WRd_12 = wpoper * Rps0
                    WRd_21 = wsoper * Rsp0
                    WRd_22 = wsoper * Rss0

                    Pre_inv_11 = 1 - Wru_11 * WRd_11 - Wru_12 * WRd_21
                    Pre_inv_12 = - Wru_11 * WRd_12 - Wru_12 * WRd_22
                    Pre_inv_21 = - Wru_21 * WRd_11 - Wru_22 * WRd_21
                    Pre_inv_22 = 1 - Wru_21 * WRd_12 - Wru_22 * WRd_22

                    det_inv = Pre_inv_11 * Pre_inv_22 - Pre_inv_21 * Pre_inv_12
                    det_inv[abs(det_inv)<=0.0001]=m.inf

                    factor_11 = Pre_inv_22 / det_inv
                    factor_21 = -Pre_inv_21 / det_inv
                    factor_12 = -Pre_inv_12 / det_inv
                    factor_22 = Pre_inv_11 / det_inv
                else:
                    factor_11=1
                    factor_21=0
                    factor_12=0
                    factor_22=1

                FWt_11 = factor_11 * wpoper * tdpp + factor_12 * wsoper * tdsp
                FWt_12 = factor_11 * wpoper * tdps + factor_12 * wsoper * tdss
                FWt_21 = factor_21 * wpoper * tdpp + factor_22 * wsoper * tdsp
                FWt_22 = factor_21 * wpoper * tdps + factor_22 * wsoper * tdss

                tWR_11 = tupp * wpoper * Rpp0 + tups * wsoper * Rsp0
                tWR_12 = tupp * wpoper * Rps0 + tups * wsoper * Rss0
                tWR_21 = tusp * wpoper * Rpp0 + tuss * wsoper * Rsp0
                tWR_22 = tusp * wpoper * Rps0 + tuss * wsoper * Rss0

                Rpp = rdpp + tWR_11 * FWt_11 + tWR_12 * FWt_21
                Rps = rdps + tWR_11 * FWt_12 + tWR_12 * FWt_22
                Rsp = rdsp + tWR_21 * FWt_11 + tWR_22 * FWt_21
                Rss = rdss + tWR_21 * FWt_12 + tWR_22 * FWt_22

            tcalc+=time.time()-t0

        t0=time.time()

        # Add final propagation operator towards the surface
        # define kz(kx,w) upper layer for P-waves (copy from iz=1)
        kplow2=kpup2
        kzplow=kzpup

        # define kz(kx,w) upper layer for P-waves
        kslow2=ksup2
        kzslow=kzsup

        wpoper = np.exp(-1j * kzplow * abs(dz[0]))
        wsoper = np.exp(-1j * kzslow * abs(dz[0]))

        # round-trip through the upper layer
        Rpp = wpoper * Rpp * wpoper
        Rps = wpoper * Rps * wsoper
        Rsp = wsoper * Rsp * wpoper
        Rss = wsoper * Rss * wsoper

        # add surface multiples in this last step
        if surfmul:
            # get velocities
            cplow = self.vp[0]
            cslow = self.vs[0]
            cpup = cplow/10.0
            csup = cslow/10.0

            # define kz(kx,w) upper layer for P-waves
            k = w / cpup
            kpup2 = (k**2) * np.ones((1, nx))
            kzupp = np.conj(np.sqrt(kpup2 - kx**2))
            kzupp[int(nf/2)+1:,...]=-kzupp[int(nf/2)+1:,...]
            kzpup = np.real(kzupp) - 1j * np.abs(np.imag(kzupp))

            # define kz(kx,w) upper layer for S-waves
            k = w / csup
            ksup2 = (k**2) * np.ones((1, nx))
            kzupp = np.conj(np.sqrt(ksup2 - kx**2))
            kzupp[int(nf/2)+1:,...]=-kzupp[int(nf/2)+1:,...]
            kzsup = np.real(kzupp) - 1j * np.abs(np.imag(kzupp))

            # get densities
            rhoup=0.0
            rholow=self.rho[0]

            # define intermediate variables, stabilize inversion for w=0
            a = rhoup * (1 - 2 * kx2 / ksup2) - rholow * (1 - 2 * kx2 / kslow2)
            b = rhoup * (1 - 2 * kx2 / ksup2) + 2 * rholow * kx2 / kslow2
            c = rholow * (1 - 2 * kx2 / kslow2) + 2 * rhoup * kx2 / ksup2
            d = 2 * (rhoup / ksup2 - rholow / kslow2)
            a[0,...]=0.0
            b[0,...]=0.0
            c[0,...]=0.0
            d[0,...]=0.0
            e = kzplow * b + kzpup * c
            f = kzslow * b + kzsup * c
            g = a - d * kzplow * kzsup
            h = a - d * kzpup * kzslow

            # determinant and its stabilized inverse
            det = e * f + g * h * kx2
            detinv = 0.0 * det
            detinv[abs(det) > 500] = 1 / det[abs(det) > 500]

            # we only need the rup for the surface multiples
            rupp = ((-kzpup * c + kzplow * b) * f - kx2 * (a + kzsup * kzplow * d) * h) * detinv
            rusp = -2 * kzplow * cplow / cslow * kx * (a * b + c * d * kzsup * kzpup) * detinv
            rups = -2 * kzslow * cslow / cplow * kx * (a * b + c * d * kzsup * kzpup) * detinv
            russ = -((kzslow * b - kzsup * c) * e - kx2 * (a + kzslow * kzpup * d) * h) * detinv

            # for debugging we can plot some reflectivity functions
            if debug>0:
                plt.figure();plt.imshow(np.real(det[0:257,:]),cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(detinv[0:257,:]),cmap='bwr');plt.colorbar();plt.show()
                cc=5
                plt.figure();plt.imshow(np.real(rupp[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(rups[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(rusp[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()
                plt.figure();plt.imshow(np.real(russ[0:257,:]),vmin=-cc,vmax=cc,cmap='bwr');plt.colorbar();plt.show()

            # calculate denominator: (I - Rup*R)
            Pre_inv_11 = 1 - rupp * Rpp - rups * Rsp
            Pre_inv_12 = - rupp * Rps - rups * Rss
            Pre_inv_21 = - rusp * Rpp - russ * Rsp
            Pre_inv_22 = 1 - rusp * Rps - russ * Rss

            det_inv = Pre_inv_11 * Pre_inv_22 - Pre_inv_21 * Pre_inv_12
            det_inv[abs(det_inv)<=0.0001]=m.inf

            # inverse of this denominator: I/(I-Rup*R)
            factor_11 = Pre_inv_22 / det_inv
            factor_21 = -Pre_inv_21 / det_inv
            factor_12 = -Pre_inv_12 / det_inv
            factor_22 = Pre_inv_11 / det_inv
            
            # copy primary response R0 at z=0
            Rpp0 = Rpp
            Rps0 = Rps
            Rsp0 = Rsp
            Rss0 = Rss
            
            # add surface multiples: R=R0/(1-Rup*R0)
            Rpp = factor_11 * Rpp0 + factor_12 * Rsp0
            Rps = factor_11 * Rps0 + factor_12 * Rss0
            Rsp = factor_21 * Rpp0 + factor_22 * Rsp0
            Rss = factor_21 * Rps0 + factor_22 * Rss0

        # prepare for inverse exponential factor
        time_ax = dt*np.arange(0, nt).reshape((-1,1))

        # if desired, include time wavelet in the Fourier domain
        if wavelet is not None:
            if len(wavelet) != nt:
                print("Length of wavelet %d is not same as nt=%d" %(len(wavelet),nt))
            wav_w = np.fft.fft(wavelet).reshape((-1, 1))
            Rpp *= wav_w
            Rsp *= wav_w
            Rps *= wav_w
            Rss *= wav_w

        # transform back to the x - t domain, optionally apply time shift 
        # and correct for exponential factor
        pp = np.exp(-alpha * time_ax) * np.roll(np.real(np.fft.ifft2(Rpp)),ntrot,axis=0)
        sp = np.exp(-alpha * time_ax) * np.roll(np.real(np.fft.ifft2(Rsp)),ntrot,axis=0)
        ps = np.exp(-alpha * time_ax) * np.roll(np.real(np.fft.ifft2(Rps)),ntrot,axis=0)
        ss = np.exp(-alpha * time_ax) * np.roll(np.real(np.fft.ifft2(Rss)),ntrot,axis=0)

        tend=time.time()-t0

        print('tkxax=%g toper=%g tcalc=%g tend=%g' %(tkxax,toper,tcalc,tend))
        return pp,sp,ps,ss
    
#------------------------------------------------------------------------
# End of class with Kennett solver for the elastic case
#------------------------------------------------------------------------
