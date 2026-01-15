import os
import numpy as np

# Load base-survey models

Vp  = np.load("Vp_syn.npy")
Vs  = np.load("Vs_syn.npy")
Rho = np.load("Rho_syn.npy")
Por = np.load("Porosity.npy")

# Saturation baseline
Snw = 0.0 * np.ones([nz_model, nx_model])  # initial CO2 saturation
Sw = 1 - Snw

# Rock and fluid properties
RhoS = 2500
CompS = 2.71e-11
K0 = 1 / CompS

RhoNw = 700
CompNw = 6.3e-10

RhoW = 1000
CompW = 3.6e-8

# Wavelets
wav_0 = np.load('wavelet_0.npy')
wav_1 = np.load('wavelet_1.npy')
wav_2 = np.load('wavelet_2.npy')
wav_3 = np.load('wavelet_3.npy')

wav = np.stack([wav_0, wav_1, wav_2, wav_3], axis=0)

# Angles
sin2 = [0.0, 0.1, 0.2, 0.3]

def substitution(Por, Snw2, RhoS, RhoW, RhoNw, Vp_S1, Vs_S1):
    Sw = 1 - Snw
    Sw2 = 1 - Snw2

    # State 1
    Rho_S1 = (1 - Por) * RhoS + Por * (Sw * RhoW + (1 - Sw) * RhoNw)
    K_S1 = Rho_S1 * (Vp_S1 ** 2 - (4 / 3) * Vs_S1 ** 2)
    CompFl = Sw * CompW + (1 - Sw) * CompNw
    KFl = 1 / CompFl

    # Dry Rock
    Rho_Dry = Rho_S1 - Por * (Sw * RhoW + (1 - Sw) * RhoNw)
    K_Dry = K0 * (K_S1 / (K0 - K_S1) - KFl / (Por * (K0 - KFl))) / (1 + (K_S1 / (K0 - K_S1) - KFl / (Por * (K0 - KFl))))
    Vp_Dry = np.sqrt((K_Dry + (4 / 3) * Rho_S1 * Vs_S1 ** 2) / Rho_Dry)
    Vs_Dry = Vs_S1 * np.sqrt(Rho_S1 / Rho_Dry)

    # State 2
    RhoNw2 = 700
    CompNw2 = 6.3e-10
    CompFl2 = Sw2 * CompW + (1 - Sw2) * CompNw2
    KFl2 = 1 / CompFl2

    Rho_S2 = Rho_Dry + Por * (Sw2 * RhoW + (1 - Sw2) * RhoNw2)
    K_S2 = K0 * (K_Dry / (K0 - K_Dry) + KFl2 / (Por * (K0 - KFl2))) / (1 + (K_Dry / (K0 - K_Dry) + KFl2 / (Por * (K0 - KFl2))))
    Vs_S2 = Vs_Dry * np.sqrt(Rho_Dry / Rho_S2)
    Vp_S2 = np.sqrt((K_S2 + (4 / 3) * Rho_S2 * Vs_S2 ** 2) / Rho_S2)

    return Vp_S2, Vs_S2, Rho_S2, Rho_S1

def compute_relative_contrasts_interface_loop(Vp, Vs, rho):
    nz, nx = Vp.shape

    dVp_over_Vp = np.zeros((nz - 1, nx))
    dVs_over_Vs = np.zeros((nz - 1, nx))
    drho_over_rho = np.zeros((nz - 1, nx))

    for i in range(nx):
        for j in range(nz - 1):
            vp1, vp2 = Vp[j, i], Vp[j + 1, i]
            vs1, vs2 = Vs[j, i], Vs[j + 1, i]
            rho1, rho2 = rho[j, i], rho[j + 1, i]

            vp_avg = 0.5 * (vp1 + vp2)
            vs_avg = 0.5 * (vs1 + vs2)
            rho_avg = 0.5 * (rho1 + rho2)

            dVp_over_Vp[j, i] = (vp2 - vp1) / vp_avg
            dVs_over_Vs[j, i] = (vs2 - vs1) / vs_avg
            drho_over_rho[j, i] = (rho2 - rho1) / rho_avg

    # Add top row of zeros
    dVp_over_Vp = np.vstack([np.zeros((1, nx)), dVp_over_Vp])
    dVs_over_Vs = np.vstack([np.zeros((1, nx)), dVs_over_Vs])
    drho_over_rho = np.vstack([np.zeros((1, nx)), drho_over_rho])

    return dVp_over_Vp, dVs_over_Vs, drho_over_rho

def compute_RR(A, B, sin2, wav):
    nz, nx = A.shape
    ntheta = len(sin2)

    image_fit = np.zeros((ntheta, nz, nx), dtype=A.dtype)

    for i, s2 in enumerate(sin2):
        R_i = A + B * s2
        wavelet = wav[i]
        for j in range(nx):
            image_fit[i, :, j] = np.convolve(R_i[:, j], wavelet, mode='same')
    return image_fit

# Create output folder
os.makedirs("output_rr", exist_ok=True)

# Loop over 1-100
for idx in range(0, 801):
    fname = f"saturation_data/saturation_data_{idx:03d}.npy"
    Snw2 = np.load(fname)

    Vp_S2, Vs_S2, Rho_S2, _ = substitution(Por, Snw2, RhoS, RhoW, RhoNw, Vp_model, Vs_model)

    Vp_syn = np.vstack((Vp_over, Vp_S2[::-1]))
    Vs_syn = np.vstack((Vs_over, Vs_S2[::-1]))
    Rho_syn = np.vstack((Rho_over, Rho_S2[::-1]))

    dVp_over_Vp, dVs_over_Vs, drho_over_rho = compute_relative_contrasts_interface_loop(Vp_syn, Vs_syn, Rho_syn)

    Vs2_over_Vp2 = (Vs_syn**2) / (Vp_syn**2)
    A = 0.5 * (dVp_over_Vp + drho_over_rho)
    B = dVp_over_Vp - 2 * Vs2_over_Vp2 * (drho_over_rho + 2 * dVs_over_Vs)

    RR = compute_RR(A, B, sin2, wav)

    # Split and save each angle separately
    for angle_idx in range(RR.shape[0]):
        output_fname = f"output_rr/RR_{idx:03d}_angle{angle_idx}.npy"
        np.save(output_fname, RR[angle_idx])

    print(f"Processed and saved slices for saturation_data_{idx:03d}")
