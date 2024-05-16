# -----------------------------------------------------------------------------
#
# This file is part of the PowAmp package.
# Copyright (C) 2022-2024 Igor Sivchek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------

"""
This file is a script that is made to run it directly in "Spyder" IDE.

The work of a class E power amplifier (PA) with lumped elements in a frequency
range.

It makes a "smooth" tuning of a PA at the lowest frequency in a range. Then, it
searches an input impedance value of an abstract (in non-programming meaning)
load network (LN) that provides the same level of the output power without the
dynamical lossies in the active device (AD) at each higher frequency.
The aquired dependency of the PA output impedance on frequency z_out(f) can be
used to find a LN that can provide the working conditions in this frequency
range. This part of the work is not covered here.
"""

import matplotlib.pyplot as plt
import numpy as np

from powamp import make_powamp  # To create a PA instance.
import powamp.plot as paplot    # To draw plots.
import powamp.utils as pautils  # To make denormalization of some parameters.


# Create a PA instance.
pa = make_powamp('class-e:le')

# -- Initial parameters -------------------------------------------------------

# About the normalization here.
# All the normalized impedances are normalized to the real part of a required
# input impedance value of a load network (LN) at a LN tuning frequency.

# Number of harmonics to calculate. The more harmonics are used, the more
# precise results will be, but the more time it will take.
h_len = 64
# Set an active device (AD) name (i. e. type). Available AD models:
# 'bidirect:switch' - equal bidirectional conductance in the conductive state.
# 'forward:switch' - only forward conductance in the conductive state.
# 'freewheel:switch' - forward conductance in the conductive state and
#     freewheeling diode which can conduct the current in the backward
#     direction.
ad_name = 'freewheel:switch'
# Only 'zbar' LN can be used to tune a PA in a frequency range.
# 'zbar' - an almost ideal filter of the 1st voltage harmonic.
load_name = 'zbar'
# A working frequency of the PA.
f_wrk = 4e6
# Duty cycle. This is the share of time period during which an AD is in the
# conductive state.
dc = 0.5
# Power supply voltage.
v_pwr = 20.0
# Normalizing resistance. The real part of the PA output impedance.
r_norm = 5.0
# Phase difference between the 1st harmonics of output voltage and current.
# In radians.
phi_vi = 0.0
# Normalized parasitic series resistance of inductors.
r_l_n = 1e-6
# Normalized parasitic parallel conductance of capacitors.
g_c_n = 1e-6
# Normalized resistance of the AD in the forward direction during the
# conductive state.
r_ad_n = 0.1
# Normalized resistance of the AD's freewheeiling diode in the diode's
# conductive direction.
# The parameter affects only when the AD is of 'freewheel:switch' type.
r_fwd_n = 0.2
# Normalized susceptance of the AD parasitic output capacitor "ca".
b_ca_h1n = 0.01
# Normalized susceptance of the bypass / DC-block capacitor "cb".
b_cb_h1n = 30
# Normalized reactance of the AC-block / RF-choke / DC-feed inductor "lb".
x_lb_h1n = 10
# Normalized susceptance of the forming subcircuit's capacitor "cf".
b_cf_h1n = 0.3
# Normalized reactance of the forming subcircuit's inductor "lf".
x_lf_h1n = 1.2

# A boolean flag that can be: True or False.
# Use an automatically generated initial guess for a "smooth" tuning.
# If it is used, then "b_cf_n" and "x_lf_n" values that are set here will not
# affect on the tuning result. Otherwise, these values will be considered as an
# initial guess.
auto_guess = True
# Do a "z_out" tuning of the PA at a certain frequency.
do_z_out_tuning = False
# Show frequency characteristics of the PA.
show_freq_characs = False
# Show time characteristics of the PA.
show_time_characs = False
# Show some frequency dependecies of the PA.
show_freq_deps = True

# -- Calculation of derived parameters ----------------------------------------

# Working angular frequency.
w_wrk = 2*np.pi*f_wrk
# LN input impedance at a LN tuning frequency.
z_out_h1 = r_norm*(1 + 1j*np.tan(phi_vi))
# AD resistance in the conductive state.
r_ad = r_ad_n*r_norm
# Resistance of the AD's freewheeling diode in the diode's conductive
# direction.
r_fwd = r_fwd_n*r_norm
# Parasitic conductance of the capacitors.
g_c = g_c_n/r_norm
# Parasitic resistance of the inductors.
r_l = r_l_n*r_norm
# Capacitance of the "ca" capacitor.
c_ca = pautils.c_from(b_n=b_ca_h1n, r=r_norm, w=w_wrk)
# Capacitance of the "cb" capacitor.
c_cb = pautils.c_from(b_n=b_cb_h1n, r=r_norm, w=w_wrk)
# Capacitance of the "cf" capacitor.
c_cf = pautils.c_from(b_n=b_cf_h1n, r=r_norm, w=w_wrk)
# Inductance of the "lb" inductor.
l_lb = pautils.l_from(x_n=x_lb_h1n, r=r_norm, w=w_wrk)
# Inductance of the "lf" inductor.
l_lf = pautils.l_from(x_n=x_lf_h1n, r=r_norm, w=w_wrk)

# Active device.
if ad_name == 'bidirect:switch':
    ad_cfg = {'name': ad_name, 'dc': dc, 'r_ad': r_ad}
elif ad_name == 'forward:switch':
    ad_cfg = {'name': ad_name, 'dc': dc, 'r_ad': r_ad}
elif ad_name == 'freewheel:switch':
    ad_cfg = {'name': ad_name, 'dc': dc, 'r_ad': r_ad, 'r_fwd': r_fwd}
else:
    raise ValueError("An unexpected AD name: {ad_name}".format(ad_name))

# Load network.
if load_name == 'zbar':
    load_cfg = {
        'name': load_name, 'highness': 9, 'f_cent': f_wrk, 'z_cent': z_out_h1}
else:
    raise ValueError("An unexpected load name: {load_name}".format(load_name))

# -- Setting the PA parameters ------------------------------------------------

# The method chaining technique is used.
pa \
    .set_hb_options(h_len=h_len, f_wrk=f_wrk) \
    .set_params(
        v_pwr=v_pwr, c_ca=c_ca, c_cb=c_cb, l_lb=l_lb, c_cf=c_cf, l_lf=l_lf) \
    .set_extra_resists(g_c=g_c, r_l=r_l) \
    .config_ad(**ad_cfg).config_load(**load_cfg)

# -- Tune the PA --------------------------------------------------------------

# Check that if a LN is tunable, it is tuned properly.
lonetcheck = pa.check_load()
if not lonetcheck.success:
    raise RuntimeError(
        "The load network is tunable, but was not tuned properly.\n" +
        lonetcheck.message)

def announce_freq_range(tunres):
    header = f"-- Class E tuning: frequency range --"
    print("\n", header, sep="")
    print(tunres)
    print("-"*len(header), "\n", sep="")

# Tuning in a frequency range.
tunres = pa.tune_freq_range(freqrat=1.5, pts=6, auto_guess=auto_guess)
announce_freq_range(tunres)

# Tuning at another frequency.
if do_z_out_tuning:
    # An extra high working frequency.
    f_extrahigh = 1.6*f_wrk
    # Set the PA parameters and initial estimation.
    pa.set_f_wrk(f_extrahigh) \
        .config_load(f_cent=f_extrahigh, z_cent=tunres.z_out[-1])
    # Use the same target level of the active output power.
    tunres_extrahigh = pa.tune_z_out(p_out_h1=np.real(tunres.s_out_h1[0]))

# -- Simulations and plots ----------------------------------------------------

# Check that the tuning at the lowest frequency in the range is successful.
if not tunres.success[0]:
    raise RuntimeError(
        "The tuning at the lowest frequency has failed.\n" +
        tunres.message[0])

# If a tuning in a frequency range was successful, the tuner will set the PA
# parameters that relates to the lowest frequency in the range.
# However, if (do_z_out_tuning == True), the parameters will be different.

# Set the PA paramters that relates to the lowest frequency in the range.
pa.set_f_wrk(tunres.f_wrk[0]) \
    .config_load(f_cent=tunres.f_wrk[0], z_cent=tunres.z_out_h1[0])

# Simulation at the lowest frequency in the range.
simdata_low = pa.simulate()

# Plots at the lowest frequency.
if show_freq_characs:   paplot.spectrum(simdata_low)
if show_time_characs:   paplot.waveform(simdata_low)

# Check that the tuning at the highest frequency in the range is successful.
if not tunres.success[-1]:
    raise RuntimeError(
        "The tuning at the highest frequency has failed.\n" +
        tunres.message[-1])

# Set the PA paramters that relates to the highest frequency in the range.
pa.set_f_wrk(tunres.f_wrk[-1]) \
    .config_load(f_cent=tunres.f_wrk[-1], z_cent=tunres.z_out_h1[-1])

# Simulation at the highest frequency in the range.
simdata_high = pa.simulate()

# Plots at the highest frequency.
if show_freq_characs:   paplot.spectrum(simdata_high)
if show_time_characs:   paplot.waveform(simdata_high)

# -- Plot frequency dependencies ----------------------------------------------

def draw_freq_deps(tunres):
    fig, (axz, axs, axe) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Operation of a class E PA with LE in a frequency range")
    freqs = np.array(tunres.f_wrk)
    axz.plot(freqs, np.real(tunres.z_out_h1), label="r_out")
    axz.plot(freqs, np.imag(tunres.z_out_h1), label="x_out")
    axs.plot(freqs, np.real(tunres.s_out_h1), label="p_out")
    axs.plot(freqs, np.imag(tunres.s_out_h1), label="q_out")
    axe.plot(freqs, tunres.eff_h1, label="eff")
    # Setting the view.
    # Impedance.
    axz.set_xlabel("f, Hz")
    axz.set_ylabel("z, Ohm", rotation='horizontal')
    axz.yaxis.set_label_coords(0.0, 1.05)
    axz.legend(title="Impedance", bbox_to_anchor=(1.16, 1), loc='upper right')
    axz.grid()
    # Power.
    axs.set_xlabel("f, Hz")
    axs.set_ylabel("s, W", rotation='horizontal')
    axs.yaxis.set_label_coords(0.0, 1.05)
    axs.legend(title="Power", bbox_to_anchor=(1.16, 1), loc='upper right')
    axs.grid()
    # Efficiency.
    axe.set_xlabel("f, Hz")
    axe.set_ylabel("eff", rotation='horizontal')
    axe.yaxis.set_label_coords(0.0, 1.05)
    axe.legend(title="Efficiency", bbox_to_anchor=(1.16, 1), loc='upper right')
    axe.grid()
    # Show it.
    fig.show()

if show_freq_deps:      draw_freq_deps(tunres)
