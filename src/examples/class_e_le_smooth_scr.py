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

"Smooth" tuning of a class E power amplifier (PA) with lumped elements.

"Smooth" means the operating mode in which:
1. No active device (AD) voltage gap occures when the AD switches into the
   conductive state.
2. The AD current equals zero when the AD switches into the conductive state.
There are no AD dynamical losses in this mode.

This script contains an example of setting the PA parameters, a PA tuning,
working with simulation data, and displaying plots of the PA time and frequency
characteristics.

You can try to change the PA parameters, the boolean flags below, the load
network (LN) type, etc. as you wish. For example, you can compare the
efficiency of the PA with different types of a LN.
"""

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
# Set a LN name (i. e. type). Available LN models:
# 'custom' - a completely user defined LN.
#     Not implemented in this example.
# 'zlaw' - a LN with a user defined z(w) law.
#     Not implemented in this example.
# 'zbar' - an almost ideal filter of the 1st voltage harmonic.
# 'sercir:le' - series resonant circuit with lumped elements.
# 'parcir:le' - parallel resonant circuit with lumped elements.
# 'pinet:le' - Pi network with lumped elements.
# 'teenet:le' - Tee network with lumped elements.
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

# Boolean flags that can be: True or False.
# Use an automatically generated initial guess for a "smooth" tuning.
# If it is used, then "b_cf_n" and "x_lf_n" values that are set here will not
# affect on the tuning result. Otherwise, these values will be considered as an
# initial guess.
auto_guess = True
# Do a "smooth" tuning of the PA.
do_tuning = True
# Show frequency characteristics of the PA.
show_freq_characs = False
# Show time characteristics of the PA.
show_time_characs = True

# -- Calculation of derived parameters ----------------------------------------

# Working angular frequency.
w_wrk = 2*np.pi*f_wrk
# LN input impedance at a LN tuning frequency.
z_out = r_norm*(1 + 1j*np.tan(phi_vi))
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
if load_name == 'custom':
    raise NotImplementedError
elif load_name == 'zlaw':
    raise NotImplementedError
elif load_name == 'zbar':
    load_cfg = {
        'name': load_name, 'highness': 9, 'f_cent': f_wrk, 'z_cent': z_out}
elif load_name == 'sercir:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out,
        'q_eqv': 2.0, 'g_cs': g_c, 'r_ls': r_l}
elif load_name == 'parcir:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out,
        'q_eqv': 2.0, 'g_cp': g_c, 'r_lp': r_l}
elif load_name == 'pinet:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out,
        'midcoef': 0.75, 'g_ci': g_c, 'g_co': g_c, 'r_lm': r_l,
        'r_out': r_norm}
elif load_name == 'teenet:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out,
        'midcoef': 0.75, 'g_cm': g_c, 'r_li': r_l, 'r_lo': r_l,
        'r_out': r_norm}
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

def announce_smooth(tunres):
    header = f"-- Class E tuning: 'smooth' --"
    print("\n", header, sep="")
    print(tunres)
    print("-"*len(header), "\n", sep="")

# "Smooth" tuning.
if do_tuning:
    tunres = pa.tune_smooth(auto_guess=auto_guess)
    announce_smooth(tunres)

# -- Simulation ---------------------------------------------------------------

# Get the PA parameters.
params = pa.get_params()

# Simulation.
simdata = pa.simulate()

# Some electrical characteristics.
# Voltage on the AD in frequency and time domains.
v_ad_f = simdata.get_charac(charac='v', probe='ad', domain='f')
v_ad_t = simdata.get_v_ad_t()

# Derived characteristics.
# Amplitudes of voltages and currents in the PA.
vi_amps = simdata.get_vi_amps()
# Magnitudes of voltages and currents in the PA.
vi_mags = simdata.get_vi_mags()
# The maximum energies stored in the PA's reactive elements.
cl_energs = simdata.get_cl_energs()
# The average power consumed from the voltage source "v_pwr".
p_pwr = simdata.get_p_pwr_avg()
# The output full power of the 1st harmonic.
s_out_h1 = simdata.get_s_out_h1()
# The average power dissipated in the AD.
p_ad = simdata.get_p_ad_avg()
# The average power consumed by the LN.
p_out_avg = simdata.get_p_out_avg()
# PA efficiency at the 1st harmonic frequency.
eff_h1 = simdata.get_eff_h1()
# Modified power output capability.
mpoc = simdata.get_mpoc()

# Observed phase difference between the 1st harmonics of output voltage and
# current.
def calc_phi_vi_out_h1():
    v_out_f = simdata.get_v_out_f()
    i_out_f = simdata.get_i_out_f()
    z_out_h1 = v_out_f[1]/i_out_f[1]
    return np.arctan(np.imag(z_out_h1)/np.real(z_out_h1))

phi_vi_observed = calc_phi_vi_out_h1()

# -- Plots --------------------------------------------------------------------

if show_freq_characs:   paplot.spectrum(simdata)
if show_time_characs:   paplot.waveform(simdata)
