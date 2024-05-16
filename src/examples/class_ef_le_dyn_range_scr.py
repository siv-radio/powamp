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

The work of a class EF2 power amplifier (PA) with lumped elements in a wide
dynamic range.

It is well known that a class E PA with a small DC-feed inductance can operate
in a wide dynamic range of output power ["Switchmode RF and Microwave Power
Amplifiers", Andrei Grebennikov, Nathan O. Sokal, Marc J. Franco, 2nd ed.,
2012, p. 305]. It means that the PA can hold a high efficiency while its output
active power reduces several times from its peak level. It strongly depends on
the active device (AD) properties.
A class EF2 PA with a small DC-feed inductance also has this feature and this
script covers this subject.

/*remove it*/
I publicly mentioned about this feature in ["Outphasing modulation in a series
circuit with Chireix compensation in class EF2", I. V. Sivchek, "St. Petersburg
Polytechnical University Journal. Computer Science. Telecommunication and
Control Systems", Saint Petersburg, Russia, 2018, vol. 11, iss. 1, p. 7-17, in
Russian]. (I made an error in simulation settings there that led to a wrong
overall result, but I know how to fix it.)

"""

import matplotlib.pyplot as plt
import numpy as np

from powamp import make_powamp  # To create a PA instance.
import powamp.plot as paplot    # To draw plots.
import powamp.utils as pautils  # To make denormalization of some parameters.


# Create a PA instance.
pa = make_powamp('class-ef:le')

# -- Initial parameters -------------------------------------------------------

# About the normalization here.
# All the normalized impedances are normalized to the real part of a required
# input impedance value of a load network (LN) at a LN tuning frequency.

# Number of harmonics to calculate. The more harmonics are used, the more
# precise results will be, but the more time it will take. Recommendations:
# 32 - fast, 64 - more precise, 192 - precise (slow).
h_len = 32
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
dc = 0.35
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
b_cb_h1n = 10
# Normalized reactance of the AC-block / RF-choke / DC-feed inductor "lb".
# This value is relatively small here.
x_lb_h1n = 0.9*r_norm
# Normalized susceptance of the forming subcircuit's capacitor "cf".
b_cf_h1n = 0.24
# Normalized reactance of the forming subcircuit's inductor "lf".
x_lf_h1n = 2.5
# Normalized susceptance of the series resonant subcircuit's capacitor "cs".
b_cs_h1n = 0.21
# The voltage harmonic number on the AD that will be suppressed by the series
# resonant subcircuit.
h_sup = 2
# Use an automatically generated initial guess for a "maximize_mpoc" tuning.
# If it is used, then "b_cf_n", "x_lf_n", "b_cs_n", and "dc" values that are
# set here will not affect on the tuning result. Otherwise, these values will
# be considered as an initial guess. Available options:
# (True, 2), (True, 3) or False.
auto_guess = (True, h_sup)

# Boolean flags that can be: True or False.
# Show frequency characteristics of the PA.
show_freq_characs = False
# Show time characteristics of the PA.
show_time_characs = True
# Show some characteristics in a dynamic range.
show_dyn_range = True

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
# Capacitance of the "cs" capacitor.
c_cs = pautils.c_from(b_n=b_cs_h1n, r=r_norm, w=w_wrk)
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
        'name': load_name, 'highness': 9, 'f_cent': f_wrk, 'z_cent': z_out_h1}
elif load_name == 'sercir:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out_h1,
        'q_eqv': 2.0, 'g_cs': g_c, 'r_ls': r_l}
elif load_name == 'parcir:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out_h1,
        'q_eqv': 2.0, 'g_cp': g_c, 'r_lp': r_l}
elif load_name == 'pinet:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out_h1,
        'midcoef': 0.75, 'g_ci': g_c, 'g_co': g_c, 'r_lm': r_l,
        'r_out': r_norm}
elif load_name == 'teenet:le':
    load_cfg = {
        'name': load_name, 'f_tun': f_wrk, 'z_in_req': z_out_h1,
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
    .config_ad(**ad_cfg) \
    .config_load(**load_cfg)

# -- Tune the PA --------------------------------------------------------------

# Check that if a LN is tunable, it is tuned properly.
lonetcheck = pa.check_load()
if not lonetcheck.success:
    raise RuntimeError(
        "The load network is tunable, but was not tuned properly.\n" +
        lonetcheck.message)

def announce_sercir(tunres):
    header = f"-- Class EF{h_sup} tuning: suppressor --"
    print("\n", header, sep="")
    print(tunres)
    print("-"*len(header), "\n", sep="")

def announce_max_mpoc(tunres):
    header = f"-- Class EF{h_sup} tuning: MPOC maximization --"
    print("\n", header, sep="")
    print(tunres)
    print("-"*len(header), "\n", sep="")

def check_sercir(tunres):
    if not tunres.success:
        raise RuntimeError(
            "The series resonant subcircuit was not tuned properly.\n"
            "Cannot continue the tuning procedure.\n" +
            tunres.message)

if not auto_guess:
    tunres_sup = pa.tune_sercir(h_sup=h_sup, b_cs_h1n=b_cs_h1n)
    announce_sercir(tunres_sup)
    check_sercir(tunres_sup)
tunres_max_mpoc = pa.maximize_mpoc(auto_guess=auto_guess)
announce_max_mpoc(tunres_max_mpoc)

if not tunres_max_mpoc.success:
    raise RuntimeError(
        "The power amplifier was not tuned properly.\n"
        "Cannot continue the research.\n" +
        tunres_max_mpoc.message)

# Get the PA parameters.
params = pa.get_params()

# -- Simulations in a wide dynamic range --------------------------------------

# Interesting simulation data.
list_z_out_h1 = list()
list_p_pwr_avg = list()
list_s_out_h1 = list()
list_eff_h1 = list()

for coef in [0.05, 0.06, 0.07, 0.08, 0.09,
             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # Set a new input impedance of the LN.
    z_out_h1 = coef*r_norm + 0j
    if pa.get_load_name() == 'zbar':
        pa.config_load(z_cent=z_out_h1)
    else:
        pa.config_load(z_in_req=z_out_h1)
    # Simulate.
    simdata = pa.simulate()
    # The output impedance at the working frequency.
    list_z_out_h1.append(z_out_h1)
    # The average power consumed from the voltage source "v_pwr".
    list_p_pwr_avg.append(simdata.get_p_pwr_avg())
    # The output full power of the 1st harmonic.
    list_s_out_h1.append(simdata.get_s_out_h1())
    # PA efficiency at the 1st harmonic frequency.
    list_eff_h1.append(simdata.get_eff_h1())

# -- Plots --------------------------------------------------------------------

if show_freq_characs:   paplot.spectrum(simdata)
if show_time_characs:   paplot.waveform(simdata)

def draw_dyn_range(*, arr_r_out_h1, arr_p_pwr_avg, arr_p_out_h1, arr_eff_h1):
    fig, (axp, axe) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Operation of a class EF2 PA with LE in a dynamic range")
    axp.plot(arr_r_out_h1, arr_p_pwr_avg, label="p_pwr")
    axp.plot(arr_r_out_h1, arr_p_out_h1, label="p_out")
    axe.plot(arr_r_out_h1, arr_eff_h1, label="eff")
    # Setting the view.
    # Power.
    axp.set_xlabel("r, Ohm")
    axp.set_ylabel("p, W", rotation='horizontal')
    axp.yaxis.set_label_coords(0.0, 1.05)
    axp.legend(title="Power", bbox_to_anchor=(1.16, 1), loc='upper right')
    axp.grid()
    # Efficiency.
    axe.set_xlabel("r, Ohm")
    axe.set_ylabel("eff", rotation='horizontal')
    axe.yaxis.set_label_coords(0.0, 1.05)
    axe.legend(title="Efficiency", bbox_to_anchor=(1.16, 1), loc='upper right')
    axe.grid()
    # Show it.
    fig.show()

if show_dyn_range:
    draw_dyn_range(
        arr_r_out_h1=np.array(np.real(list_z_out_h1)),
        arr_p_pwr_avg=np.array(list_p_pwr_avg),
        arr_p_out_h1=np.array(np.real(list_s_out_h1)),
        arr_eff_h1=np.array(list_eff_h1))
