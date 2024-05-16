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
A model of a class EFx power amplifier with lumped elements.
"""

from collections import Sequence
from collections import namedtuple

import numpy as np
import scipy.optimize as scipyopt

from . import optim as pao
from . import utils as pau
from .circprim import CurrGenLE, FormCirLE, SerCirLE
from .one_ad import OneAD
from .scopeguard import ScopeGuardManual
from .simdata import SimDataBase, add_charac_getters


class Model(OneAD, CurrGenLE, FormCirLE, SerCirLE):
    """
    A model of a class EFx power amplifier (PA) with lumped elements.

    Equivalent circuit
    ------------------

     Independent |                     Passive                     | Active
     voltage     |                     linear                      | device
     source      |                     subcircuit                  |

     (2)          i_lb           (1)                         i_lin    i_ad
      ------------->--l_lb--r_lb--o------------------------o---<------->---
      |                           |                        |              |
      |                           v i_cb                   |              |
      |                           |                        |              |
      |                        ---o---                     v i_ca         |
      |                        |     |                     |              |
      v i_pwr                 c_cb  g_cb   ---g_cs---      o------        |
      |        (5)             | (4) |     |        |      |     |        |
    v_pwr       ---l_lf--r_lf--o--o--o-----o--c_cs--o(3)  c_ca  g_ca      ad
      |         |                 |                 |      |     |        |
      |         v i_lf/i_out      v i_cf  i_cs/i_ls v      o------        |
      |         |                 |                 |      |              |
      |       z_out            ---o---             l_ls    |              |
      |         |              |     |              |      |              |
      |         |             c_cf  g_cf           r_ls    |              |
      |         |              |     |              |      |              |
      ----------o--------------o-----o--------------o------o---------------
              (gnd)

            Types of the active device (AD):
    AD w. equal      AD w. forward    AD w. forward
    bidirectional    conduction       conduction & an
    conduction                        anti-parallel diode
    ("bidirect:      ("forward:       ("freewheel:
      switch")         switch")         switch")

       ----            ----            ----
          |               |               |
          |               |               |
          |               |               |
          |               |               |
          |               |               o---------
          |               |               |        |
          |               ad              ad      ---
          ad              |               |       /d\
          |               v               v       ---
          |               |               |        |
          |               |               o---------
          |               |               |
          |               |               |
          |               |               |
          |               |               |
       ----            ----            ----

    AD laws
    ------------------------------------------------------
    | device | bidirect: | forward:      | freewheel:    |
    |        | switch    | switch        | switch        |
    ------------------------------------------------------
    | ad     | r_ad(t)   | r_ad(v_ad, t) | r_ad(v_ad, t) |
    | d      | -         | -             | r_fwd(v_ad)   |
    ------------------------------------------------------

    Some notations:
    AD / ad - active device.
    d - freewheeling diode.
    r_ad(t) / r_ad(v_ad, t) - AD resistance law. A non-zero value in
        conductive state is useful by itself and helps to avoid convergence
        problems.
    r_fwd(v_ad) - resistance law of a freewheeling diode in its forward
        conductive state.
    v_pwr - power supply voltage.
    i_pwr - power supply current.
    i_lin - linear subcircuit current.
    v_ad - voltage on the AD.
    i_ad - current through the AD.
    t - time.

    Parameters
    ----------
    The PA parameters can be collected into a "Params" "namedtuple" object.
    It contains following fields:
    name : str
        The name of the power amplifier.
    ad : namedtuple
        A set of paramters of an active device.
    load : namedtuple
        A set of paramters of a load network.
    v_pwr : float
        Power supply constant voltage.
    c_ca : float
        Adjacent / anode-cathode / active device / parasitic output
        capacitance of the AD. Linear assumption.
    g_ca : float
        A small shunt conductance.
    c_cb : float
        A bypass / DC-block capacitance.
    g_cb : float
        A small shunt conductance to avoid convergence problems.
    l_lb : float
        AC-block / RF-choke / DC-feed inductance.
    r_lb : float
        A small resistance to avoid convergence problems.
    c_cf : float
        A forming subcircuit's capacitance.
    g_cf : float
        A small shunt conductance.
    l_lf : float
        A forming subcircuit's inductance.
    r_lf : float
        A small resistance.
    c_cs : float
        A series resonant subcircuit's capacitance.
    g_cs : float
        A small shunt conductance.
    l_ls :
        A series resonant subcircuit's inductance.
    r_ls : float
        A small resistance.
    z_out : complex
        An output impedance. It is an input impedance value of a load
        network at a certain frequency.

    Characteristics
    ---------------
    Electrical characteristics of the PA in time and frequency domains are
    stored in a "SimData" object.
    The list of characteristics:
    v_ad     voltage on the AD.
    i_ad     current through the AD.
    v_pwr    voltage of power supply "v_pwr".
    i_pwr    current through the voltage source "v_pwr"
             (equal to negative "i_lb").
    v_ca     voltage on the AD anode-cathode capacitor "ca" (equal to "v_ad").
    i_ca     current through the AD anode-cathode capacitor "ca".
    v_lb     voltage on the DC-feed inductor "lb".
    i_lb     current through the DC-feed inductor "lb".
    v_cb     voltage on the DC current blocking capacitor "cb".
    i_cb     current through the DC current blocking capacitor "cb".
    v_cf     voltage on the forming subcircuit's capacitor "cf".
    i_cf     current through the forming subcircuit's capacitor "cf".
    v_lf     voltage on the forming subcircuit's inductor "lf".
    i_lf     current through the forming subcircuit's inductor "lf"
             (equal to "i_out").
    v_cs     voltage on the series resonant subcircuit's capacitor "cs".
    i_cs     current through the series resonant subcircuit's capacitor "cs".
    v_ls     voltage on the series resonant subcircuit's inductor "ls".
    i_ls     current through the series resonant subcircuit's inductor "ls"
             (equal to "i_cs").
    v_out    output voltage.
    i_out    output current (equal to "i_lf").
    All these capacitors and inductors contain resistors which are taken into
    account during the calculation of related voltages and currents.

    Notes
    -----
    The AD has only quasistatic losses (i. e. in forward and/or backward
    conductive states).
    No sources of dynamic losses (i. e. at switching moments) has been
    implemented ("ca" capacitor is an external component).

    The AD of "bidirect:switch" type is linear, while "forward:switch"
    and "freewheel:switch" are not. Hence "bidirect:switch" type does not
    require an iterative algorithm to find the required AD voltage that brings
    convergance.

    If "c_cb" value is high enough, then "c_cf" and "c_ca" have almost parallel
    connection, and their values can be summarized. This is the reason why
    "c_cf" and "c_ca" are not distinguished in many scientific works.
    If a "c_ca" value is high enough, it can be used as "c_cf". In this case,
    "c_cf" can be disabled (reduced to a negligible value). This is the reason
    why there is no additional "c_cf" in many scientific works when a
    particular AD with its own "c_ca" is used.
    "c_cf" is useful when there is a necessity in additional capacitance to
    work in class EFx. It is better to add this capacitance there because the
    voltage on "c_cf" is significantly lower than on "c_ca".
    """
    # This "Model" class and its "Tuner" heir have many common with the class E
    # PA model. Therefore there can be an itch for making a code refactoring to
    # avoid the code duplication. However, during the development process,
    # there were many times when the similarities between the two models were
    # broken. Due to that fact, the models remain separated and flexible for
    # further development. 2024.02.09.

    _name = None  # Settable from the outside.

    Probes = namedtuple('Probes', 'pwr ad ca cb lb cf lf cs ls out')

    Params = namedtuple(
        'Params',
        'name ad load \
         v_pwr c_ca g_ca c_cb g_cb l_lb r_lb c_cf g_cf l_lf r_lf \
         c_cs g_cs l_ls r_ls z_out_h1')

    # Public methods.

    def __init__(self):
        """
        A constructor of a power amplifier (PA) model object.
        It presets parameters of a tuned PA.
        """
        # Call the constructors of the base classes.
        OneAD.__init__(self)
        CurrGenLE.__init__(self)
        FormCirLE.__init__(self)
        SerCirLE.__init__(self)
        # The default parameters are obtained with (h_len = 192).
        # Basic model parameters.
        self.set_h_len(64).set_t_wrk(2*np.pi)
        # Current generator.
        self.set_v_pwr(1.0).set_c_ca(0.01).set_c_cb(10.0).set_l_lb(30.0)
        # Forming subcircuit.
        self.set_c_cf(0.155).set_l_lf(2.42)
        # Series resonant subcircuit.
        self.set_c_cs(0.1252).set_l_ls(1.996)
        # Set default g_ca, g_cb, r_lb, g_cf, r_lf, g_cs, r_ls values.
        self.set_extra_resists(g_c=1e-6, r_l=1e-6)
        # Active device.
        self.config_ad(name='bidirect:switch', dc=0.352, r_ad=0.1)
        # Load network.
        self.config_load(name='zbar', highness=9,
                         z_cent=1.0, f_cent=self.get_f_wrk())
        # The end of the "Model" constructor.

    def set_extra_resists(self, *, g_c, r_l):
        """
        Set additional resistances of capacitors and inductors to achieve
        better convergence.

        Parameters
        ----------
        g_c : float
            A parallel conductance of a capacitor.
            Usually it is a small number at a working frequency.
        r_l : float
            A series resistance of an inductor.
            Usually it is a small number at a working frequency.

        Returns
        -------
        "self" reference to the caller object.

        Notes
        -----
        This method is made for user's convenience. If there is no needness to
        take into account specific losses in reactive components, one can set
        all the resistances at once using this method. As an alternative, it is
        possible to set each resistance separately.
        """
        self.set_g_ca(g_c)
        self.set_g_cb(g_c);    self.set_r_lb(r_l)
        self.set_g_cf(g_c);    self.set_r_lf(r_l)
        self.set_g_cs(g_c);    self.set_r_ls(r_l)
        return self

    def get_h_sup(self):
        """
        Get a voltage harmonic number that is suppressed by the series resonant
        subcircuit.
        h_sup = w_sup/w_wrk, where
        w_sup - suppressed angular frequency,
        w_wrk - working angular frequency.

        Returns
        -------
        h_sup : float
            A suppressed voltage harmonic number.
            If it cannot calculate "w_sup", then "h_sup" equals "0.0".

        Notes
        -----
        It is better to always check if a tuning of the series resonant
        subcircuit has been successful before a usage of this method.
        """
        w_sup = self._find_w_sup().w_sup
        return w_sup/self.get_w_wrk()

    def set_params(self, *,
                   v_pwr=None,
                   c_ca=None, g_ca=None,
                   c_cb=None, g_cb=None, l_lb=None, r_lb=None,
                   c_cf=None, g_cf=None, l_lf=None, r_lf=None,
                   c_cs=None, g_cs=None, l_ls=None, r_ls=None):
        """
        Set parameters of a power amplifier (PA) model.
        To understand the meaning of the parameters, see the model description.
        """
        # Current generator.
        if v_pwr is not None:           self.set_v_pwr(v_pwr)
        if c_ca is not None:            self.set_c_ca(c_ca)
        if g_ca is not None:            self.set_g_ca(g_ca)
        if c_cb is not None:            self.set_c_cb(c_cb)
        if g_cb is not None:            self.set_g_cb(g_cb)
        if l_lb is not None:            self.set_l_lb(l_lb)
        if r_lb is not None:            self.set_r_lb(r_lb)
        # Forming subcircuit.
        if c_cf is not None:            self.set_c_cf(c_cf)
        if g_cf is not None:            self.set_g_cf(g_cf)
        if l_lf is not None:            self.set_l_lf(l_lf)
        if r_lf is not None:            self.set_r_lf(r_lf)
        # Series resonant subcircuit.
        if c_cs is not None:            self.set_c_cs(c_cs)
        if g_cs is not None:            self.set_g_cs(g_cs)
        if l_ls is not None:            self.set_l_ls(l_ls)
        if r_ls is not None:            self.set_r_ls(r_ls)
        return self

    def get_params(self):
        """
        Get all the parameters of a power amplifier (PA) instance.

        Returns
        -------
        result : Params
            A "namedtuple" that contains parameters of the PA. To understand
            the meaning of the parameters, see the model description.
        """
        return self.Params(
            name='class-ef:le',
            ad=self.get_ad_config(), load=self.get_load_config(),
            v_pwr=self.get_v_pwr(),
            c_ca=self.get_c_ca(), g_ca=self.get_g_ca(),
            c_cb=self.get_c_cb(), g_cb=self.get_g_cb(),
            l_lb=self.get_l_lb(), r_lb=self.get_r_lb(),
            c_cf=self.get_c_cf(), g_cf=self.get_g_cf(),
            l_lf=self.get_l_lf(), r_lf=self.get_r_lf(),
            c_cs=self.get_c_cs(), g_cs=self.get_g_cs(),
            l_ls=self.get_l_ls(), r_ls=self.get_r_ls(),
            z_out_h1=self.get_z_out())

    def simulate(self):
        """
        Make a simulation and get a "SimData" object that contains parameters
        and electrical characteristics of the power amplifier (PA).

        Returns
        -------
        result : SimData
            An object that contains parameters and electrical characteristics
            in frequency and time domains of the PA. It also has methods to
            calculate some useful secondary electrical characteristics (powers,
            energies, etc.).
        """
        s_len = self.get_s_len()  # Number of using time samples.

        # Creation of a "v_pwr_f" vector.
        v_pwr_f = np.zeros((s_len,))
        v_pwr_f[0] = self.get_v_pwr()

        # Calculation of "v_ad_f" and "i_ad_f" vectors.
        y_11_f = self._get_y_11_matrix()
        i_es_f = self._get_i_es_vector()
        v_ad_t, i_ad_t = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                         try_linear=False)
        v_ad_f = self._t2f_array(v_ad_t)  # Harmonics of AD voltage.
        i_ad_f = self._t2f_array(i_ad_t)  # Harmonics of current through the AD.

        # Using the admittance function of (c_ca || g_ca).
        i_ca_f = self._get_curr_in_fd(self._get_y_ca, v_ad_f)

        # Using the admittance function of
        # ((c_cb || g_cb) -- (((c_cs || g_cs) -- l_ls -- r_ls) ||
        # (c_cf || g_cf) || (l_lf -- r_lf -- z_out))).
        i_cb_f = self._get_curr_in_fd(self._get_y_op, v_ad_f)
        i_lb_f = i_ad_f + i_ca_f + i_cb_f  # i_pwr_f = -i_lb_f.

        # Using the admittance function of
        # (((c_cs || g_cs) -- l_ls -- r_ls) || (c_cf || g_cf) ||
        # (l_lf -- r_lf -- z_out)).
        v_cf_f = self._get_volt_in_fd(self._get_y_fso, i_cb_f)  # Harmonics of voltage on node 4.

        # Using the admittance function of ((c_cs || g_cs) -- l_ls -- r_ls).
        i_ls_f = self._get_curr_in_fd(self._get_y_sc, v_cf_f)

        # Using the admittance function of (l_ls -- r_ls).
        v_ls_f = self._get_volt_in_fd(self._get_y_ls, i_ls_f)

        # Using the admittance function of (l_lf -- r_lf -- z_out).
        i_out_f = self._get_curr_in_fd(self._get_y_obr, v_cf_f)

        # Using the "y_out" admittance function.
        v_out_f = self._get_volt_in_fd(self._get_y_out, i_out_f)

        # Voltages.
        sim_db_mgr = self._make_sim_db_manager().create_db()
        sim_db_mgr.add_row(probes=('pwr',), data=v_pwr_f)
        sim_db_mgr.add_row(probes=('ad', 'ca'), data=v_ad_f)
        sim_db_mgr.add_row(probes=('cb',), data=v_ad_f-v_cf_f)
        sim_db_mgr.add_row(probes=('lb',), data=v_pwr_f-v_ad_f)
        sim_db_mgr.add_row(probes=('cf',), data=v_cf_f)
        sim_db_mgr.add_row(probes=('lf',), data=v_cf_f-v_out_f)
        sim_db_mgr.add_row(probes=('cs',), data=v_cf_f-v_ls_f)
        sim_db_mgr.add_row(probes=('ls',), data=v_ls_f)
        sim_db_mgr.add_row(probes=('out',), data=v_out_f)
        sim_db_v = sim_db_mgr.extract_db()

        # Currents.
        sim_db_mgr.create_db()
        sim_db_mgr.add_row(probes=('pwr',), data=-i_lb_f)
        sim_db_mgr.add_row(probes=('ad',), data=i_ad_f)
        sim_db_mgr.add_row(probes=('ca',), data=i_ca_f)
        sim_db_mgr.add_row(probes=('cb',), data=i_cb_f)
        sim_db_mgr.add_row(probes=('lb',), data=i_lb_f)
        sim_db_mgr.add_row(probes=('cf',), data=i_cb_f-i_ls_f-i_out_f)
        sim_db_mgr.add_row(probes=('lf', 'out'), data=i_out_f)
        sim_db_mgr.add_row(probes=('cs', 'ls'), data=i_ls_f)
        sim_db_i = sim_db_mgr.extract_db()

        return SimData(hb_opts=self.get_hb_options(),
                       params=self.get_params(),
                       db_v=sim_db_v, db_i=sim_db_i)
        # The end of "Model.simulate" method.

    # Protected methods.

    def _get_y_obr(self, w):  # Subcircuit 1: output branch.
        """
        An admittance of (l_lf -- r_lf -- z_out) subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return 1/(self._get_z_lf(w) + self._get_z_out(w=w))

    def _get_y_sc(self, w):  # Subcircuit 2: series subcircuit.
        """
        An admittance of ((c_cs || g_cs) -- l_ls -- r_ls) subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return 1/(self._get_z_cs(w) + self._get_z_ls(w))

    def _get_y_fso(self, w):  # Subcircuit 3: forming subcircuit, series subcircuit, output branch.
        """
        An admittance of (((c_cs || g_cs) -- l_ls -- r_ls) ||
        (c_cf || g_cf) || (l_lf -- r_lf -- z_out)) subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return self._get_y_cf(w) + self._get_y_obr(w) + self._get_y_sc(w)

    def _get_y_op(self, w):  # Subcircuit 4: one-port network.
        """
        An admittance of
        ((c_cb || g_cb) -- (((c_cs || g_cs) -- l_ls -- r_ls) ||
        (c_cf || g_cf) || (l_lf -- r_lf -- z_out))) subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return 1/(self._get_z_cb(w) + 1/self._get_y_fso(w))

    def _get_y_11(self, w):
        """
        Y11 parameter of the passive linear quadripole.
        It is equal to an admittance of ((c_ca || g_ca) || (l_lb -- r_lb) ||
        ((c_cb || g_cb) -- (((c_cs || g_cs) -- l_ls -- r_ls) ||
        (c_cf || g_cf) || (l_lf -- r_lf -- z_out))))
        with the shorted power supple voltage source "v_pwr".

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        # y_11 = i_lin/v_1 | v_2 = 0.
        # y_11 = -i_ad/v_ad | v_pwr = 0.
        # y_lb = 1/(1j*w*l_lb + r_lb).
        # See the "y_op" definition above.
        # i_ad = -v_ad/(1/(y_ca + y_lb + y_op)).
        # i_ad = -(y_ca + y_lb + y_op)*v_ad.
        # y_11 = (y_ca + y_lb + y_op)*v_ad/v_ad.
        # y_11 = y_ca + y_lb + y_op = y_ca + y_op - y_12.
        return self._get_y_ca(w) + self._get_y_lb(w) + self._get_y_op(w)

    def _get_y_12(self, w):
        """
        Y12 parameter of the passive linear quadripole.
        It is equal to a negative admittance of (l_lb -- r_lb).

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y_12 : complex
            Y12 parameter.
        """
        # y_12 = i_lin/v_2 | v_1 = 0.
        # y_12 = -i_ad/v_pwr | v_ad = 0.
        # y_lb = 1/(1j*w*l_lb + r_lb).
        # i_ad = y_lb*v_pwr.
        # y_12 = -y_lb*v_pwr/v_pwr = -y_lb.
        return -self._get_y_lb(w)

    def _get_y_11_matrix(self):
        """
        It creates and returns a Y11 parameter matrix.

        Returns
        -------
        y_11_f : ndarray of float
            The Y11 parameter matrix.
        """
        return self._make_admittance_matrix(self._get_y_11)

    def _get_i_es_vector(self):
        """
        It creates and returns "i_es" vector.
        i_es = y_12*v_2, where
        y_12 - Y12 parameter of the passive linear quadripole,
        v_2 - voltage between nodes 2 and 0.
        "i_es" defines the equivalent current source between nodes 1 and 0.
        Its current directs from node 1 to 0.
        Only the 0th element is not zero in this case.

        Returns
        -------
        i_es_f : ndarray of float
            The equivalent current source vector.
        """
        i_es = np.zeros((self.get_s_len(),))
        i_es[0] = np.real(self._get_y_12(0.0))*self.get_v_pwr()
        return i_es

    # The end of "Model" class.


class Tuner(Model):
    """
    The tuner of a class EFx power amplifier (PA) with lumped elements.
    It contains the methods of the PA tuning.
    """

    # Public methods.

    def __init__(self):
        """
        Constructor of the tuner.
        """
        # Call the constructor of the base class.
        Model.__init__(self)

    # Note: this "namedtuple" is part of the public interface.
    SerCirTuning = SerCirLE._SerCirTuning

    def tune_sercir(self, *, h_sup=None, b_cs_h1n=None, x_ls_h1n=None):
        """
        Tune the series resonant subcircuit.
        This subcircuit is used to suppress a certain harmonic of the active
        device voltage.
        There are 3 available combinations of the method parameters:
        1. "h_sup", "b_cs_h1n". An "l_ls" value will be calculated
           automatically.
        2. "h_sup", "x_ls_h1n". A "c_cs" value will be calculated
           automatically.
        3. "b_cs_h1n", "x_ls_h1n". Manual setting.
        Only these couples of parameters can be used simultaneously.
        The usage of "h_sup", "b_cs_h1n", and "x_ls_h1n" values in one method
        call will cause an exception throwing.
        The usage of only an "h_sup" value will cause an exception throwing.
        If a tuning has been failed, the original "c_cs" and "l_ls" values
        will be restored.

        Parameters
        ----------
        h_sup : float
            A harmonic number to suppress. h_sup > 0.
        b_cs_h1n : float
            Normalized susceptance of a series resonant subcircuit's
            capacitance. b_cs_h1n > 0.
            b_cs_h1n = w_wrk*c_cs*r_out, where
            w_wrk - working angular frequency,
            c_cs - capacitance of the series resonant subcircuit,
            r_out - (normalizing) input load network resistance.
        x_ls_h1n : float
            Normalized reactance of a series resonant subcircuit's inductance.
            x_ls_h1n > 0.
            x_ls_h1n = w_wrk*l_ls/r_out, where
            l_ls - inductance of the series resonant subcircuit.
        The normalized values are used here because they do not depend on
        working frequency and input load resistance values. It is more
        convenient than the direct usage of "c_cs" and "l_ls" values.

        Returns
        -------
        result : SerCirTuning
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and
                "False" otherwise.
            status : int
                An error code.
                0 - solution has been successfully found.
                1 - cannot find a "c_cs" value.
                2 - cannot find a "w_sup" value.
                10 - The load network is tunable, but cannot be tuned with
                     current settings.
            message : str
                A message that contains the description of a result.
            w_sup : float
                An angular frequency of a harmonic to suppress.
            c_cs : float
                Capacitance of the series resonant subcircuit.
            l_ls : float
                Inductance of the series resonant subcircuit.

        Notes
        -----
        Call this method after manual setting of the power amplifier parameters
        and before calling any other tuning method. However, it is not
        necessary to call this method if an automatically generated initial
        estimation is used.
        """
        # Check that if a load is tunable, it has been successfully tuned.
        lochres = self.check_load()
        if not lochres.success:
            message = \
                "Cannot tune the series resonant subcircuit.\n" \
                "Cannot tune the load network.\n" + lochres.message
            return self.SerCirTuning(
                success=False, status=10, message=message,
                w_sup=np.nan, c_cs=np.nan, l_ls=np.nan)
        # Auxiliary variables.
        r_norm = np.real(self.get_z_out())
        w_wrk = self.get_w_wrk()
        # The suppressed frequency.
        if h_sup is not None:   w_sup = h_sup*self.get_w_wrk()
        else:                   w_sup = None
        # Series resonant subcircuit's capacitance.
        if b_cs_h1n is not None:
            c_cs = pau.c_from(b_n=b_cs_h1n, r=r_norm, w=w_wrk)
        else:
            c_cs = None
        # Series resonant subcircuit's inductance.
        if x_ls_h1n is not None:
            l_ls = pau.l_from(x_n=x_ls_h1n, r=r_norm, w=w_wrk)
        else:
            l_ls = None
        # The result.
        return self._tune_sercir(w_sup=w_sup, c_cs=c_cs, l_ls=l_ls)

    SmoothTuning = namedtuple(
        'SmoothTuning',
        'success status message c_cf l_lf v_ad_t0 dv_ad_t0 \
         ss_evals hb_iters')

    def tune_smooth(self):
        """
        Find parameters of the forming subcircuit elements (i. e. "c_cf" and
        "l_lf") that provide "smooth" steady-state response.
        It uses "scipy.optimize.root" function with "hybr" option to solve a
        system of nonlinear equations.
        A user can predefine initial estimation by setting "c_cf" and "l_lf"
        values.

        Returns
        -------
        result : SmoothTuning
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and
                "False" otherwise.
            status : int
                "status" flag from "root" result. It has 2 special cases:
                10 - The load network is tunable, but cannot be tuned with
                     current settings.
                13 - A solution was found, but is wrong because (v_ad_t < 0) at
                     some moment (it is a simplified condition, but mostly it
                     is correct).
            message : str
                A message that describes the result of a method call.
            c_cf : float
                A value of forming subcircuit's capacitance. If a solution was
                not found, it contains the last unsuccessful value.
            l_lf : float
                A value of forming subcircuit's inductance. If a solution was
                not found, it contains the last unsuccessful value.
            v_ad_t0 : float
                An extrapolated value of AD voltage when the AD switches into
                conductive state. One of two goals.
            dv_ad_t0 : float
                An extrapolated value of the derivative of AD voltage when the
                AD switches into conductive state. One of two goals.
            ss_evals : int
                The total number of steady-state response evaluations of the
                power amplifier (PA) that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.

        Notes
        -----
        The developer of this library can imagine that someone in a very
        specific case may need to find class EFx parameters with (v_ad_t < 0).
        This method does not support this option. A user can manually tune the
        PA or write their own algorithm on that purpose.

        There is no solution for some sets of "c_cs", "l_ls", and "dc"
        parameters.
        Unlike class E PA model, this method here is auxiliary.
        """
        # Check that if a load is tunable, it has been successfully tuned.
        lochres = self.check_load()
        if not lochres.success:
            message = \
                "Cannot tune the power amplifier.\n" \
                "Cannot tune the load network.\n" + lochres.message
            return self.SmoothTuning(
                success=False, status=10, message=message,
                c_cf=np.nan, l_lf=np.nan,
                v_ad_t0=np.nan, dv_ad_t0=np.nan,
                ss_evals=0, hb_iters=0)
        # HB counter manager: create a counter.
        hb_cnt = self._make_hb_counter()
        # The original (before an execution of the algorithm) values of
        # "c_cf" and "l_lf".
        c_cf_orig = self.get_c_cf()
        l_lf_orig = self.get_l_lf()
        def restore_fsc():  # fsc - forming subcircuit.
            self.set_c_cf(c_cf_orig).set_l_lf(l_lf_orig)
        # The original values of the tunable parameters will be restored if
        # a tuning procedure will fail.
        with ScopeGuardManual(restore_fsc) as sgm_fsc:
            # For normalization.
            w = self.get_w_wrk()
            r_norm = np.real(self.get_z_out())
            # Initial values of the tunable parameters.
            # x_init = [b_cf_n_orig, x_lf_n_orig].
            x_init = [pau.b_n_from(c=c_cf_orig, r=r_norm, w=w),
                      pau.x_n_from(l=l_lf_orig, r=r_norm, w=w)]
            # Try to find a solution.
            # Use a linearized version of the AD if it exists.
            # Note: "args" in the "root" function is a "tuple" of the goal
            # function's extra parameters.
            output = scipyopt.root(
                fun=self.__opt_goals_smooth,
                x0=x_init, args=(True,), method='hybr')
            b_cf_n, x_lf_n = output.x
            c_cf = pau.c_from(b_n=b_cf_n, r=r_norm, w=w)
            l_lf = pau.l_from(x_n=x_lf_n, r=r_norm, w=w)
            # Denormalized errors.
            v_ad_t0, dv_ad_t0 = output.fun*self.get_v_pwr()
            # Check if a solution has been found.
            if output.success:
                # Set the new "c_cf" & "l_lf" values.
                self.set_c_cf(c_cf).set_l_lf(l_lf)
                # Check that the voltage on the AD is not less than zero at
                # each moment.
                tuncheck = self.__check_tuning()
                if not tuncheck.success:
                    # The solution is invalid: (min(v_ad_t) < 0). No commit.
                    return self.SmoothTuning(
                        success=False, status=13, message=tuncheck.message,
                        c_cf=c_cf, l_lf=l_lf,
                        v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
                        ss_evals=hb_cnt.get_ss_evals(),
                        hb_iters=hb_cnt.get_hb_iters())
                    # No commit. The original "c_cf" & "l_lf" values will
                    # be restored.
                # The solution is correct. Commit the solution.
                sgm_fsc.commit()
            # If no solution has been found, no commit will be produced. The
            # original "c_cf" & "l_lf" values will be restored in that case.
        return self.SmoothTuning(
            success=output.success,
            status=output.status, message=output.message,
            c_cf=c_cf, l_lf=l_lf, v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
            ss_evals=hb_cnt.get_ss_evals(), hb_iters=hb_cnt.get_hb_iters())
        # The end of "Tuner.tune_smooth" method.

    MaxCcs = namedtuple(
        'MaxCcs',
        'success status message c_cf l_lf c_cs l_ls v_ad_t0 dv_ad_t0 \
         alg_iters ss_evals hb_iters')

    def maximize_c_cs(self, *, extra_check=True):
        """
        Maximize a "c_cs" value and provide class EFx "smooth" mode.
        "c_cs" maximization reduces Q factor of the series resonant subcircuit.
        It also provides the maximum (or maybe close to it) modified power
        output capability (MPOC) value for a given duty cycle of an active
        device (AD). It is easier to maximize a "c_cs" value than MPOC.
        It is based on "scipy.optimize.minimize" with "trust-constr" option.

        Parameters
        ----------
        extra_check : bool, optional
            The flag of extra check of an optimization result. If it is "True",
            additional check will be made to find if AD voltage is not less
            than zero at any moment. "False" means that no additional check
            will be made. It is useful if the algorithm is used inside another
            algorithm, like in "maximize_mpoc".

        Returns
        -------
        result : MaxCcs
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and
                "False" otherwise.
            status : int
                "status" from "minimize" result. It has 3 special cases:
                10 - The load network is tunable, but cannot be tuned with
                     current settings.
                13 - A solution was found, but is wrong because (v_ad_t < 0)
                     at some moment (it is a simplified condition, but mostly
                     it is correct). This special case can occur only if
                     "extra_check" is "True".
                15 - Cannot start the main tuning procedure because a series
                     resonant subcircuit tuning has failed.
            message : str
                A message that describes the result of a method call.
            c_cf : float
                A value of forming subcircuit's capacitance. If a solution was
                not found, it contains the last unsuccessful value.
            l_lf : float
                A value of forming subcircuit's inductance. If a solution was
                not found, it contains the last unsuccessful value.
            c_cs : float
                A value of series resonant subcircuit's capacitance. If a
                solution was not found, it contains the last unsuccessful
                value. The minimization goal.
            l_ls : float
                A value of series resonant subcircuit's inductance. If a
                solution was not found, it contains the last unsuccessful
                value. The value depends on a "c_cs" value.
            v_ad_t0 : float
                An extrapolated value of AD voltage when the AD switches into
                conductive state. One of two constraints.
            dv_ad_t0 : float
                An extrapolated value of the derivative of AD voltage when the
                AD switches into conductive state. One of two constraints.
            alg_iters : int
                A number of algorithm iterations. The "nit" number from
                "minimize" result.
            ss_evals : int
                The total number of steady-state response evaluations of the
                power amplifier that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.

            Notes
            -----
            This method is public on auxiliary purpose.

            It is not mathematically proved that the maximum "c_cs" value
            always gives the maximum MPOC value for a certain duty cycle value.
            It is just an author's observation. So that, there is a
            possibility, that there can be some conditions under which, this
            statement is (significantly) wrong.
        """
        # Check that if a load is tunable, it has been successfully tuned.
        lochres = self.check_load()
        if not lochres.success:
            message = \
                "Cannot tune the power amplifier.\n" \
                "Cannot tune the load network.\n" + lochres.message
            return self.MaxCcs(
                success=False, status=10, message=message,
                c_cf=np.nan, l_lf=np.nan, c_cs=np.nan, l_ls=np.nan,
                v_ad_t0=np.nan, dv_ad_t0=np.nan,
                alg_iters=0, ss_evals=0, hb_iters=0)
        # HB counter manager: create a counter.
        hb_cnt = self._make_hb_counter()
        # The original (before an execution of the algorithm) values of
        # "c_cf", "l_lf", "c_cs", and "l_ls".
        c_cf_orig = self.get_c_cf()
        l_lf_orig = self.get_l_lf()
        c_cs_orig = self.get_c_cs()
        l_ls_orig = self.get_l_ls()

        def restore_params():
            self.set_c_cf(c_cf_orig).set_l_lf(l_lf_orig)
            self.set_c_cs(c_cs_orig).set_l_ls(l_ls_orig)
        # The original values of the tunable parameters will be restored if
        # a tuning procedure will fail.
        with ScopeGuardManual(restore_params) as sgm_cl:
            # Working angular frequency (for normalization).
            w = self.get_w_wrk()
            # Real part of input load impedance (for normalization).
            r_norm = np.real(self.get_z_out())
            # Parasitic conductance of the "cs" capacitor.
            g_cs = self.get_g_cs()
            # Find the angular frequency that must be suppressed.
            wsres = self._find_w_sup()
            if not wsres.success:
                # Cannot start the optimization process.
                message = \
                    "Cannot start the optimization process.\n" + \
                    wsres.message
                return self.MaxCcs(
                    success=False, status=15, message=message,
                    c_cf=c_cf_orig, l_lf=l_lf_orig,
                    c_cs=c_cs_orig, l_ls=l_ls_orig,
                    v_ad_t0=np.nan, dv_ad_t0=np.nan,
                    alg_iters=0,
                    ss_evals=hb_cnt.get_ss_evals(),
                    hb_iters=hb_cnt.get_hb_iters())
            w_sup = wsres.w_sup
            # Initial values of the tunable parameters.
            # x_init = [b_cf_n_orig, x_lf_n_orig, b_cs_n_orig]
            x_init = [pau.b_n_from(c=c_cf_orig, r=r_norm, w=w),
                      pau.x_n_from(l=l_lf_orig, r=r_norm, w=w),
                      pau.b_n_from(c=c_cs_orig, r=r_norm, w=w)]
            # Minimization goal (negative "b_cs_n").
            def goal(x):
                # Initial value is "c_cs". Normalized value is "b_cs_n".
                # b_c = w*c, b_c_n = w*c*r.
                return -x[2]
            # Set bounds of the tunable paramters.
            bounds = scipyopt.Bounds([0, 0, 0], [np.inf, np.inf, np.inf])
            # Set nonlinear constraints (class EFx "smooth" mode).
            def nl_cons(x):
                c_cs = pau.c_from(b_n=x[2], r=r_norm, w=w)
                l_ls = SerCirLE._l_ls_from(w_sup=w_sup, c_cs=c_cs, g_cs=g_cs)
                self.set_c_cs(c_cs).set_l_ls(l_ls)
                # Use a linearized version of the AD if it exists.
                return self.__opt_goals_smooth(x, True)
            nonlinear_constraint = scipyopt.NonlinearConstraint(
                nl_cons, lb=0, ub=0, jac='2-point', hess=scipyopt.SR1())
            # Maximize a "c_cs" value in the "smooth" mode.
            output = scipyopt.minimize(
                fun=goal, x0=x_init,
                method='trust-constr', jac='2-point', hess=scipyopt.SR1(),
                constraints=nonlinear_constraint, options={'verbose': 0},
                bounds=bounds)
            b_cf_n, x_lf_n, b_cs_n = output.x
            c_cf = pau.c_from(b_n=b_cf_n, r=r_norm, w=w)
            l_lf = pau.l_from(x_n=x_lf_n, r=r_norm, w=w)
            c_cs = pau.c_from(b_n=b_cs_n, r=r_norm, w=w)
            l_ls = SerCirLE._l_ls_from(w_sup=w_sup, c_cs=c_cs, g_cs=g_cs)
            # Denormalized errors.
            # print(f"maximize_c_cs: output.constr={output.constr}")  # Test.
            v_ad_t0, dv_ad_t0 = output.constr[0]*self.get_v_pwr()
            # Check if a solution has been found.
            if output.success:
                # Set the new "c_cf", "l_lf", "c_cs", and "l_ls" values.
                self.set_c_cf(c_cf).set_l_lf(l_lf)
                self.set_c_cs(c_cs).set_l_ls(l_ls)
                if extra_check:  # Do a solution check if it is requested.
                    # Check that the voltage on the AD is not less than zero at
                    # each moment.
                    tuncheck = self.__check_tuning()
                    if not tuncheck.success:
                        # The solution is invalid: (min(v_ad_t) < 0). No commit.
                        return self.MaxCcs(
                            success=False, status=13, message=tuncheck.message,
                            c_cf=c_cf, l_lf=l_lf, c_cs=c_cs, l_ls=l_ls,
                            v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
                            alg_iters=output.nit,
                            ss_evals=hb_cnt.get_ss_evals(),
                            hb_iters=hb_cnt.get_hb_iters())
                        # No commit. The original "c_cf", "l_lf", "c_cs", and
                        # "l_ls" values will be restored.
                # The solution is correct or not checked. Commit the solution.
                sgm_cl.commit()
            # If no solution has been found, no commit will be produced. The
            # original "c_cf", "l_lf", "c_cs", and "l_ls" values will be
            # restored in that case.
        return self.MaxCcs(
            success=output.success,
            status=output.status, message=output.message,
            c_cf=c_cf, l_lf=l_lf, c_cs=c_cs, l_ls=l_ls,
            v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0, alg_iters=output.nit,
            ss_evals=hb_cnt.get_ss_evals(), hb_iters=hb_cnt.get_hb_iters())
        # The end of "Tuner.maximize_c_cs".

    # Options of an automatical initial guess.
    AGOptions = namedtuple('AGOptions', 'active h_sup')
    # Note: it is mostly made for a user's convenience to explicitly show the
    # meaning of the "maximize_mpoc" default arguments. A user can avoid the
    # usage of this "namedtuple" in favour of "tuple".

    MaxMPOC = namedtuple(
        'MaxMPOC',
        'success status message dc c_cf l_lf c_cs l_ls mpoc v_ad_t0 dv_ad_t0 \
         alg_iters subalg_evals ss_evals hb_iters')

    def maximize_mpoc(self, *, auto_guess=AGOptions(active=True, h_sup=2)):
        """
        Maximize a modified power output capability (MPOC) value and provide
        class EFx "smooth" mode.
        It is based on ".optim.optimize" algorithm. The algorithm searches
        the optimum duty cycle "dc" value using "maximize_c_cs" method to find
        "c_cf", "l_lf", "c_cs", and "l_ls" values for each "dc" value.

        Parameters
        ----------
        auto_guess : AGOptions, tuple, list, or bool
            1. A "namedtuple" (by default) that contains following fields:
            active : bool
                If the flag is "True", the algorithm uses an automatically
                generated "dc", "c_cf", "l_lf", and "c_cs" initial estimation.
                This estimation works fine in a wide variety of suitable PA
                parameters.
                If it is "False", then internal "dc", "c_cf", "l_lf", "c_cs",
                and "l_ls" values will be used on that purpose.
                It has built-in settings for class EF2 and EF3. In case of
                class EF3, it slightly changes the suppressed frequency to
                achive better symmetry of active device voltage peaks.
            h_sup : float
                A suppressed hurmonic number. It can be equal to "2" or "3".
                It is used only if (active == True). If (active == False),
                this field can contain anything or even be absent (in case of
                using "tuple" or "list" instead of the "namedtuple").
            2. A boolean flag that contains "False". The meaning here is the
            same as in case of "AGOptions(active=False, h_sup=None)". It cannot
            contain "True".
            Examples of acceptable "auto_guess" values:
            AGOptions(active=True, h_sup=2); AGOptions(active=True, h_sup=3);
            AGOptions(active=False, h_sup=None);
            (True, 2); (True, 3); (False, 2); (False, 3); (False, None);
            False.

        Returns
        -------
        result : MaxMPOC
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and "False"
                otherwise.
            status : int
                "status" from "optimize" result. It has 3 special cases:
                10 - The load network is tunable, but cannot be tuned with
                     current settings.
                12 - The automatic initial estimation option is used, but it
                     cannot be found.
                13 - A solution was found, but is wrong because (v_ad_t < 0) at
                     some moment (it is a simplified condition, but mostly it
                     is correct).
            message : str
                A message that describes the result of a method call.
            dc : float
                A value of duty cycle. The time during which the AD is in
                conductive state.
            c_cf : float
                A value of forming subcircuit's capacitance. If solution was
                not found, it contains the last unsuccessful value.
            l_lf : float
                A value of forming subcircuit's inductance. If solution was
                not found, it contains the last unsuccessful value.
            c_cs : float
                A value of series resonant subcircuit's capacitance. If
                solution was not found, it contains the last unsuccessful
                value.
            l_ls : float
                A value of series resonant subcircuit's inductance. If
                solution was not found, it contains the last unsuccessful
                value.
            mpoc : float
                An MPOC value. The maximization goal.
            v_ad_t0 : float
                An extrapolated value of AD voltage when the AD switches into
                conductive state. One of two constraints.
            dv_ad_t0 : float
                An extrapolated value of the derivative of AD voltage when the
                AD switches into conductive state. One of two constraints.
            alg_iters : int
                A number of algorithm iterations. The "nit" number from
                "scipy.optimize.minimize" result.
            subalg_evals : int
                A number of "maximize_c_cs" evaluations.
            ss_evals : int
                The total number of steady-state response evaluations of the
                power amplifier (PA) that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.

        Notes
        -----
        This is the main tuning method of a class EFx PA.
        It has high computational cost. Therefore, it is recommended to use
        a relatively low harmonic number "h_len" at the beginning of a
        research. For example, (h_len == 32). The precision (i. e. an
        "h_len" value) can be increased at the final stage of the research.
        """
        # Initial guess parameters.
        def process_agoptions(auto_guess):
            # It returns an "AGOptions" instance or throws an exception.
            if isinstance(auto_guess, bool):
                if not auto_guess:
                    return self.AGOptions(active=False, h_sup=None)
                else:
                    raise ValueError("Expected 'h_sup' parameter, but it is not received.")
            elif isinstance(auto_guess, Sequence):  # namedtuple, tuple, list.
                length = len(auto_guess)
                if length < 1:
                    raise ValueError(f"Expected at least 1 parameter ('active'), but {length} parameters received.")
                if length > 2:
                    raise ValueError(f"Expected no more than 2 parameters, but {length} parameters received.")
                active = auto_guess[0]
                if active:
                    if length != 2:
                        raise ValueError(f"Expected the 2nd parameter ('h_sup'), but {length} parameters received.")
                    h_sup = float(auto_guess[1])
                    return self.AGOptions(active=True, h_sup=h_sup)
                else:
                    return self.AGOptions(active=False, h_sup=None)
            else:
                raise TypeError(f"Unexpected 'auto_guess' type: {type(auto_guess)}.")
        agoptions = process_agoptions(auto_guess)
        # Check that if a load is tunable, it has been successfully tuned.
        lochres = self.check_load()
        if not lochres.success:
            message = \
                "Cannot tune the power amplifier.\n" \
                "Cannot tune the load network.\n" + lochres.message
            return self.MaxMPOC(
                success=False, status=10, message=message,
                dc=np.nan, c_cf=np.nan, l_lf=np.nan, c_cs=np.nan, l_ls=np.nan,
                mpoc=np.nan, v_ad_t0=np.nan, dv_ad_t0=np.nan,
                alg_iters=0, subalg_evals=0, ss_evals=0, hb_iters=0)
        # HB counter manager: create a counter.
        hb_cnt = self._make_hb_counter()
        # The original (before an execution of the algorithm) values of "dc",
        # "c_cf", "l_lf", "c_cs", and "l_ls".
        dc_orig = self.get_ad_config().dc
        c_cf_orig = self.get_c_cf()
        l_lf_orig = self.get_l_lf()
        c_cs_orig = self.get_c_cs()
        l_ls_orig = self.get_l_ls()
        def restore_params():
            self.config_ad(dc=dc_orig)
            self.set_c_cf(c_cf_orig).set_l_lf(l_lf_orig)
            self.set_c_cs(c_cs_orig).set_l_ls(l_ls_orig)
        # The original values of the tunable parameters will be restored if
        # a tuning procedure will fail.
        with ScopeGuardManual(restore_params) as sgm_params:
            s_len = self.get_s_len()
            w_wrk = self.get_w_wrk()
            r_norm = np.real(self.get_z_out())
            # Initial values of the tunable parameters.
            if agoptions.active:
                autoestim = self.__get_initial_estimation(
                    h_sup=agoptions.h_sup)
                # Check if an automatically generated initial guess exists.
                if not autoestim.success:
                    return self.MaxMPOC(
                        success=False, status=12, message=autoestim.message,
                        dc=np.nan, c_cf=np.nan, l_lf=np.nan, c_cs=np.nan, l_ls=np.nan,
                        mpoc=np.nan, v_ad_t0=np.nan, dv_ad_t0=np.nan,
                        alg_iters=0, subalg_evals=0,
                        ss_evals=hb_cnt.get_ss_evals(),
                        hb_iters=hb_cnt.get_hb_iters())
                # Use a small step size in the optimization algorithm to
                # reduce the searching area if an approximate "dc" value is
                # known.
                relmaxstep = 0.03
                x_init = int(autoestim.dc*s_len)
                c_cf = pau.c_from(b_n=autoestim.b_cf_h1n, r=r_norm, w=w_wrk)
                l_lf = pau.l_from(x_n=autoestim.x_lf_h1n, r=r_norm, w=w_wrk)
                c_cs = pau.c_from(b_n=autoestim.b_cs_h1n, r=r_norm, w=w_wrk)
                l_ls = pau.l_from(x_n=autoestim.x_ls_h1n, r=r_norm, w=w_wrk)
                self.set_c_cf(c_cf).set_l_lf(l_lf)
                self.set_c_cs(c_cs).set_l_ls(l_ls)
            else:  # No automatically generated initial guess is used.
                # The internal model values will be used.
                x_init = max(1, int(dc_orig*s_len))
                relmaxstep = 0.05
            # Maximization goal (MPOC).
            def goal(sp):
                self.config_ad(dc=sp/s_len)
                # Tune the PA with the maximum "c_cs" value.
                suboutput = self.maximize_c_cs(extra_check=False)
                l_ls = self.get_l_ls()  # A dependent on "c_cs" variable.
                # The 0th row here consists of tuned variables;
                # the 1st row contains the errors of constraints.
                state = (np.array([suboutput.c_cf, suboutput.l_lf, suboutput.c_cs, l_ls]),
                         np.array([suboutput.v_ad_t0, suboutput.dv_ad_t0]))
                if suboutput.success:
                    return pao.FResult(
                        success=True, value=self.__get_mpoc(), state=state)
                else:
                    return pao.FResult(
                        success=False, value=None, state=state)
            # Try to find a solution. Optimization to get the maximum MPOC
            # value by tuning a duty cycle "dc" value.
            output = pao.optimize(
                func=goal,
                x_begin=int(0.2*s_len), x_end=max(1, int(0.8*s_len)),
                x_init=x_init, maxstep=max(1, int(relmaxstep*s_len)),
                comp=pao.cmp_gt)
            # If "x" range is empty, "output.x" is "None". In fact, it can be
            # started with (h_len = 2).
            # All "pao.optimize" results, except the empty range, give "fstate"
            # values. So that, it is possible to extract "fstate" information
            # right after running the "optimize" function.
            # For debugging.
            # dc = (output.x + 1)/s_len
            # dc = output.x/s_len + 1e-12
            # dc = (output.x - 2)/s_len
            dc = output.x/s_len
            c_cf, l_lf, c_cs, l_ls = output.fstate[0]
            v_ad_t0, dv_ad_t0 = output.fstate[1]
            # Check if a solution has been found.
            if output.success:
                self.config_ad(dc=dc)
                self.set_c_cf(c_cf).set_l_lf(l_lf)
                self.set_c_cs(c_cs).set_l_ls(l_ls)
                # For debugging.
                # print(f"maximize_mpoc: fstate={output.fstate}\n"
                #       f"maximize_mpoc: sp={output.x}\n"
                #       f"maximize_mpoc: mpoc_from_opt={output.f}\n"
                #       f"maximize_mpoc: mpoc={self.__get_mpoc()}")
                # Check that the voltage on the AD is not less than zero at
                # each moment.
                tuncheck = self.__check_tuning()
                if not tuncheck.success:
                    # The solution is invalid.
                    return self.MaxMPOC(
                        success=False, status=13, message=tuncheck.message,
                        dc=dc, c_cf=c_cf, l_lf=l_lf, c_cs=c_cs, l_ls=l_ls,
                        mpoc=output.f, v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
                        alg_iters=output.nit, subalg_evals=output.nfev,
                        ss_evals=hb_cnt.get_ss_evals(),
                        hb_iters=hb_cnt.get_hb_iters())
                    # No commit. The original "c_cf", "l_lf", "c_cs", "l_ls",
                    # and "dc" values will be restored.
                # The solution is correct. Commit the solution.
                sgm_params.commit()
            # If no solution has been found, no commit will be produced. The
            # original "c_cf", "l_lf", "c_cs", "l_ls", and "dc" values will be
            # restored in that case.
        return self.MaxMPOC(
            success=output.success,
            status=output.status, message=output.message,
            dc=dc, c_cf=c_cf, l_lf=l_lf, c_cs=c_cs, l_ls=l_ls,
            mpoc=output.f, v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
            alg_iters=output.nit, subalg_evals=output.nfev,
            ss_evals=hb_cnt.get_ss_evals(), hb_iters=hb_cnt.get_hb_iters())
        # The end of "Tuner.maximize_mpoc".

    # Private methods.

    __InitEstim = namedtuple(
        'InitEstim',
        'success status message h_sup dc b_cf_h1n x_lf_h1n b_cs_h1n x_ls_h1n')
    # Note: it is possible to make it returns denormalized values, but
    # normalized values are used here to achieve better style consistency with
    # the class E power amplifier model.

    def __get_initial_estimation(self, h_sup):
        """
        Produce an initial estimation for "maximize_mpoc".
        It can generate estimations for class EF2 and EF3 power amplifiers.
        It strictly requires to have a load network that has been successfully
        checked.

        Parameters
        ----------
        h_sup : float
            A suppressed hurmonic number. Acceptable approximate values:
            2 - class EF2;
            3 - class EF3.
            It does not provide solutions for any other suppressed harmonics.

        Returns
        -------
        result : InitEstim
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if an initial estimation has been
                calculated and "False" otherwise.
            status : int
                An error code. "0" in case of success.
                See "message" field to get more information.
            message : str
                A message that describes the result of a method call.
            h_sup : float
                A suppressed hurmonic number.
            dc : float
                A value of duty cycle. The time during which the active device
                is in conductive state.
            b_cf_h1n : float
                A noramalized susceptance value of forming subcircuit's
                capacitor at a working frequency.
            x_lf_h1n : float
                A noramalized reactance value of forming subcircuit's inductor
                at a working frequency.
            b_cs_h1n : float
                A noramalized susceptance value of series resonant subcircuit's
                capacitor at a working frequency.
            x_ls_h1n : float
                A noramalized reactance value of series resonant subcircuit's
                inductor at a working frequency.
        """
        def get_fail_intro():
            return "Cannot produce an initial estimation."
        # Check that if a load is tunable, it has been successfully tuned.
        # <The check was here.>
        # Parameters.
        z_out_h1 = self.get_z_out()
        r_out_h1 = np.real(z_out_h1);           x_out_h1 = np.imag(z_out_h1)
        g_cf = self.get_g_cf()
        r_lf = self.get_r_lf()
        # Goal.
        if abs(h_sup - 2) < 0.2:  # h_sup == approx(2).
            dc = 0.35
            y_11_h1n = 0.15 - 0.05j
            h_sup = 2.0
            b_cs_h1n = 0.13
        elif abs(h_sup - 3) < 0.2:  # h_sup == approx(3).
            dc = 0.55
            y_11_h1n = 0.30 - 0.21j
            h_sup = 3.12
            b_cs_h1n = 0.12
        else:
            return self.__InitEstim(
                success=False, status=1,
                message=get_fail_intro() + "\n"
                f"There is no estimation for h_sup = {h_sup}.",
                h_sup=h_sup, dc=np.nan, b_cf_h1n=np.nan, x_lf_h1n=np.nan,
                b_cs_h1n=np.nan, x_ls_h1n=np.nan)
        # Working and suppressed angular frequencies.
        w_wrk = self.get_w_wrk()
        w_sup = h_sup*w_wrk
        # Find "c_cs", "l_ls", and "x_ls_h1n".
        y_11_h1 = y_11_h1n/r_out_h1
        c_cs = pau.c_from(b_n=b_cs_h1n, r=r_out_h1, w=w_wrk)
        g_cs = self.get_g_cs()
        l_ls = SerCirLE._l_ls_from(w_sup=w_sup, c_cs=c_cs, g_cs=g_cs)
        x_ls_h1n = pau.x_n_from(l=l_ls, r=r_out_h1, w=w_wrk)
        # Find an intermediate admittance.
        y_ca_h1 = self._get_y_ca(w=w_wrk)
        y_cb_h1 = self._get_y_cb(w=w_wrk)
        z_lb_h1 = self._get_z_lb(w=w_wrk)
        y_cs_h1 = g_cs + 1j*w_wrk*c_cs
        z_ls_h1 = self.get_r_ls() + 1j*w_wrk*l_ls
        y_op = y_11_h1 - y_ca_h1 - 1.0/z_lb_h1
        y_fso_h1 = 1.0/(1.0/y_op - 1.0/y_cb_h1)
        g_fso_h1 = np.real(y_fso_h1);           b_fso_h1 = np.imag(y_fso_h1)
        y_sc_h1 = 1.0/(1.0/y_cs_h1 - z_ls_h1)
        g_sc_h1 = np.real(y_sc_h1);             b_sc_h1 = np.imag(y_sc_h1)
        # Find "x_lf_h1" and "x_lf_h1n".
        radicand_x = (r_lf + r_out_h1)/(g_fso_h1 - g_sc_h1 - g_cf) - (r_lf + r_out_h1)**2
        if radicand_x < 0.0:
            return self.__InitEstim(
                success=False, status=2,
                message=get_fail_intro() + "\n"
                "Cannot find an 'x_lf_h1n' value.\n"
                "The radicand is less than zero.",
                h_sup=h_sup, dc=dc, b_cf_h1n=np.nan, x_lf_h1n=np.nan,
                b_cs_h1n=b_cs_h1n, x_ls_h1n=x_ls_h1n)
        x_lf_h1 = np.sqrt(radicand_x) - x_out_h1
        if x_lf_h1 < 0.0:
            return self.__InitEstim(
                success=False, status=3,
                message=get_fail_intro() + "\n"
                "Cannot find an 'x_lf_h1n' value.\n"
                "The value is less than zero.",
                h_sup=h_sup, dc=dc, b_cf_h1n=np.nan, x_lf_h1n=np.nan,
                b_cs_h1n=b_cs_h1n, x_ls_h1n=x_ls_h1n)
        x_lf_h1n = x_lf_h1/r_out_h1
        # Find "b_cf_h1" and "b_cf_n".
        b_cf_h1 = (g_fso_h1 - g_sc_h1 - g_cf)*(x_lf_h1 + x_out_h1)/(r_lf + r_out_h1) + \
            b_fso_h1 - b_sc_h1
        if b_cf_h1 < 0.0:
            return self.__InitEstim(
                success=False, status=4,
                message=get_fail_intro() + "\n"
                "Cannot find a 'b_cf_n' value.\n"
                "The value is less than zero.",
                h_sup=h_sup, dc=dc, b_cf_n=np.nan, x_lf_h1n=x_lf_h1n,
                b_cs_h1n=b_cs_h1n, x_ls_h1n=x_ls_h1n)
        b_cf_n = b_cf_h1*r_out_h1
        # Successful result.
        return self.__InitEstim(
            success=True, status=0,
            message="An initial estimation has been successfully found.",
            h_sup=h_sup, dc=dc, b_cf_h1n=b_cf_n, x_lf_h1n=x_lf_h1n,
            b_cs_h1n=b_cs_h1n, x_ls_h1n=x_ls_h1n)
        # The end of "Tuner.__get_initial_estimation".

    __TuningCheck = namedtuple('TuningCheck', 'success message')

    def __check_tuning(self):
        """
        Check that (v_ad_t > 0) at each moment.

        Returns
        -------
        result : TuningCheck
            success : bool
                A flag that is "True" if a solution is correct and "False"
                otherwise.
            message : str
                A message that tells the details of a check.
        """
        # Simulate.
        y_11_f = self._get_y_11_matrix()
        i_es_f = self._get_i_es_vector()
        v_ad_t, i_ad_t = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                         try_linear=True)
        # Check.
        v_pwr = self.get_v_pwr()
        r_out_h1 = np.real(self.get_z_out())
        if v_ad_t.min() < -1e-6*v_pwr or i_ad_t.min() < -1e-6*v_pwr/r_out_h1:
            message = \
                "Invalid solution.\n" \
                "Voltage on the active device is less than zero\n" \
                "at some moment."
            return self.__TuningCheck(success=False, message=message)
        message = "The solution is correct."
        return self.__TuningCheck(success=True, message=message)
        # The end of "Tuner.__check_tuning_smooth" method.

    def __opt_goals_smooth(self, x, *args):
        """
        It returns a "list" of errors for a class E tuning procedure. Each
        error value indicates how far the related characteristic is from its
        optimal state. Ideally, the errors must be equal to zeros.

        Class E conditions (goals):
        1. No voltage steps when an active device (AD) switches into conductive
           state.
           v_ad(t=t_wrk) == v_ad(t=0).
        2. The derivative of v_ad(t) equals zero at the moment when an AD
           switches into conductive state. Also it means that the current
           through the AD must start from zero value.
           dv_ad(t=t_wrk)/dt == 0; i_ad(t=0) == 0.
        dt - a very small amount of time (a time sample in this case).
        v_ad(t=t_wrk), dv_ad(t=t_wrk), and i_ad(t=t_wrk) are extrapolations.

        Parameters
        ----------
        x : list, tuple, or ndarray of float
            A "list" of normalized capacitive susceptance "b_cf_n" and
            normalized inductive reactance "x_lf_n" values, i. e.
            x = [b_cf_n, x_lf_n].
        args : tuple
            A "tuple" that can contain this argument:
            try_linear : bool
                If this flag is "True", then it uses a linear representation of
                the AD if it exists. If it is "False", it will always use the
                full nonlinear representation of an AD.

        Returns
        -------
        errors : list of float
            A "list" of normalized errors for an algorithm of solving a system
            of nonlinear equations.
            errors = [(v_ad(t=t_wrk) - v_ad(t=0))/v_pwr,
                      (dv_ad(t=t_wrk)/dt)/v_pwr], where
            v_pwr - power supply voltage.

        Notes
        -----
        The method is made to work with "scipy.optimize.fsolve" and "root".
        Warning: "v_ad_t" must have at least 3 elements.
        """
        # Note: although it would be convenient to have "try_linear" parameter
        # for some tuning scenarios, it is not necessarily for now. It can
        # always be "True".
        try_linear = args[0]

        # Denormalization.
        # Inductance. X = w*L. L = X/w. X = Xn*Rnorm. L = Xn*Rnorm/w.
        # Capacitance. B = w*C, X = -1/(w*C). C = B/w, C = -1/(w*X).
        # B = Bn/Rnorm. C = Bn/(w*Rnorm), C = -1/(w*Xn*Rnorm).
        # x_init = [b_cf_n, x_lf_n].
        w = self.get_w_wrk()
        r_norm = np.real(self.get_z_out())
        c_cf = pau.c_from(b_n=x[0], r=r_norm, w=w)
        l_lf = pau.l_from(x_n=x[1], r=r_norm, w=w)
        self.set_c_cf(c_cf).set_l_lf(l_lf)

        y_11_f = self._get_y_11_matrix()
        i_es_f = self._get_i_es_vector()
        v_ad_t, _ = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                    try_linear=try_linear)
        # Conditions:
        # 1. The first value is a linear extrapolation of "v_ad_t" into the
        #    (t == t_wrk) moment minus an actual "v_ad_t" value at that moment.
        # 2. The second value is an extrapolation of the derivative "v_ad_t" at
        #    the (t == t_wrk) moment.
        # These formulas work when all the "dt" time samples are equal to each
        # other.
        # "v_ad_t[i]" gives a "float" value.
        # Devided by "v_pwr" to normalize the errors.
        v_pwr = self.get_v_pwr()
        # This is the full version that can be used in case of an AD that
        # brings (v_ad_t[0] != 0).
        # return [(2*v_ad_t[-1] - v_ad_t[-2] - v_ad_t[0])/v_pwr,
        #         (2*v_ad_t[-1] - 3*v_ad_t[-2] + v_ad_t[-3])/v_pwr]
        # A simplified, more stable, and precise condition if (v_ad_t[0] == 0).
        return [(2*v_ad_t[-1] - v_ad_t[-2])/v_pwr,
                (2*v_ad_t[-1] - 3*v_ad_t[-2] + v_ad_t[-3])/v_pwr]
        # The end of "Tuner.__opt_goals_smooth" method.

    def __get_i_out_h1(self, v_ad_t):
        """
        Get the 1st harmonic of output current.

        Parameters
        ----------
        v_ad_t : ndarray of float
            An array of voltage samples on the active device in time domain.

        Returns
        -------
        i_out_h1 : complex
            The 1st harmonic of output current.
        """
        # It does not require the entire set of harmonics. Just only the 1st
        # one.
        # Calculate "v_ad_h1" from "v_ad_t".
        v_ad_h1 = self._select_harmonic(ta=v_ad_t, h=1)
        # y_obr: (l_lf -- r_lf -- z_out)
        # y_sc: ((c_cs || g_cs) -- l_ls -- r_ls)
        # y_fso: (((c_cs || g_cs) -- l_ls -- r_ls) ||
        #         (c_cf || g_cf) || (l_lf -- r_lf -- z_out))
        # y_op: ((c_cb || g_cb) -- (((c_cs || g_cs) -- l_ls -- r_ls) ||
        #        (c_cf || g_cf) || (l_lf -- r_lf -- z_out)))
        # i_out_h1 = y_obr_h1*v_cf_h1
        # v_cf_h1 = i_cb_h1/y_fso_h1
        # i_cb_h1 = y_op_h1*v_ad_h1
        # i_out_h1 = v_ad_h1*y_obr_h1*y_op_h1/y_fso_h1
        w = self.get_w_wrk()  # Working angular frequency.
        # For debugging.
        # result = v_ad_h1*self._get_y_obr(w)*self._get_y_op(w)/self._get_y_fso(w)
        # print("__get_i_out_h1 result:", result)
        # return result
        return v_ad_h1*(self._get_y_obr(w)*self._get_y_op(w)/self._get_y_fso(w))
        # The end of "Tuner.__get_i_out_h1" method.

    def __get_s_out_h1(self, v_ad_t=None, i_out_h1=None):
        """
        Get the 1st harmonic of full (complex) output power.

        Parameters
        ----------
        v_ad_t : ndarray of float, optional
            An array of voltage samples on the active device.
        i_out_h1 : complex, optional
            The 1st harmonic of output current.

        Returns
        -------
        s_out_h1 : complex
            The 1st harmonic of full (complex) output power.
        """
        # If there is no "v_ad_t", it will be calculated automatically.
        if v_ad_t is None:
            y_11_f = self._get_y_11_matrix()
            i_es_f = self._get_i_es_vector()
            v_ad_t, _ = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                        try_linear=False)
        # If there is no "i_out_h1", it will be calculated automatically.
        if i_out_h1 is None:
            i_out_h1 = self.__get_i_out_h1(v_ad_t)
        # The main calculations.
        return 0.5*self.get_z_out()*abs(i_out_h1)**2
        # The end of "Tuner.__get_s_out_h1" method.

    def __get_mpoc(self, *, v_ad_t=None, i_ad_t=None, i_out_h1=None):
        """
        Get a modified power output capability (MPOC) value.
        mpoc = p_out_h1/(v_ad_max*i_ad_rms), where
        p_out_h1 - average output active power of the 1st harmonic,
        v_ad_max - the maximum voltage on the active device (AD),
        i_ad_rms - root mean square current through the AD.

        Parameters
        ----------
        v_ad_t : ndarray
            An array of samples of active device voltage.

        Returns
        -------
        mpoc : float
            An MPOC value.
        """
        # If there are no "v_ad_t", "i_ad_t", or i_out_h1, they will be
        # calculated automatically.
        if v_ad_t is None or i_ad_t is None:
            y_11_f = self._get_y_11_matrix()
            i_es_f = self._get_i_es_vector()
            v_ad_t, i_ad_t = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                             try_linear=False)
        # If there is no "i_out_h1", it will be calculated automatically.
        if i_out_h1 is None:
            i_out_h1 = self.__get_i_out_h1(v_ad_t)
        # The main calculations.
        r_out_h1 = np.real(self.get_z_out())
        i_out_h1m = abs(i_out_h1)
        p_out_h1 = 0.5*r_out_h1*i_out_h1m**2
        v_ad_max = v_ad_t.max()
        i_ad_rms = np.sqrt((i_ad_t**2).sum()/i_ad_t.size)
        return p_out_h1/(v_ad_max*i_ad_rms)
        # The end of "Tuner.__get_mpoc" method.

    # The end of "Tuner" class.


class SimData(SimDataBase):
    """
    Simulation data class.
    An object of this class stores simulation data, i. e. electrical
    characteristics and the original parameters of the simulation.
    """

    Probes = namedtuple('Probes', 'pwr ad ca cb lb cf lf cs ls out')

    # Capacitors go first, inductors go second.
    CLEnergs = namedtuple('CLEnergs', 'ca cb cf cs lb lf ls')

    def __init__(self, *, hb_opts, params, db_v, db_i):
        """
        Constructor of a "SimData" object.

        Parameters
        ----------
        hb_opts : OneAD.HBOptions
            Options of a harmonic balance simulation.
        params : Model.Params
            Parameters of a power amplifier model.
        db_v : CharacDB
            A database of voltages in time and frequency domains.
        db_i : CharacDB
            A database of currents in time and frequency domains.
        """
        SimDataBase.__init__(self, hb_opts=hb_opts, params=params,
                             db_v=db_v, db_i=db_i)

    # Get parameters.

    def get_v_pwr(self):            return self.__params.v_pwr
    def get_c_ca(self):             return self.__params.c_ca
    def get_g_ca(self):             return self.__params.g_ca
    def get_c_cb(self):             return self.__params.c_cb
    def get_g_cb(self):             return self.__params.g_cb
    def get_l_lb(self):             return self.__params.l_lb
    def get_r_lb(self):             return self.__params.r_lb
    def get_c_cf(self):             return self.__params.c_cf
    def get_g_cf(self):             return self.__params.g_cf
    def get_l_lf(self):             return self.__params.l_lf
    def get_r_lf(self):             return self.__params.r_lf
    def get_c_cs(self):             return self.__params.c_cs
    def get_g_cs(self):             return self.__params.g_cs
    def get_l_ls(self):             return self.__params.l_ls
    def get_r_ls(self):             return self.__params.r_ls
    def get_z_out_h1(self):         return self.__params.z_out_h1

    # The end of "SimData" class.

# Add methods to get each electrical characteristic (i. e. voltages and
# currents).
add_charac_getters(SimData)
