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
A model of a class E power amplifier with lumped elements.
"""

from collections import namedtuple

import numpy as np
import scipy.optimize as scipyopt

from . import hbmath as hbm
from . import utils as pau
from .circprim import CurrGenLE, FormCirLE
from .one_ad import OneAD
from .scopeguard import ScopeGuardManual
from .simdata import SimDataBase, add_charac_getters


class Model(OneAD, CurrGenLE, FormCirLE):
    """
    A model of a class E power amplifier (PA) with lumped elements.

    Equivalent circuit
    ------------------

     Independent |             Passive             | Active
     voltage     |             linear              | device
     source      |             subcircuit          |

     (2)          i_lb           (1)         i_lin    i_ad
      ------------->--l_lb--r_lb--o--------o---<------->---
      |                           |        |              |
      |                           v i_cb   |              |
      |                           |        |              |
      |                        ---o---     v i_ca         |
      |                        |     |     |              |
      v i_pwr                 c_cb  g_cb   o------        |
      |        (4)             | (3) |     |     |        |
    v_pwr       ---l_lf--r_lf--o--o---    c_ca  g_ca      ad
      |         |                 |        |     |        |
      |         v i_lf/i_out      v i_cf   o------        |
      |         |                 |        |              |
      |       z_out            ---o---     |              |
      |         |              |     |     |              |
      |         |             c_cf  g_cf   |              |
      |         |              |     |     |              |
      ----------o--------------o-----o-----o---------------
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
    work in class E. It is better to add this capacitance there because the
    voltage on "c_cf" is significantly lower than on "c_ca".
    """

    _name = None  # Settable from the outside.

    Probes = namedtuple('Probes', 'pwr ad ca cb lb cf lf out')

    Params = namedtuple(
        'Params',
        'name ad load \
         v_pwr c_ca g_ca c_cb g_cb l_lb r_lb c_cf g_cf l_lf r_lf z_out_h1')

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
        # The default parameters are obtained with (h_len = 512).
        # Basic model parameters.
        self.set_h_len(64).set_t_wrk(2*np.pi)
        # Current generator.
        self.set_v_pwr(1.0).set_c_ca(0.01).set_c_cb(30.0).set_l_lb(10.0)
        # Forming subcircuit.
        self.set_c_cf(0.310).set_l_lf(1.34)
        # Set default g_ca, g_cb, r_lb, g_cf, r_lf values.
        self.set_extra_resists(g_c=1e-6, r_l=1e-6)
        # Active device.
        self.config_ad(name='bidirect:switch', dc=0.5, r_ad=0.1)
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
        return self

    def set_params(self, *,
                   v_pwr=None,
                   c_ca=None, g_ca=None,
                   c_cb=None, g_cb=None, l_lb=None, r_lb=None,
                   c_cf=None, g_cf=None, l_lf=None, r_lf=None):
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
        # Note that "name" format affects on the plotting subsystem.
        return self.Params(
            name='class-e:le',
            ad=self.get_ad_config(), load=self.get_load_config(),
            v_pwr=self.get_v_pwr(),
            c_ca=self.get_c_ca(), g_ca=self.get_g_ca(),
            c_cb=self.get_c_cb(), g_cb=self.get_g_cb(),
            l_lb=self.get_l_lb(), r_lb=self.get_r_lb(),
            c_cf=self.get_c_cf(), g_cf=self.get_g_cf(),
            l_lf=self.get_l_lf(), r_lf=self.get_r_lf(),
            z_out_h1=self.get_z_out())

    def setup_settings(self, file):
        """
        Experimental. Not implemented.
        Load setting from an .ini file that contains the parameters of a power
        amplifier.
        """
        raise NotImplementedError

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
        # ((c_cb || g_cb) -- ((c_cf || g_cf) || (l_lf -- r_lf -- z_out))).
        i_cb_f = self._get_curr_in_fd(self._get_y_op, v_ad_f)
        i_lb_f = i_ad_f + i_ca_f + i_cb_f  # i_pwr_f = -i_lb_f.

        # Using the admittance function of
        # ((c_cf || g_cf) || (l_lf -- r_lf -- z_out)).
        v_cf_f = self._get_volt_in_fd(self._get_y_fo, i_cb_f)

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
        sim_db_mgr.add_row(probes=('out',), data=v_out_f)
        sim_db_v = sim_db_mgr.extract_db()

        # Currents.
        sim_db_mgr.create_db()
        sim_db_mgr.add_row(probes=('pwr',), data=-i_lb_f)
        sim_db_mgr.add_row(probes=('ad',), data=i_ad_f)
        sim_db_mgr.add_row(probes=('ca',), data=i_ca_f)
        sim_db_mgr.add_row(probes=('cb',), data=i_cb_f)
        sim_db_mgr.add_row(probes=('lb',), data=i_lb_f)
        sim_db_mgr.add_row(probes=('cf',), data=i_cb_f-i_out_f)
        sim_db_mgr.add_row(probes=('lf', 'out'), data=i_out_f)
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

    def _get_y_fo(self, w):  # Subcircuit 2: forming subcircuit, output branch.
        """
        An admittance of ((c_cf || g_cf) || (l_lf -- r_lf -- z_out))
        subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return self._get_y_cf(w) + self._get_y_obr(w)

    def _get_y_op(self, w):  # Subcircuit 3: one-port network.
        """
        An admittance of
        ((c_cb || g_cb) -- ((c_cf || g_cf) || (l_lf -- r_lf -- z_out)))
        subcircuit.

        Parameters
        ----------
        w : float
            Angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return 1/(1/self._get_y_fo(w) + 1/self._get_y_cb(w))

    def _get_y_11(self, w):
        """
        Y11 parameter of the passive linear quadripole.
        It is equal to an admittance of ((c_ca || g_ca) || (l_lb -- r_lb) ||
        ((c_cb || g_cb) -- ((c_cf || g_cf) || (l_lf -- r_lf -- z_out))))
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
        # i_ad = -v_ad/(1/(y_ca + y_lb + y_op)) = -(y_ca + y_lb + y_op)*v_ad.
        # y_11 = (y_ca + y_lb + y_op)*v_ad/v_ad = y_ca + y_lb + y_op.
        # y_11 = y_ca + y_op - y_12.
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
        # "r_lb" is necessary to avoid a singular matrix.
        # To get information about the general case of Y parameters, see
        # ["Osnovy teorii tsepey", V. P. Popov, 1985, p. 349. In Russian].
        # For particular examples, see ["Network Theory", "Chapter 7.
        # Two port networks", p. 495 - 583. The author is unknown].
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


# Note: this tuner is made separated from the model because of the programming
# convenience. The separate tuner class will guarantee that the model does not
# rely on any methods from the tuner class. Thus it reduces potential code
# complexity of the model. On the other hand, it breaks encapsulation because
# some of the methods in the model class could be made private if the model
# and tuner are joint.
class Tuner(Model):
    """
    The tuner of a class E power amplifier (PA) with lumped elements.
    It contains the methods of the PA tuning.
    """

    # Public methods.

    def __init__(self):
        """
        Constructor of the tuner.
        """
        # Call the constructor of the base class.
        Model.__init__(self)

    SmoothTuning = namedtuple(
        'SmoothTuning',
        'success status message c_cf l_lf v_ad_t0 dv_ad_t0 ss_evals hb_iters')
    # Note 1: "scipy.optimize.fsolve" and "root" do not provide a number of
    # iterations performed by the algorithm.
    # Note 2: the numerical space of status in "tune_smooth" and "tune_z_out"
    # results are common because both of these methods are used in
    # "tune_freq_range".

    def tune_smooth(self, *, auto_guess=True):
        """
        Find parameters of the forming subcircuit elements (i. e. "c_cf" and
        "l_lf") that provide "smooth" steady-state response.
        It uses "scipy.optimize.root" function with "hybr" option to solve a
        system of nonlinear equations.
        A user can use an automatically generated initial estimation or can
        predefine its own "c_cf" and "l_lf" values.

        Parameters
        ----------
        auto_guess : bool, optional
            If the flag is "True", the algorithm uses an automatically
            generated "c_cf" and "l_lf" initial estimation. This estimation
            works fine in a wide variety of suitable power amplifier (PA)
            parameters.
            If it is "False", internal "c_cf" and "l_lf" values are used on
            that purpose.

        Returns
        -------
        result : SmoothTuning
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and
                "False" otherwise.
            status : int
                "status" flag from "root" result. It has 3 special cases:
                10 - The load network is tunable, but cannot be tuned with
                     current settings.
                12 - The automatic initial estimation option is used, but an
                     estimation cannot be found.
                13 - A solution was found, but is wrong because (v_ad_t < 0) at
                     some moment (it is a simplified condition, but mostly it
                     is correct).
            message : str
                A message that describes the result of the method call.
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
                The total number of steady-state response evaluations of the PA
                that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.

        Notes
        -----
        The developer of this library can imagine that someone in a very
        specific case may need to find class E parameters with (v_ad_t < 0).
        This method does not support this option due to the overcomplexity and
        related issues with an algorithm prototype that was made to solve this
        task. A user can manually tune the PA or write their own algorithm on
        that purpose.
        """
        # Check that if a load is tunable, it has been successfully tuned.
        lochres = self.check_load()  # Load check result.
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
            if auto_guess:
                autoestim = self.__get_initial_estimation()
                # Check if an automatically generated initial guess exists.
                if not autoestim.success:
                    return self.SmoothTuning(
                        success=False, status=12, message=autoestim.message,
                        c_cf=np.nan, l_lf=np.nan,
                        v_ad_t0=np.nan, dv_ad_t0=np.nan,
                        ss_evals=hb_cnt.get_ss_evals(),
                        hb_iters=hb_cnt.get_hb_iters())
                x_init = [autoestim.b_cf_h1n, autoestim.x_lf_h1n]
            else:
                x_init = [pau.b_n_from(c=c_cf_orig, r=r_norm, w=w),
                          pau.x_n_from(l=l_lf_orig, r=r_norm, w=w)]
            # print(f"tune_smooth, x_init={x_init}")  # Test.
            # Try to find a solution using a linearized representation of the
            # AD.
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
                tuncheck = self.__check_tuning_smooth()
                if not tuncheck.success:
                    # The solution is invalid: (min(v_ad_t) < 0). No commit.
                    return self.SmoothTuning(
                        success=False, status=13, message=tuncheck.message,
                        c_cf=c_cf, l_lf=l_lf,
                        v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
                        ss_evals=hb_cnt.get_ss_evals(),
                        hb_iters=hb_cnt.get_hb_iters())
                # The solution is correct. Commit the solution.
                sgm_fsc.commit()
            # If no solution has been found, no commit will be produced. The
            # original "c_cf" & "l_lf" values will be restored in that case.
            # print(f"tune_smooth: {output}")  # Test.
        return self.SmoothTuning(
            success=output.success,
            status=output.status, message=output.message,
            c_cf=c_cf, l_lf=l_lf, v_ad_t0=v_ad_t0, dv_ad_t0=dv_ad_t0,
            ss_evals=hb_cnt.get_ss_evals(), hb_iters=hb_cnt.get_hb_iters())
        # The end of "Tuner.tune_smooth" method.

    ZoutTuning = namedtuple(
        'ZoutTuning',
        'success status message z_out_h1 v_ad_t0 p_out_h1 ss_evals hb_iters')

    def tune_z_out(self, *, p_out_h1, r_norm=None):
        """
        Find load network parameters for working at one frequency.
        The goals:
        1. No voltage steps when the active device (AD) switches into
           conductive state (i. e. no switching losses).
           v_ad(t=t_wrk) == v_ad(t=0).
        2. Provides the same output power at each working frequency in the
           range.
           p_out_h1(f=fx) == p_out_h1(f=f0).
        The working frequency is the internal frequency of a power amplifier
        (PA).
        p_out_h1 - target real part of the 1st harmonic of output power (i. e.
        active power).
        It searches a complex value of input load impedance that satisfies
        these conditions.
        x = [r_out, x_out].
        The algorithm can work only with "zbar" load network.
        It uses "scipy.optimize.root" to solve a system of nonlinear
        equations.
        The function becomes useful when "tune_freq_range" gets stuck at some
        frequency. It provides the ability to manually choose initial
        parameters before starting the tuning at that frequency.

        Parameters
        ----------
        p_out_h1 : float
            A target value of the real (active) part of the 1st harmonic of
            output power.
        r_norm : float, optional
            Normalizing resistance. By default it is the real part of load
            network input impedance.

        Returns
        -------
        result : ZoutTuning
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a solution has been found and
                "False" otherwise.
            status : int
                "status" flag from "root" result. It has 2 special cases:
                11 - The load network must have 'zbar' name, but it does not.
                14 - A solution was found, but is wrong because (v_ad_t < 0)
                     when the AD is in non-conductive state.
            message : str
                A message that describes the result of the method call.
            z_out_h1 : complex
                A value of load network input impedance that satisfies the
                required conditions. If no solution was found, it contains the
                last unsuccessful value.
            v_ad_t0 : float
                An extrapolated value of AD voltage when the AD switches into
                conductive state. One of two goals.
            p_out_h1 : float
                An observed value output active power of the 1st harmonic.
                One of two goals.
            ss_evals : int
                The total number of steady-state response evaluations of the PA
                that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.
        """
        # Check if a load network type is "zbar".
        zbares = self.__check_zbar()
        if not zbares.success:
            message = "Cannot tune the power amplifier.\n" + zbares.message
            return self.ZoutTuning(
                success=False, status=11, message=message,
                z_out_h1=hbm.CX_NAN, v_ad_t0=np.nan, p_out_h1=np.nan,
                ss_evals=0, hb_iters=0)
        # HB counter manager: create a counter.
        hb_ctr = self._make_hb_counter()
        # The original (before an execution of the algorithm) values of a
        # central frequency and input impedance of the load network.
        load_cfg_orig = self.get_load_config()
        def restore_load():
            self.config_load(f_cent=load_cfg_orig.f_cent,
                             z_cent=load_cfg_orig.z_cent)
        # The original parameters of the load network will be restored if a
        # tuning procedure will fail.
        with ScopeGuardManual(restore_load) as sgm_load:
            # For normalization.
            if r_norm is None:      r_norm = np.real(self.get_z_out())
            self.config_load(f_cent=self.get_f_wrk())  # Set a central frequency.
            # x_init = [r_out_n, x_out_n].
            x_init = [np.real(load_cfg_orig.z_cent)/r_norm,
                      np.imag(load_cfg_orig.z_cent)/r_norm]
            output = scipyopt.root(
                fun=self.__opt_goals_freq_range,
                x0=x_init, args=(p_out_h1, r_norm), method='hybr')
            r_out_h1, x_out_h1 = output.x*r_norm
            z_out_h1 = complex(r_out_h1, x_out_h1)
            # Denormalized errors.
            v_ad_t0 = output.fun[0]*self.get_v_pwr()
            # output.fun[1] = (p_out_h1_obs - p_out_h1)/p_out_h1,
            # p_out_h1_obs = (1 + output.fun[1])*p_out_h1.
            p_out_h1_obs = (1 + output.fun[1])*p_out_h1
            # Set a new "z_out_h1" value if a solution has been found.
            if output.success:
                self.config_load(z_cent=z_out_h1)
                # Check that there is no (v_ad_t < 0) while the AD is in
                # non-conductive state.
                tuncheck = self.__check_tuning_z_out()
                if tuncheck.success:
                    # Confirm that the ScopeGuard can be deactivated.
                    sgm_load.commit()
                else:  # Invalid solution.
                    return self.ZoutTuning(
                        success=False, status=14, message=tuncheck.message,
                        z_out_h1=z_out_h1, v_ad_t0=v_ad_t0, p_out_h1=p_out_h1_obs,
                        ss_evals=hb_ctr.get_ss_evals(),
                        hb_iters=hb_ctr.get_hb_iters())
            # If no solution has been found, no commit will be produced. The
            # original "f_cent" & "z_cent" values will be restored in that
            # case.
        return self.ZoutTuning(
            success=output.success,
            status=output.status, message=output.message,
            z_out_h1=z_out_h1, v_ad_t0=v_ad_t0, p_out_h1=p_out_h1_obs,
            ss_evals=hb_ctr.get_ss_evals(), hb_iters=hb_ctr.get_hb_iters())
        # The end of "Tuner.tune_z_out".

    FreqRangeTuning = namedtuple(
        'FreqRangeTuning',
        'success status message c_cf l_lf f_wrk z_out_h1 s_out_h1 v_ad_max \
         i_ad_rms eff_h1 mpoc mpoc_total alg_iters ss_evals hb_iters')

    def tune_freq_range(self, *, freqrat, pts, auto_guess=True):
        """
        Tune a class E power amplifier (PA) for working in a frequency range.
        At the lowest frequency, it tries to find "c_cf" and "l_lf" parameters
        of the forming subcircuit. To do that, it calls the "tune_smooth"
        method. After that, it tries to find a "z_out_h1" value at each higher
        frequency in the range with the same output power and absence of
        switching losses. The "tune_z_out" method is used on that purpose.

        The lowest frequency in the range is the internal frequency of a PA.

        Parameters
        ----------
        freqrat : float
            Maximum to minimum frequencies ratio in the range.
            freqrat = fmax/fmin. freqrat > 1.
        pts : int
            Number of points in the range. pts >= 2.
        auto_guess : bool, optional
            If the flag is "True", the algorithm uses an automatically
            generated "c_cf" and "l_lf" initial estimation. This estimation
            works fine in a wide variety of suitable PA parameters.
            If it is "False", internal "c_cf" and "l_lf" values are used on
            that purpose.

        Returns
        -------
        result : FreqRangeTuning
            A "namedtuple" that contains following fields:
            success : list of bool
                A list of boolean flags for each frequency. Set to "True" if a
                solution has been found and "False" otherwise.
            status : list of int
                A list of integer flags for each frequency. Set to "1" if a
                solution was found, otherwise refer to the respective "message"
                to get more information. A special case:
                16 - No tuning produced.
            message : list of str
                A list of messages. They describe the solution details for each
                frequency.
            c_cf : float
                A forming subcircuit's capacitance.
            l_lf : float
                A forming subcircuit's inductance.
            f_wrk : list of float
                A list of frequencies in the range.
            z_out_h1 : list of complex
                A list of load impedances in the frequency range.
            s_out_h1 : list of complex
                A list of complex output powers in the frequency range.
            v_ad_max : list of float
                A list of maximum voltages on the active device (AD) in the
                frequency range.
            i_ad_rms : list of float
                A list of root mean square currents through the AD in the
                frequency range.
            eff_h1 : list of float
                A list of energy conversion efficiencies for the 1st harmonic
                of output power. At each frequency, it is calculated as
                eff_h1 = e_out_h1/e_pwr, where
                e_out_h1 - output energy of the 1st harmonic during a period of
                time "t_wrk",
                e_pwr - energy consumed from the power supply source "v_pwr"
                during a period of time "t_wrk".
            mpoc : list of float
                A list of values of modified power output capability (MPOC) in
                the frequency range. At each frequency it is calculated as
                mpoc = p_out_h1/(v_ad_max*i_ad_rms), where
                p_out_h1 - active power of the 1st harmonic in the load,
                v_ad_max - the maximum voltage on the AD,
                i_ad_rms - root mean square current through the AD.
            mpoc_total : float
                The total value of MPOC in the whole frequency range. It is
                calculated as
                mpoc_total = p_out_h1/(v_ad_max_max*i_ad_rms_max), where
                p_out_h1 - active power of the 1st harmonic in the load (it is
                the same at each frequency if the tuning is successful),
                v_ad_max_max - the maximum voltage on the AD in the frequency
                range,
                i_ad_rms_max - the maximum value of root mean square current
                through the AD in the frequency range.
                Only successful tuning cases are taken into account.
            alg_iters : int
                Number of iterations of the algorithm. It is equal to the
                number of frequencies in the range. Equals "1" if a tuning
                at the lowest frequency has been failed.
            ss_evals : int
                The total number of steady-state response evaluations of the PA
                that are made during the method call.
            hb_iters : int
                The total number of harmonic balance iterations that are made
                during the method call.
            Each "list" here contains the respective variables for all
            frequencies in the range.
            "state" and "message" are mostly taken from "scipy.optimize.root"
            output ("state" and "message" respectively).

        Notes
        -----
        If no solution has been found at some higher frequencies, it is
        possible to use the "tune_z_out" method with distinct parameters. One
        can manually try to find a good initial estimation. That method is
        specifically made public on that purpose.
        """
        # Check if a load network type is "zbar".
        zbares = self.__check_zbar()
        if not zbares.success:
            message = \
                "Cannot tune the power amplifier in a frequency range.\n" + \
                zbares.message
            return self.FreqRangeTuning(
                success=[False], status=[11], message=[message],
                c_cf=np.nan, l_lf=np.nan,
                f_wrk=[np.nan], z_out_h1=[hbm.CX_NAN], s_out_h1=[hbm.CX_NAN],
                v_ad_max=[np.nan], i_ad_rms=[np.nan],
                eff_h1=[np.nan], mpoc=[np.nan], mpoc_total=np.nan,
                alg_iters=0, ss_evals=0, hb_iters=0)
        # HB counter manager: create a counter.
        hb_cnt = self._make_hb_counter()

        # A small island of Hungarian notation.
        # Creation of the "list"s that will be in the result.
        # "write_necessary_data" and "write_optional_data" nested functions are
        # used to fill them. Beware of data inconsistency; if an element is
        # added into one "list", the respective data must be placed in all the
        # other "list"s.
        list_success = [False]*pts;             list_status = [16]*pts
        list_message = [""]*pts;                list_f_wrk = [np.nan]*pts
        list_z_out_h1 = [hbm.CX_NAN]*pts;       list_s_out_h1 = [hbm.CX_NAN]*pts
        list_v_ad_max = [np.nan]*pts;           list_i_ad_rms = [np.nan]*pts
        list_eff_h1 = [np.nan]*pts;             list_mpoc = [np.nan]*pts

        # Auxiliary functions to calculate optional characteristics.
        def calc_mpoc_total(*, p_out_h1):
            v_ad_max_max = 0.0
            i_ad_rms_max = 0.0
            for idx in range(0, pts):
                if list_success[idx]:  # If a solution was found.
                    v_ad_max = list_v_ad_max[idx]
                    i_ad_rms = list_i_ad_rms[idx]
                    if v_ad_max_max < v_ad_max:     v_ad_max_max = v_ad_max
                    if i_ad_rms_max < i_ad_rms:     i_ad_rms_max = i_ad_rms
            return pau.mpoc(v_ad_max=v_ad_max_max, i_ad_rms=i_ad_rms_max,
                            p_out_h1=p_out_h1)

        def calc_eff_h1(*, v_ad_t, p_out_h1):
            # The 0th harmonic of AD voltage.
            # It is not strictly equal to "v_pwr" because of "r_lb". Some
            # voltage drops on it.
            v_ad_h0 = v_ad_t.sum()/v_ad_t.size
            y_12_h0 = np.real(self._get_y_12(0.0))
            v_pwr = self.get_v_pwr()  # Power supply voltage.
            i_pwr_h0 = y_12_h0*(v_pwr - v_ad_h0)
            p_pwr_avg = -v_pwr*i_pwr_h0
            return p_out_h1/p_pwr_avg

        # There are two functions that produce calculations of output
        # characteristics: one for necessary and one for optional ones.
        # Necessary: success, status, message, f_wrk, z_out_h1.
        # Optional: s_out_h1, v_ad_max, i_ad_rms, eff_h1, mpoc.
        # Separate: c_cf, l_lf, mpoc_total; "message" in case of "tune_smooth"
        # failure.

        # To fill: list_success, list_status, list_message, list_f_wrk,
        # list_z_out_h1.
        def write_necessary_data(*, idx, output):
            list_success[idx] = output.success
            list_status[idx] = output.status
            list_message[idx] = output.message
            list_f_wrk[idx] = self.get_f_wrk()
            list_z_out_h1[idx] = self.get_z_out()

        # There are a couple of private methods to calculate additional result
        # values:
        # __get_i_out_h1(self, v_ad_t)
        # __get_s_out_h1(self, v_ad_t=None, i_out_h1=None)

        # To fill: list_s_out_h1, list_v_ad_max, list_i_ad_rms, list_eff_h1,
        # list_mpoc.
        def write_optional_data(*, idx, output, try_linear):
            # Values that are neccessary to find "v_ad_t" and "i_ad_t".
            y_11_f = self._get_y_11_matrix()
            i_es_f = self._get_i_es_vector()
            # To find "s_out_h1", "eff_h1", and MPOC values.
            v_ad_t, i_ad_t = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                             try_linear=try_linear)
            # The 1st harmonic of output current.
            i_out_h1 = self.__get_i_out_h1(v_ad_t=v_ad_t)
            # Maximum voltage on the AD.
            v_ad_max = v_ad_t.max()
            # Root mean square current through the AD.
            i_ad_rms = pau.rms(arr=i_ad_t)
            # The 1st harmonic of full output power.
            s_out_h1 = self.__get_s_out_h1(v_ad_t=v_ad_t, i_out_h1=i_out_h1)
            # Real part of the 1st harmonic of output power.
            p_out_h1 = np.real(s_out_h1)
            # Writing data into the "list"s.
            list_s_out_h1[idx] = s_out_h1
            list_v_ad_max[idx] = v_ad_max
            list_i_ad_rms[idx] = i_ad_rms
            list_eff_h1[idx] = calc_eff_h1(v_ad_t=v_ad_t, p_out_h1=p_out_h1)
            list_mpoc[idx] = pau.mpoc(v_ad_max=v_ad_max, i_ad_rms=i_ad_rms,
                                      p_out_h1=p_out_h1)

        # Tune the power amplifier at the lowest frequency in a range.
        output = self.tune_smooth(auto_guess=auto_guess)
        # These result values will be obtained no matter successeful the
        # tuning is or not.
        c_cf = output.c_cf
        l_lf = output.l_lf
        write_necessary_data(idx=0, output=output)
        # Check if the tuning was successful. Interrupt if it was not.
        if not output.success:
            # A special message in case of a failure.
            list_message[0] += \
                "\nSolution at the lowest frequency was not found.\n" \
                "Please, try to change input data."
            return self.FreqRangeTuning(
                success=list_success, status=list_status, message=list_message,
                c_cf=c_cf, l_lf=l_lf,
                f_wrk=list_f_wrk, z_out_h1=list_z_out_h1, s_out_h1=list_s_out_h1,
                v_ad_max=list_v_ad_max, i_ad_rms=list_i_ad_rms,
                eff_h1=list_eff_h1, mpoc=list_mpoc, mpoc_total=np.nan,
                alg_iters=1, ss_evals=hb_cnt.get_ss_evals(),
                hb_iters=hb_cnt.get_hb_iters())

        # Continue if the tuning was successful.

        # These result values are written in case of a successful tuning.
        # It uses a linear representation of the AD which is correct for most
        # tuned PAs.
        write_optional_data(idx=0, output=output, try_linear=True)
        # Real part of the 1st harmonic of output power at the lowest frequency
        # in the range. It is used as a target value for higher frequencies.
        p_out_h1 = np.real(list_s_out_h1[0])

        def restore_cfg():
            self.set_f_wrk(list_f_wrk[0])
            self.config_load(f_cent=list_f_wrk[0], z_cent=list_z_out_h1[0])

        with ScopeGuardManual(restore_cfg):
            # For normalization: "r_out" at the lowest frequency.
            r_norm = np.real(self.get_z_out())
            # Math:
            # fmax = freqrat*fmin,
            # frange = fmax - fmin,
            # df = frange/(pts - 1) = (freqrat*fmin - fmin)/(pts - 1) =
            # = fmin*(freqrat - 1)/(pts - 1).
            # Check:
            # f = [2, 3, 4, 5, 6], pts = 5, fmin = 2, fmax = 6,
            # range = 6 - 2 = 4, df = 4/(5 - 1) = 1.
            df = self.get_f_wrk()*(freqrat - 1)/(pts - 1)  # Frequency step.
            for idx in range(1, pts):
                self.set_f_wrk(self.get_f_wrk() + df)
                output = self.tune_z_out(p_out_h1=p_out_h1, r_norm=r_norm)
                write_necessary_data(idx=idx, output=output)
                write_optional_data(idx=idx, output=output, try_linear=False)
            # Restore the minimum frequency and the load impedance value at
            # it after the tuning procedure end.

        # The total MPOC calculation.
        mpoc_total = calc_mpoc_total(p_out_h1=p_out_h1)

        return self.FreqRangeTuning(
            success=list_success, status=list_status, message=list_message,
            c_cf=c_cf, l_lf=l_lf,
            f_wrk=list_f_wrk, z_out_h1=list_z_out_h1, s_out_h1=list_s_out_h1,
            v_ad_max=list_v_ad_max, i_ad_rms=list_i_ad_rms,
            eff_h1=list_eff_h1, mpoc=list_mpoc, mpoc_total=mpoc_total,
            alg_iters=pts, ss_evals=hb_cnt.get_ss_evals(),
            hb_iters=hb_cnt.get_hb_iters())
        # The end of "Tuner.tune_freq_range" method.

    # Private methods.

    def __check_zbar(self):
        """
        Check if a load network name (type in non-programming meaning) is
        "zbar".

        Returns
        -------
        result : LoadCheck
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if the load network name is "zbar",
                "False" otherwise.
            status : int
                An error code.
            message : str
                A message that describes the result of a check.
        """
        load_name = self.get_load_name()
        if load_name != 'zbar':
            return self.LoadCheck(
                success=False, status=1, message=
                "An unacceptable load network type: '{0}'.\n"
                "Only a 'zbar' load network is allowed.".format(load_name))
        return self.LoadCheck(
            success=True, status=0, message=
            "The load network has 'zbar' type.")
        # The end of "Tuner.__check_zbar" method.

    __InitEstim = namedtuple(
        'InitEstim',
        'success status message b_cf_h1n x_lf_h1n')

    def __get_initial_estimation(self):
        """
        Produce an initial estimation for "tune_smooth" method.
        It strictly requires to have a load network that has been successfully
        checked.

        Returns
        -------
        result : InitEstim
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if an initial estimation has been
                calculated and "False" otherwise.
            status : int
                An error code. "0" in case of success.
                See "message" field to get more details.
            message : str
                A message that describes the result of a method call.
            b_cf_h1n : float
                A normalized susceptance value of forming subcircuit's
                capacitor at a working frequency.
            x_lf_h1n : float
                A normalized reactance value of forming subcircuit's
                inductor at a working frequency.
        """
        def get_fail_intro():
            return "Cannot produce an initial estimation."
        # If a load network is tunable, check that it has been successfully
        # tuned.
        # <The check was here.>
        # Parameters.
        w_wrk = self.get_w_wrk()
        z_out_h1 = self.get_z_out()
        r_out_h1 = np.real(z_out_h1);       x_out_h1 = np.imag(z_out_h1)
        g_cf = self.get_g_cf()
        r_lf = self.get_r_lf()
        # Goal.
        g_11_h1 = 0.4/r_out_h1;             b_11_h1 = -0.7*g_11_h1
        y_11_h1 = g_11_h1 + 1j*b_11_h1
        # Find an intermediate admittance value.
        y_ca_h1 = self._get_y_ca(w=w_wrk)
        y_cb_h1 = self._get_y_cb(w=w_wrk)
        z_lb_h1 = self._get_z_lb(w=w_wrk)
        y_op_h1 = y_11_h1 - y_ca_h1 - 1.0/z_lb_h1
        y_fo_h1 = 1.0/(1.0/y_op_h1 - 1.0/y_cb_h1)
        g_fo_h1 = np.real(y_fo_h1);         b_fo_h1 = np.imag(y_fo_h1)
        # Find "x_lf_h1" and "x_lf_h1n".
        radicand_x = (r_lf + r_out_h1)/(g_fo_h1 - g_cf) - (r_lf + r_out_h1)**2
        if radicand_x < 0.0:
            return self.__InitEstim(
                success=False, status=1,
                message=get_fail_intro() + "\n"
                "Cannot find an 'x_lf_h1n' value.\n"
                "The radicand is less than zero.",
                b_cf_h1n=np.nan, x_lf_h1n=np.nan)
        x_lf_h1 = np.sqrt(radicand_x) - x_out_h1
        if x_lf_h1 < 0.0:
            return self.__InitEstim(
                success=False, status=2,
                message=get_fail_intro() + "\n"
                "Cannot find an 'x_lf_h1n' value.\n"
                "The value is less than zero.",
                b_cf_h1n=np.nan, x_lf_h1n=np.nan)
        x_lf_h1n = x_lf_h1/r_out_h1
        # Find "b_cf_h1" and "b_cf_h1n".
        b_cf_h1 = (g_fo_h1 - g_cf)*(x_lf_h1 + x_out_h1)/(r_lf + r_out_h1) + b_fo_h1
        if b_cf_h1 < 0.0:
            return self.__InitEstim(
                success=False, status=3,
                message=get_fail_intro() + "\n"
                "Cannot find a 'b_cf_h1n' value.\n"
                "The value is less than zero.",
                b_cf_h1n=np.nan, x_lf_h1n=x_lf_h1n)
        b_cf_h1n = b_cf_h1*r_out_h1
        # Successful result.
        return self.__InitEstim(
            success=True, status=0,
            message="An initial estimation has been successfully found.",
            b_cf_h1n=b_cf_h1n, x_lf_h1n=x_lf_h1n)
        # The end of "Tuner.__get_initial_estimation" method.

    __TuningCheck = namedtuple('TuningCheck', 'success message')

    def __check_tuning_smooth(self):
        """
        Check that (v_ad_t > 0) at each moment.

        Returns
        -------
        result : TuningCheck
            A "namedtuple" that contains following fields:
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

    def __check_tuning_z_out(self):
        """
        Check that (v_ad_t > 0) when the active device is in non-conductive
        state.

        Returns
        -------
        result : TuningCheck
            A "namedtuple" that contains following fields:
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
                                         try_linear=False)
        # Select the values that characterize non-conductive state of the AD.
        s_len = self.get_s_len()
        dc = self.get_ad_config().dc  # The AD must have a duty cycle.
        sp = hbm.samples_per_pulse(s_len=s_len, dc=dc)
        v_shut = v_ad_t[sp:s_len]
        i_shut = i_ad_t[sp:s_len]
        # Check.
        v_pwr = self.get_v_pwr()
        r_out_h1 = np.real(self.get_z_out())
        if v_shut.min() < -1e-6*v_pwr or i_shut.min() < -1e-6*v_pwr/r_out_h1:
            message = \
                "Invalid solution.\n" \
                "Voltage on the active device (AD) is less than zero\n" \
                "while the AD is in non-conductive state."
            return self.__TuningCheck(success=False, message=message)
        message = "The solution is correct."
        return self.__TuningCheck(success=True, message=message)
        # The end of "Tuner.__check_tuning_z_out" method.

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

    def __opt_goals_freq_range(self, x, *args):
        """
        It returns a "list" of errors for tuning an "z_out_h1" value. Each
        error value indicates how far the related characteristic is from its
        optimal state. Ideally, the errors must be equal to zeros.

        Conditions (goals):
        1. No voltage steps when an active device (AD) switches into conductive
           state.
           v_ad(t=t_wrk) == v_ad(t=0).
        2. Constant real part value of the 1st harmonic of output power.
           p_out_h1(f=fx) == p_out_h1(f=f0).
        dt - a very small amount of time (a time sample in this case).
        f0 - class E "smooth" tuning frequency.
        fx - some other (higher) frequency.
        p_out_h1 -  real (active) part of the 1st harmonic of output power.
        v_ad(t=t_wrk) is extrapolation.

        Parameters
        ----------
        x : list, tuple, or ndarray of float
            A "list" of normalized real and imaginary parts of input impedance
            values of the load network, i. e.
            x = [real(z_out_h1n), imag(z_out_h1n)].
        args : tuple of float
            A tuple that contains two elements:
            p_out_h1 : float
                Real (active) part of the 1st harmonic of output power.
                It cannot be equal to zero.
            r_norm : float
                Normalizing resistance. For example, real part of input load
                impedance at the lowest frequency in a range. It is used to
                normalize tunable variables.
                z_out_h1n = z_out_h1/r_norm.

        Returns
        -------
        errors : list of float
            A "list" of normalized errors for an algorithm of solving a system
            of nonlinear equations.
            errors = [(v_ad(t=t_wrk) - v_ad(t=0))/v_pwr,
                      (p_out_h1(f=fx) - p_out_h1(f=f0))/p_out_h1(f=f0)], where
            v_pwr - power supply voltage.

        Notes
        -----
        The method is made to work with "scipy.optimize.fsolve" and "root".
        Warning: "v_ad_t" must have at least 3 elements.
        """
        r_norm = args[1]
        self.config_load(z_cent=(x[0] + 1j*x[1])*r_norm)

        # For debugging.
        # print(f"__opt_goals_freq_range, type of 'args': {type(args)}")

        v_pwr = self.get_v_pwr()
        p_out_h1 = args[0]

        y_11_f = self._get_y_11_matrix()
        i_es_f = self._get_i_es_vector()
        # Samples of AD voltage. Current is not used. It can use a nonlinear AD
        # model, because AD current can be less than zero at some moments.
        v_ad_t, _ = self._get_vi_ad(y_11_f=y_11_f, i_es_f=i_es_f,
                                    try_linear=False)
        # Magnitude of the 1st harmonic of output current.
        i_out_h1m = abs(self.__get_i_out_h1(v_ad_t))
        # Observed value of real part of the 1st harmonic of output power.
        p_out_h1_obs = 0.5*np.real(self.get_z_out())*i_out_h1m**2

        # For debugging.
        # result = [(2*v_ad_t[-1] - v_ad_t[-2] - v_ad_t[0])/v_pwr,
        #           (p_out_h1_actual - p_out_h1)/p_out_h1]
        # print("__opt_goals_freq_range, i_out_h1m:", i_out_h1m)
        # print("__opt_goals_freq_range, p_out_h1:", p_out_h1)
        # print("__opt_goals_freq_range, result:", result)
        # return result

        # Normalized errors are used.
        # This is the full version that can be used in case of an AD that
        # brings (v_ad_t[0] != 0).
        # return [(2*v_ad_t[-1] - v_ad_t[-2] - v_ad_t[0])/v_pwr,
        #         (p_out_h1_obs - p_out_h1)/p_out_h1]
        # A simplified, more stable, and precise condition if (v_ad_t[0] == 0).
        return [(2*v_ad_t[-1] - v_ad_t[-2])/v_pwr,
                (p_out_h1_obs - p_out_h1)/p_out_h1]
        # The end of "Tuner.__opt_goals_freq_range" method.

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
        # y_op: ((c_cb || g_cb) -- ((c_cf || g_cf) || (l_lf -- r_lf -- z_out)))
        # y_fo: ((c_cf || g_cf) || (l_lf -- r_lf -- z_out))
        # y_obr: (l_lf -- r_lf -- z_out)
        # i_cb_h1 = y_op*v_ad_h1
        # v_cf_h1 = i_cb_h1/y_fo
        # i_out_h1 = y_obr*v_cf_h1
        # i_out_h1 = (y_obr*y_op/y_fo)*v_ad_h1
        w = self.get_w_wrk()  # Working angular frequency.
        # For debugging.
        # result = (self._get_y_obr(w)*self._get_y_op(w)/self._get_y_fo(w))*v_ad_h1
        # print("__get_i_out_h1, result:", result)
        # return result
        return (self._get_y_obr(w)*self._get_y_op(w)/self._get_y_fo(w))*v_ad_h1
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

    # The end of "Tuner" class.


class SimData(SimDataBase):
    """
    Simulation data class.
    An object of this class stores simulation data, i. e. electrical
    characteristics and the original parameters of the simulation.
    """

    Probes = namedtuple('Probes', 'pwr ad ca cb lb cf lf out')

    # Capacitors go first, inductors go second.
    CLEnergs = namedtuple('CLEnergs', 'ca cb cf lb lf')

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
    def get_z_out_h1(self):         return self.__params.z_out_h1

    # The end of "SimData" class.

# Add methods to get each electrical characteristic (i. e. voltages and
# currents).
add_charac_getters(SimData)
