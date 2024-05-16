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
Load networks based on lumped elements.

Some abbreviations.
LN - load network.
PA - power amplifier.
TL - transmission line.
LE - lumped element.

If a load network can have both lumped element and transmission line versions,
then it has additional letters at the class name end:
LE - lumped elements;
TL - transmission lines;
MC - mixed components (with LE & TL).

About usage of these load networks.
These LNs use lazy evaluation mechanism of tuning. When basic parameters are
set, they postopone a tuning procedure until some of the results are requested.
Despite that there is no a separate tuning method, you can consider
"get_params" method as its analogue. Calling this method after setting
parameters using "set_params" method will start a tuning procedure and return
the tuning result. You can examine this result by yourself or automatically (it
has a special flag for that), and make a decision whether you can use the LN
with its parameters or not.
In case of lossless LEs, these LNs can provide any necessary value of required
input impedance "z_in_req" at a certain working angular frequency "w". However,
in case of lossy LEs, a tuning procedure is not always successful. If you do
not need these additional lossies, you can set related resistances to values
that make the LEs almost ideal (i. e. very low capacitor's parallel
conductances and very low inductor's series resistances). Otherwise it is
recommended to always check that a tuning procedure is successful.
"""

from collections import namedtuple

import numpy as np

from ._loadabstract import Tunable, TunRes

__all__ = [
    'SerCirLE',
    'ParCirLE',
    'PiNetLE',
    'TeeNetLE'
]


class SerCirLE(Tunable):
    """
    A series resonant circuit (sRLC).

    Equivalent circuit
    ------------------

        ---g_cs---
        |        |
    o---o--c_cs--o--l_ls--r_ls---
                                |
    in                        r_out
                                |
    o----------------------------

    cs, ls - series capacitor and inductor respectively.
    c_cs, l_ls, r_out - the main elements of the circuit.
    r_out is the effective output load resistance.
    g_cs, r_ls - parasitic parameters of "cs" and "ls" respectively.
    c_cs, l_ls, r_out are tunable parameters.
    g_cs, r_ls are free parameters.
    c_cs, g_cs, l_ls, r_ls, r_out values are represented by floating point
    numbers greater than zero.
    """

    _name = None
    Params = namedtuple('Params', 'name tunres q_eqv c_cs g_cs l_ls r_ls r_out')

    def __init__(self):
        Tunable.__init__(self)
        # Primary parameters of the circuit.
        self.__c_cs = 1.0  # Capacitance.
        self.__g_cs = 1e-6  # Capasitor's parasitic parallel conductance.
        self.__l_ls = 1.0  # Inductance.
        self.__r_ls = 1e-6  # Inductor's parasitic series resistance.
        self.__r_out = 1.0  # Effective load resistance.
        # Required parameters of the circuit at a tuning frequency.
        self.__q_eqv = 1.0  # Required quality factor of the equivalent circuit at a tuning frequency.

    def _assign_params(self, *, g_cs=None, r_ls=None, q_eqv=None):
        # Main circuit parameters.
        if g_cs is not None:    self.__g_cs = float(g_cs)
        if r_ls is not None:    self.__r_ls = float(r_ls)

        # Required parameters.
        if q_eqv is not None:     self.__q_eqv = float(q_eqv)

    def _retrieve_params(self):
        return self.Params(name=self._name, tunres=self._get_tunres(),
                           q_eqv=self.__q_eqv,
                           c_cs=self.__c_cs, g_cs=self.__g_cs,
                           l_ls=self.__l_ls, r_ls=self.__r_ls,
                           r_out=self.__r_out)

    def _eval_z_in(self, w):
        """
        Evaluate an input impedance value of a load network at a given angular
        frequency.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        z_in : complex
            An input impedance value.
        """
        z_cs = 1.0/(self.__g_cs + 1j*w*self.__c_cs)
        z_ls = self.__r_ls + 1j*w*self.__l_ls
        return self.__r_out + z_cs + z_ls

    def _eval_y_in(self, w):
        """
        Evaluate an input admittance value of a load network at a given angular
        frequency.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y_in : complex
            An input admittance value.
        """
        return 1.0/self._eval_z_in(w)

    def _tune(self):
        """
        Tune a series resonant circuit.
        It calculates "c_cs", "l_ls", "r_out" values that provide a required
        "z_in" value at a given frequency.

        Returns
        -------
        tunres : TunRes
            The result of a tuning procedure. See "_get_tunres" to get more
            information.

        Notes
        -----
        It uses an equivalent circuit representation for calculations. This
        representation is valid at only one frequency, in this case at a tuning
        frequency.

        o----c_eqv--l_eqv---
                           |
        in               r_eqv
                           |
        o-------------------

        c_eqv, l_eqv, r_eqv - the main elements of the equivalent circuit.
        """
        w_tun = self._get_w_tun()
        z_in_req = self._get_z_in_req()
        # Resonant angular frequency of the equivalent circuit.
        # (w_eqv > 0) always.
        b1 = z_in_req.imag/(self.__q_eqv*z_in_req.real)
        w_eqv = 0.5*w_tun*(-b1 + float(np.sqrt(b1**2 + 4)))
        # Parameters of the equivalent circuit.
        # These values are always positive.
        r_eqv = z_in_req.real
        l_eqv = self.__q_eqv*r_eqv/w_eqv  # l_ls == l_eqv.
        c_eqv = 1.0/(w_eqv*self.__q_eqv*r_eqv)
        # Parameters of the original circuit.
        d2 = 1.0 - 4.0*(self.__g_cs/(w_tun*c_eqv))**2  # Discriminant No. 2.
        if d2 < 0:
            return TunRes(
                success=False, status=1, message=
                "Discriminant (1 - 4*(g_cs/(w*c_eqv))^2) is less than zero: {0}.\n"
                "Cannot evaluate capacitance.\n"
                "Possible reason: too high 'g_cs' value.".format(d2),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        c_cs = 0.5*c_eqv*(1 + float(np.sqrt(d2)))
        r_out = r_eqv - 1.0/(((w_tun*c_cs)**2)/self.__g_cs + self.__g_cs) - self.__r_ls
        if r_out < 0:
            return TunRes(
                success=False, status=2, message=
                "Output resistance 'r_out' is less than zero: {0}.\n"
                "Possible reasons: too high 'g_cs' value and/or to high 'r_ls' values.".format(r_out),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        self.__c_cs = c_cs
        self.__l_ls = l_eqv
        self.__r_out = r_out
        return TunRes(
            success=True, status=0, message="The tuning is successful.",
            f_tun=self._get_f_tun(),
            z_in_req=self._get_z_in_req(),
            z_in_h1=self._eval_z_in(w=self._get_w_tun()))

# -- The end of "SerCir" class ------------------------------------------------


class ParCirLE(Tunable):
    """
    A parallel resonant circuit (pRLC).

    Equivalent circuit
    ------------------

    o-----o-----o-----o------
          |     |     |     |
          |     |    l_lp   |
    in   c_cp  g_cp   |   r_out
          |     |    r_lp   |
          |     |     |     |
    o-----o-----o-----o------

    cp, lp - parallel capacitor and inductor respectively.
    c_cp, l_lp, r_out - the main elements of the circuit.
    r_out is the effective output load resistance.
    g_cp, r_lp - parasitic parameters of "cp" and "lp" respectively.
    c_cp, l_lp, r_out are tunable parameters.
    g_cp, r_lp are free parameters.
    c_cp, g_cp, l_lp, r_lp, r_out values are represented by floating point
    numbers greater than zero.
    """

    _name = None
    Params = namedtuple('Params', 'name tunres q_eqv c_cp g_cp l_lp r_lp r_out')

    def __init__(self):
        Tunable.__init__(self)
        # Primary parameters of the circuit.
        self.__c_cp = 1.0  # Capacitance.
        self.__g_cp = 1e-6  # Capasitor's parasitic parallel conductance.
        self.__l_lp = 1.0  # Inductance.
        self.__r_lp = 1e-6  # Inductor's parasitic series resistance.
        self.__r_out = 1.0  # Effective load resistance.
        # Required parameters of the circuit at a tuning frequency.
        self.__q_eqv = 1.0  # Required quality factor of the equivalent circuit at a tuning frequency.

    def _assign_params(self, *, g_cp=None, r_lp=None, q_eqv=None):
        # Main circuit parameters.
        if g_cp is not None:    self.__g_cp = float(g_cp)
        if r_lp is not None:    self.__r_lp = float(r_lp)

        # Required parameters.
        if q_eqv is not None:     self.__q_eqv = float(q_eqv)

    def _retrieve_params(self):
        return self.Params(name=self._name, tunres=self._get_tunres(),
                           q_eqv=self.__q_eqv,
                           c_cp=self.__c_cp, g_cp=self.__g_cp,
                           l_lp=self.__l_lp, r_lp=self.__r_lp,
                           r_out=self.__r_out)

    def _eval_z_in(self, w):
        return 1.0/self._eval_y_in(w)

    def _eval_y_in(self, w):
        y_cp = self.__g_cp + 1j*w*self.__c_cp
        y_lp = 1.0/(self.__r_lp + 1j*w*self.__l_lp)
        return y_cp + y_lp + 1.0/self.__r_out

    def _tune(self):
        """
        Tune a parallel resonant circuit.
        It calculates "c_cp", "l_lp", "r_out" values that provide a required
        "z_in" value at a given frequency.

        Returns
        -------
        tunres : TunRes
            The result of a tuning procedure. See "_get_tunres" to get more
            information.

        Notes
        -----
        It uses an equivalent circuit representation for calculations. This
        representation is valid at only one frequency, in this case at a tuning
        frequency.

        o-----o------o-------
              |      |      |
        in  c_eqv  l_eqv  r_eqv
              |      |      |
        o-----o------o-------

        c_eqv, l_eqv, r_eqv - the main elements of the equivalent circuit.
        """
        w_tun = self._get_w_tun()
        z_in_req = self._get_z_in_req()
        # Resonant angular frequency of the equivalent circuit.
        # (w_eqv > 0) always.
        b1 = z_in_req.imag/(self.__q_eqv*z_in_req.real)  # It is not actually "b1", there must be "-" sign.
        w_eqv = 0.5*w_tun*(b1 + float(np.sqrt(b1**2 + 4)))
        # Parameters of the equivalent circuit.
        # These values are always positive.
        r_eqv = z_in_req.real*((z_in_req.imag/z_in_req.real)**2 + 1)
        l_eqv = r_eqv/(w_eqv*self.__q_eqv)
        c_eqv = self.__q_eqv/(w_eqv*r_eqv)  # c_cp == c_eqv.
        # Parameters of the original circuit.
        d2 = 1 - (2.0*self.__r_lp/(w_tun*l_eqv))**2  # Discriminant No. 2.
        if d2 < 0:
            return TunRes(
                success=False, status=1, message=
                "Discriminant (1 - (2*r_lp/(w*l_eqv))^2) is less than zero: {0}.\n"
                "Cannot evaluate inductance.\n"
                "Possible reason: too high 'r_lp' value.".format(d2),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        l_lp = 0.5*l_eqv*(1 + float(np.sqrt(d2)))
        r_out = 1.0/(1.0/r_eqv - self.__g_cp - self.__r_lp/(self.__r_lp**2 + (w_tun*l_lp)**2))
        if r_out < 0:
            return TunRes(
                success=False, status=2, message=
                "Output resistance 'r_out' is less than zero: {0}.\n"
                "Possible reasons: too high 'g_cp' value and/or to high 'r_lp' values.".format(r_out),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        self.__c_cp = c_eqv
        self.__l_lp = l_lp
        self.__r_out = r_out
        return TunRes(
            success=True, status=0, message="The tuning is successful.",
            f_tun=self._get_f_tun(),
            z_in_req=self._get_z_in_req(),
            z_in_h1=self._eval_z_in(w=self._get_w_tun()))

# -- The end of "ParCir" class ------------------------------------------------


class PiNetLE(Tunable):
    """
    A Pi network.

    Equivalent circuit
    ------------------

    o-----o-----o--l_lm--r_lm--o-----o------
          |     |              |     |     |
    in   c_ci  g_ci           c_co  g_co r_out
          |     |              |     |     |
    o-----o-----o--------------o-----o------

    ci, co - input and output capacitors respectively.
    lm - middle inductor.
    c_ci, c_co, l_lm, r_out - the main elements of the circuit.
    r_out is the effective output load resistance.
    g_ci, g_co, r_lm - parasitic parameters of "ci", "co", and "lm"
    respectively.
    c_ci, c_co, l_lm are tunable parameters.
    g_ci, g_co, r_lm, r_out are free parameters.
    c_ci, g_ci, c_co, g_co, l_lm, r_lm, r_out values are represented by
    floating point numbers greater than zero.
    """

    _name = None
    Params = namedtuple('Params', 'name tunres midcoef c_ci g_ci l_lm r_lm c_co g_co r_out')

    def __init__(self):
        Tunable.__init__(self)
        # Primary parameters of the circuit.
        self.__c_ci = 1.1  # Input capacitance.
        self.__g_ci = 1e-6  # Input capasitor's parasitic parallel conductance.
        self.__c_co = 1.1  # Output capacitance.
        self.__g_co = 1e-6  # Output capasitor's parasitic parallel conductance.
        self.__l_lm = 1.1  # Middle inductance.
        self.__r_lm = 1e-6  # Middle inductor's parasitic series resistance.
        self.__r_out = 1.0  # Effective load resistance.
        # A coefficient that is used to get a value of equivalent resistance in
        # the middle point
        # r_mp = midcoef/(g_in + g_out).
        # The less its value is, the more efficient filtering quality and the
        # narrower frequency band are.
        # Recommendation (approximate): midcoef E (0, 1].
        self.__midcoef = 1.0

    def _assign_params(self, *, g_ci=None, g_co=None, r_lm=None, r_out=None, midcoef=None):
        # Main circuit parameters.
        if g_ci is not None:        self.__g_ci = float(g_ci)
        if g_co is not None:        self.__g_co = float(g_co)
        if r_lm is not None:        self.__r_lm = float(r_lm)
        if r_out is not None:       self.__r_out = float(r_out)

        # Auxiliary parameters.
        if midcoef is not None:     self.__midcoef = float(midcoef)

    def _retrieve_params(self):
        return self.Params(name=self._name, tunres=self._get_tunres(),
                           midcoef=self.__midcoef,
                           c_ci=self.__c_ci, g_ci=self.__g_ci,
                           l_lm=self.__l_lm, r_lm=self.__r_lm,
                           c_co=self.__c_co, g_co=self.__g_co,
                           r_out=self.__r_out)

    def _eval_z_in(self, w):
        return 1.0/self._eval_y_in(w)

    def _eval_y_in(self, w):
        y_ci = self.__g_ci + 1j*w*self.__c_ci
        y_co = self.__g_co + 1j*w*self.__c_co
        z_lm = self.__r_lm + 1j*w*self.__l_lm
        return y_ci + 1.0/(z_lm + 1.0/(y_co + 1.0/self.__r_out))

    def _tune(self):
        w_tun = self._get_w_tun()
        y_in_req = 1.0/self._get_z_in_req()
        if y_in_req.real <= self.__g_ci:
            return TunRes(
                success=False, status=1, message=
                "Required input conductance value ({0}) must be greater than\n"
                "parasitic conductance of input capacitor ({1}).\n"
                "Cannot tune the load network.\n".format(y_in_req.real, self.__g_ci),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        g_out = 1.0/self.__r_out
        # Try ot find an equivalent resistance in the middle point.
        rmpres = self.__find_r_mp()
        if not rmpres.success:
            return TunRes(
                success=False, status=rmpres.status, message=rmpres.message,
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        r_mp = rmpres.r_mp
        # Susceptance of the input capacitance.
        radicand_b_ci = (y_in_req.real - self.__g_ci)/(r_mp + 0.5*self.__r_lm) - (y_in_req.real - self.__g_ci)**2  # Radicand of "b_ci".
        if radicand_b_ci < 0:
            return TunRes(
                success=False, status=4, message=
                "Radicand of 'b_ci' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_b_ci),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        b_ci = y_in_req.imag + float(np.sqrt(radicand_b_ci))
        if b_ci <= 0:
            return TunRes(
                success=False, status=5, message=
                "Required susceptance value of the input capacitance\n"
                "is not greater than zero: {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reason: too low required negative input susceptance value.".format(b_ci),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        # Susceptance of the output capacitance.
        radicand_b_co = (g_out + self.__g_co)/(r_mp - 0.5*self.__r_lm) - (g_out + self.__g_co)**2
        if radicand_b_co <= 0:
            return TunRes(
                success=False, status=6, message=
                "Radicand of 'b_co' is not greater than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_b_co),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        b_co = float(np.sqrt(radicand_b_co))
        # Reactance of the middle inductance.
        radicand_x_lm1 = (r_mp + 0.5*self.__r_lm)/(y_in_req.real - self.__g_ci) - (r_mp + 0.5*self.__r_lm)**2
        if radicand_x_lm1 < 0:
            return TunRes(
                success=False, status=7, message=
                "Radicand No. 1 of 'x_lm' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_x_lm1),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        radicand_x_lm2 = (r_mp - 0.5*self.__r_lm)/(g_out + self.__g_co) - (r_mp - 0.5*self.__r_lm)**2
        if radicand_x_lm2 < 0:
            return TunRes(
                success=False, status=8, message=
                "Radicand No. 2 of 'x_lm' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_x_lm2),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        x_lm = float(np.sqrt(radicand_x_lm1) + np.sqrt(radicand_x_lm2))
        # Setting values of elements.
        self.__c_ci = b_ci/w_tun
        self.__c_co = b_co/w_tun
        self.__l_lm = x_lm/w_tun
        return TunRes(
            success=True, status=0, message="The tuning is successful.",
            f_tun=self._get_f_tun(),
            z_in_req=self._get_z_in_req(), z_in_h1=self._eval_z_in(w=self._get_w_tun()))

    __FoundRmp = namedtuple('FoundRmp', 'success status message r_mp')

    # Mathematically, the checking of bounds of input values will guarantee
    # that "b_ci", "b_co", "x_lm" are real positive values. However,
    # evaluations with floating point numbers can produce a situation when
    # there can arise a very small negative number under a square root which
    # was produced by the imprecise math implementation in a computer
    # architecture. This situation will cause an exception throwing. To avoid
    # it, each non-negativity of each radicand must be checked explicitly.
    def __find_r_mp(self):
        y_in_req = 1.0/self._get_z_in_req()
        g_out = 1.0/self.__r_out
        r_mp = self.__midcoef/(y_in_req.real + g_out)
        # Check if "r_mp" is not too low.
        if r_mp <= 0.5*self.__r_lm:
            return self.__FoundRmp(
                success=False, status=2, message=
                "An equivalent resistance value 'r_mp' in the middle point\n"
                "is not greater than (r_lm/2): {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reasons: too low 'midcoef' and/or too high 'r_lm' values.\n".format(r_mp),
                r_mp=r_mp)
        # Check if "r_mp" is not too high.
        if y_in_req.imag < 0:
            r_mp_max3 = (y_in_req.real - self.__g_ci)/(y_in_req.imag**2 + (y_in_req.real - self.__g_ci)**2) - 0.5*self.__r_lm
        else:
            r_mp_max3 = float('inf')
        r_mp_max = min(
            1.0/(y_in_req.real - self.__g_ci) - 0.5*self.__r_lm,
            1.0/(g_out + self.__g_co) + 0.5*self.__r_lm,
            r_mp_max3)
        if r_mp >= r_mp_max:
            return self.__FoundRmp(
                success=False, status=3, message=
                "An equivalent resistance value 'r_mp' in the middle point\n"
                "is too high: {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reasons: too high 'midcoef' and/or too high 'g_ci' and/or\n"
                "too high 'g_co' and/or too high 'r_lm' values.".format(r_mp),
                r_mp=r_mp)
        return self.__FoundRmp(
            success=True, status=0, message="'r_mp' is successfully found.",
            r_mp=r_mp)

# -- The end of "PiNetLE" class -----------------------------------------------


class TeeNetLE(Tunable):
    """
    A Tee network.

    Equivalent circuit
    ------------------

    o--l_li--r_li--o-----o--l_lo--r_lo---
                   |     |              |
    in            c_cm  g_cm          r_out
                   |     |              |
    o--------------o-----o---------------

    li, lo - input and output inductors respectively.
    c_cm - middle capaciotor.
    c_cm, l_li, l_lo, r_out - the main elements of the circuit.
    r_out is the effective output load resistance.
    g_cm, r_li, r_lo - parasitic parameters of "cm", "li", and "lo"
    respectively.
    c_cm, l_li, l_lo are tunable parameters.
    g_cm, r_li, r_lo, r_out are free parameters.
    c_cm, g_cm, l_li, r_li, l_lo, r_lo, r_out values are represented by
    floating point numbers greater than zero.
    """

    _name = None
    Params = namedtuple('Params', 'name tunres midcoef l_li r_li c_cm g_cm l_lo r_lo r_out')

    def __init__(self):
        Tunable.__init__(self)
        # Primary parameters of the circuit.
        self.__c_cm = 1.0  # Middle capacitance.
        self.__g_cm = 1e-6  # Middle capasitor's parasitic parallel conductance.
        self.__l_li = 1.0  # Input inductance.
        self.__r_li = 1e-6  # Input inductor's parasitic series resistance.
        self.__l_lo = 1.0  # Output inductance.
        self.__r_lo = 1e-6  # Output inductor's parasitic series resistance.
        self.__r_out = 1.0  # Effective load resistance.
        # A coefficient that is used to get a value of equivalent conductance
        # in the middle point
        # g_mp = midcoef/(r_in + r_out).
        # The less its value is, the more efficient filtering and the narrower
        # frequency band are.
        # Recommendation (approximate): midcoef E (0, 1].
        self.__midcoef = 1.0

    def _assign_params(self, *, g_cm=None, r_li=None, r_lo=None, r_out=None, midcoef=None):
        # Main circuit parameters.
        if g_cm is not None:        self.__g_cm = float(g_cm)
        if r_li is not None:        self.__r_li = float(r_li)
        if r_lo is not None:        self.__r_lo = float(r_lo)
        if r_out is not None:       self.__r_out = float(r_out)

        # Auxiliary parameters.
        if midcoef is not None:     self.__midcoef = float(midcoef)

    def _retrieve_params(self):
        return self.Params(name=self._name, tunres=self._get_tunres(),
                           midcoef=self.__midcoef,
                           l_li=self.__l_li, r_li=self.__r_li,
                           c_cm=self.__c_cm, g_cm=self.__g_cm,
                           l_lo=self.__l_lo, r_lo=self.__r_lo,
                           r_out=self.__r_out)

    def _eval_z_in(self, w):
        y_cm = self.__g_cm + 1j*w*self.__c_cm
        z_li = self.__r_li + 1j*w*self.__l_li
        z_lo = self.__r_lo + 1j*w*self.__l_lo
        return z_li + 1.0/(y_cm + 1.0/(self.__r_out + z_lo))

    def _eval_y_in(self, w):
        return 1.0/self._eval_z_in(w)

    def _tune(self):
        w = self._get_w_tun()
        z_in_req = self._get_z_in_req()
        if z_in_req.real <= self.__r_li:
            return TunRes(
                success=False, status=1, message=
                "Required input resistance value ({0}) must be greater than\n"
                "parasitic resistance of input inductor ({1}).\n"
                "Cannot tune the load network.\n".format(z_in_req.real, self.__r_li),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        # Try ot find an equivalent conductance in the middle point.
        gmpres = self.__find_g_mp()
        if not gmpres.success:
            return TunRes(
                success=False, status=gmpres.status, message=gmpres.message,
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        g_mp = gmpres.g_mp
        # Reactance of the input inductance. It must not be negative.
        radicand_x_li = (z_in_req.real - self.__r_li)/(g_mp + 0.5*self.__g_cm) - (z_in_req.real - self.__r_li)**2
        if radicand_x_li < 0:
            return TunRes(
                success=False, status=4, message=
                "Radicand of 'x_li' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_x_li),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        x_li = z_in_req.imag + float(np.sqrt(radicand_x_li))
        if x_li <= 0:
            return TunRes(
                success=False, status=5, message=
                "Required reactance value of the input inductance\n"
                "is not greater than zero: {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reason: too low required negative input reactance value.".format(x_li),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        # Reactance of the output inductance.
        radicand_x_lo = (self.__r_out + self.__r_lo)/(g_mp - 0.5*self.__g_cm) - (self.__r_out + self.__r_lo)**2
        if radicand_x_lo <= 0:
            return TunRes(
                success=False, status=6, message=
                "Radicand of 'x_lo' is not greater than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_x_lo),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        x_lo = float(np.sqrt(radicand_x_lo))
        # Susceptance of the middle capacitance.
        radicand_b_cm1 = (g_mp + 0.5*self.__g_cm)/(z_in_req.real - self.__r_li) - (g_mp + 0.5*self.__g_cm)**2
        if radicand_b_cm1 < 0:
            return TunRes(
                success=False, status=7, message=
                "Radicand No. 1 of 'b_cm' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_b_cm1),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        radicand_b_cm2 = (g_mp - 0.5*self.__g_cm)/(self.__r_out + self.__r_lo) - (g_mp - 0.5*self.__g_cm)**2
        if radicand_b_cm2 < 0:
            return TunRes(
                success=False, status=8, message=
                "Radicand No. 2 of 'b_cm' is less than zero: {0}.\n"
                "Cannot tune the load network.\n".format(radicand_b_cm2),
                f_tun=self._get_f_tun(),
                z_in_req=self._get_z_in_req(), z_in_h1=None)
        b_cm = float(np.sqrt(radicand_b_cm1) + np.sqrt(radicand_b_cm2))
        # Setting values of elements.
        self.__c_cm = b_cm/w
        self.__l_li = x_li/w
        self.__l_lo = x_lo/w
        return TunRes(
            success=True, status=0, message="The tuning is successful.",
            f_tun=self._get_f_tun(),
            z_in_req=self._get_z_in_req(),
            z_in_h1=self._eval_z_in(w=self._get_w_tun()))

    __FoundGmp = namedtuple('FoundGmp', 'success status message g_mp')

    # Checks to (almost) ensure that the radicands of "x_li", "x_lo", "b_cm"
    # are positive values. See the analogous comment on "PiNetLE.__find_r_mp"
    # method to get more information.
    def __find_g_mp(self):
        z_in_req = self._get_z_in_req()
        g_mp = self.__midcoef/(z_in_req.real + self.__r_out)
        # Check if "g_mp" is not too low.
        g_mp_min = 0.5*self.__g_cm
        if g_mp <= g_mp_min:
            return self.__FoundGmp(
                success=False, status=2, message=
                "An equivalent conductance value 'g_mp' in the middle point\n"
                "is not greater than (g_cm/2): {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reasons: too low 'midcoef' and/or too high 'g_cm' values.\n".format(g_mp),
                g_mp=g_mp)
        # Check if "g_mp" is not too high.
        if z_in_req.imag < 0:
            g_mp_max3 = (z_in_req.real - self.__r_li)/((z_in_req.real - self.__r_li)**2 + z_in_req.imag**2) - 0.5*self.__g_cm
        else:
            g_mp_max3 = float('inf')
        g_mp_max = min(
            1.0/(z_in_req.real - self.__r_li) - 0.5*self.__g_cm,
            1.0/(self.__r_out + self.__r_lo) + 0.5*self.__g_cm,
            g_mp_max3
        )
        if g_mp >= g_mp_max:
            return self.__FoundGmp(
                success=False, status=3, message=
                "An equivalent conductance value 'g_mp' in the middle point\n"
                "is too high: {0}.\n"
                "Cannot tune the load network.\n"
                "Possible reasons: too high 'midcoef' and/or too high 'g_cm' and/or\n"
                "too high 'r_li' and/or too high 'r_lo' values.".format(g_mp),
                g_mp=g_mp)
        # The value is OK.
        return self.__FoundGmp(
            success=True, status=0, message="'g_mp' is successfully found.",
            g_mp=g_mp)

# -- The end of "TeeNetLE" class ----------------------------------------------


# -- Load networks to do are below --------------------------------------------
# Empty.
# -- The end of load networks to do -------------------------------------------
