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
Circuit primitives.

Devices:
v_pwr - independent power supply voltage source.
ca - anode-cathode capacitor of an active device (AD).
cn - negative feedback capacitor.
cb - DC current blocking capacitor.
lb - AC current blocking inductor.
cf - forming subcircuit's capacitor.
lf - forming subcircuit's inductor.
cp - parallel resonant subcircuit's capacitor.
lp - parallel resonant subcircuit's inductor.
cs - series resonant subcircuit's capacitor.
ls - series resonant subcircuit's inductor.
"""

from collections import namedtuple

import numpy as np

from . import paslincomp as plc
from .scopeguard import ScopeGuardManual

__all__ = ['CurrGenLE', 'FormCirLE', 'SerCirLE', 'NegFbkLE']
# To add in the "list" when it will be done: 'ParCirLE'.


class CurrGenLE:
    """
    The base class for a current generator with lumped elements.
    It contains: v_pwr, ca, cb, lb.
    """

    def __init__(self):
        self.__v_pwr = 1.0
        self.__ca = plc.Capacitor()
        self.__cb = plc.Capacitor()
        self.__lb = plc.Inductor()

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # An independent constant voltage source.
    def set_v_pwr(self, val):
        self.__v_pwr = float(val)
        return self

    def get_v_pwr(self):
        return self.__v_pwr

    # An AD anode-cathode capacitor with lossies.
    def set_c_ca(self, val):
        self.__ca.set_c(val)
        return self

    def get_c_ca(self):
        return self.__ca.get_c()

    def set_g_ca(self, val):
        self.__ca.set_g(val)
        return self

    def get_g_ca(self):
        return self.__ca.get_g()

    def _get_y_ca(self, w):
        return self.__ca.get_y(w=w)

    def _get_z_ca(self, w):
        return self.__ca.get_z(w=w)

    # A DC-block / bypass capacitor.
    def set_c_cb(self, val):
        self.__cb.set_c(val)
        return self

    def get_c_cb(self):
        return self.__cb.get_c()

    def set_g_cb(self, val):
        self.__cb.set_g(val)
        return self

    def get_g_cb(self):
        return self.__cb.get_g()

    def _get_y_cb(self, w):
        return self.__cb.get_y(w=w)

    def _get_z_cb(self, w):
        return self.__cb.get_z(w=w)

    # An RF-block / RF-chocke / DC-feed inductor.
    def set_l_lb(self, val):
        self.__lb.set_l(val)
        return self

    def get_l_lb(self):
        return self.__lb.get_l()

    def set_r_lb(self, val):
        self.__lb.set_r(val)
        return self

    def get_r_lb(self):
        return self.__lb.get_r()

    def _get_y_lb(self, w):
        return self.__lb.get_y(w=w)

    def _get_z_lb(self, w):
        return self.__lb.get_z(w=w)

    # The end of "CurrGenLE" class.


class FormCirLE:
    """
    An add-on that contains forming subcircuit components: cf, lf.
    It can be used in class E and EF2 power amplifiers.
    """

    def __init__(self):
        self.__cf = plc.Capacitor()
        self.__lf = plc.Inductor()

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # A forming subcircuit's capacitor.
    def set_c_cf(self, val):
        self.__cf.set_c(val)
        return self

    def get_c_cf(self):
        return self.__cf.get_c()

    def set_g_cf(self, val):
        self.__cf.set_g(val)
        return self

    def get_g_cf(self):
        return self.__cf.get_g()

    def _get_y_cf(self, w):
        return self.__cf.get_y(w=w)

    def _get_z_cf(self, w):
        return self.__cf.get_z(w=w)

    # A forming subcircuit's inductor.
    def set_l_lf(self, val):
        self.__lf.set_l(val)
        return self

    def get_l_lf(self):
        return self.__lf.get_l()

    def set_r_lf(self, val):
        self.__lf.set_r(val)
        return self

    def get_r_lf(self):
        return self.__lf.get_r()

    def _get_y_lf(self, w):
        return self.__lf.get_y(w=w)

    def _get_z_lf(self, w):
        return self.__lf.get_z(w=w)

    # The end of "FormCirLE" class.


class SerCirLE:
    """
    An add-on that contains series resonant subcircuit components: cs, ls.
    It is a suppressor of a chosen voltage harmonic at the active device.
    It can be used in class F and EF2 power amplifiers.
    """

    def __init__(self):
        self.__cs = plc.Capacitor()
        self.__ls = plc.Inductor()

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # A series resonant circuit's capacitor.
    def set_c_cs(self, val):
        self.__cs.set_c(val)
        return self

    def get_c_cs(self):
        return self.__cs.get_c()

    def set_g_cs(self, val):
        self.__cs.set_g(val)
        return self

    def get_g_cs(self):
        return self.__cs.get_g()

    def _get_y_cs(self, w):
        return self.__cs.get_y(w=w)

    def _get_z_cs(self, w):
        return self.__cs.get_z(w=w)

    # A series resonant circuit's inductor.
    def set_l_ls(self, val):
        self.__ls.set_l(val)
        return self

    def get_l_ls(self):
        return self.__ls.get_l()

    def set_r_ls(self, val):
        self.__ls.set_r(val)
        return self

    def get_r_ls(self):
        return self.__ls.get_r()

    def _get_y_ls(self, w):
        return self.__ls.get_y(w=w)

    def _get_z_ls(self, w):
        return self.__ls.get_z(w=w)

    _FoundWsup = namedtuple(
        'FoundWsup', 'success status message w_sup')

    def _find_w_sup(self):
        """
        Find an angular frequency value that will be suppressed by this
        subcircuit.

        Returns
        -------
        result : FoundWsup
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if the calculation is successful.
                Otherwise it is "False".
            status : int
                An error code. It is "0" if a calculation is successful.
            message : str
                A message that contains the details about a calculation
                result.
            w_sup : float
                The value of a suppressed angular frequency.
                If (success == False), then it equals "0.0".
        """
        c_cs = self.get_c_cs();    g_cs = self.get_g_cs()
        l_ls = self.get_l_ls()
        # It is better to use the radicand check here than
        # (l_ls > c_cs/g_cs**2) condition check.
        # It is better to check a radicand itself than some conditions that
        # mathimatically may lead to the negative radicand. This is because of
        # floating point numbers evaluations in computers. A condition can be
        # "True", but the radican is a very small negative number which will
        # cause an exception throwing.
        radicand = 1.0/(c_cs*l_ls) - (g_cs/c_cs)**2
        if radicand < 0.0:
            message = \
                f"Cannot find a suppressed angular frequency value 'w_sup'.\n" \
                f"Radicand is less than zero: {radicand}.\n" \
                f"A conductance value of the capacitor is too high: {g_cs}."
            return self._FoundWsup(
                success=False, status=1, message=message, w_sup=0.0)
        w_sup = np.sqrt(radicand)
        message = "A suppressed angular frequency has been successfully found."
        return self._FoundWsup(
            success=True, status=0, message=message, w_sup=w_sup)

    _SerCirTuning = namedtuple(
        'SerCirTuning', 'success status message w_sup c_cs l_ls')

    def _tune_sercir(self, *, w_sup=None, c_cs=None, l_ls=None):
        """
        Tune the series resonant subcircuit.
        This subcircuit is used to suppress a certain harmonic of the active
        device voltage.
        There are 3 acceptable combinations of the method parameters:
        1. "w_sup", "c_cs". An "l_ls" value will be calculated automatically.
        2. "w_sup", "l_ls". A "c_cs" value will be calculated automatically.
        3. "c_cs", "l_ls". Manual setting.
        Only these couples of parameters can be used simultaneously.
        The usage of "w_sup", "c_cs", and "l_ls" values in one method call
        will cause an exception throwing.
        The usage of only a "w_sup" value will cause an exception throwing.
        If a tuning has been failed, the original "c_cs" and "l_ls" values
        will be restored.

        Parameters
        ----------
        w_sup : float
            The angular frequency of a harmonic to suppress.
            w_sup = h_sup*w_wrk, h_sup > 0, where
            h_sup - a harmonic number to suppress,
            w_wrk - a working angular frequency.
        c_cs : float
            Capacitance of the series resonant subcircuit.
        l_ls : float
            Inductance of the series resonant subcircuit.

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
            message : str
                A message that contains the description of a result.
            w_sup : float
                The angular frequency of a harmonic to suppress.
            c_cs : float
                Capacitance of the series resonant subcircuit.
            l_ls : float
                Inductance of the series resonant subcircuit.

        Notes
        -----
        Using an angular frequency "w_sup" instead of "c_cs" or "l_ls" provides
        the ability to automatically calculate the missing value of "l_ls" or
        "c_cs" respectively. The series resonant subcircuit will be tuned to
        suppress the required harmonic. However, if a working frequency or some
        other parameters of the series resonant subcircuit will be changed
        after that, the tuning will be lost. So that, use this tuning method in
        the end, when all other power amplifier parameters has been set.
        The model does not store a "w_sup" value, it stores only "c_cs" and
        "l_ls" values.
        Setting "w_sup" and "c_ls" values is the only reliable way to avoid a
        failure. An "l_ls" value is always calculated unconditionally.
        Setting "w_sup" and "l_ls" values can cause a failure when "c_cs" is
        calculated.
        Setting "c_cs" and "l_ls" values can cause a failure when "w_sup" is
        calculated (to check that the suppressor really suppresses a harmonic).

        Since "g_cs" usually is a small number, the subcircuit can be
        roughly represented as a classical series resonant circuit. In that
        case:
        increasing "l_ls" increases Q factor;
        increasing "c_cs" decreases Q factor.
        """
        if (w_sup is not None) and (c_cs is not None) and (l_ls is not None):
            raise RuntimeError(
                "Ambiguous series resonant subcircuit tuning.\n"
                "An attempt to set all 3 parameters.")
        # The original (before an execution of the algorithm) values of
        # "c_cf" and "l_lf".
        c_cs_orig = self.get_c_cs()
        l_ls_orig = self.get_l_ls()
        def restore_sc():
            self.set_c_cs(c_cs_orig).set_l_ls(l_ls_orig)
        # The original values of the tunable parameters will be restored if
        # a tuning procedure will fail.
        with ScopeGuardManual(restore_sc) as sgm_sc:
            # Setting "c_cs" and/or "l_ls".
            if w_sup is not None:
                # Convert "w_sup" into "float" instead of a check that it of
                # a numerical type.
                w_sup = float(w_sup)
                g_cs = self.get_g_cs()
                if c_cs is not None:  # "w_sup" and "c_cs" are defined.
                    l_ls = SerCirLE._l_ls_from(
                        w_sup=w_sup, c_cs=c_cs, g_cs=g_cs)
                    # print(f"SerCirLE._tune_sercir: l_ls={l_ls}")  # Test.
                    self.set_c_cs(c_cs)
                    self.set_l_ls(l_ls)
                    # It will not continue the execution flow even if "l_ls"
                    # is not "None" here.
                elif l_ls is not None:  # "w_sup" and "l_ls" are defined.
                    ccsres = SerCirLE.__find_c_cs(
                        w_sup=w_sup, g_cs=g_cs, l_ls=l_ls)
                    if not ccsres.success:
                        # No commit. The original "c_cs" & "l_ls" values will
                        # be restored.
                        return self._SerCirTuning(
                            success=False, status=1, message=ccsres.message,
                            w_sup=w_sup, c_cs=ccsres.c_cs, l_ls=l_ls)
                    self.set_c_cs(ccsres.c_cs)
                    self.set_l_ls(l_ls)
                else:  # Only "w_sup" is defined.
                    # No commit. The original "c_cs" & "l_ls" values will be
                    # restored.
                    raise RuntimeError(
                        "Ambiguous series resonant subcircuit definition.\n"
                        "Cannot use only a harmonic number.")
            else:  # "w_sup" is not defined.
                if c_cs is not None:            self.set_c_cs(c_cs)
                if l_ls is not None:            self.set_l_ls(l_ls)
                wsres = self._find_w_sup()
                w_sup = wsres.w_sup
                if not wsres.success:
                    # No commit. The original "c_cs" & "l_ls" values will be
                    # restored.
                    return self.__self._SerCirTuning(
                        success=False, status=2, message=wsres.message,
                        w_sup=w_sup,
                        c_cs=self.get_c_cs(), l_ls=self.get_l_ls())
            # The solution is correct. Commit the solution.
            sgm_sc.commit()
        return self._SerCirTuning(
            success=True, status=0,
            message="Parameters has been successfully set.",
            w_sup=w_sup, c_cs=self.get_c_cs(), l_ls=self.get_l_ls())

    __FoundCcs = namedtuple('FoundCcs', 'success status message c_cs')

    @classmethod
    def __find_c_cs(cls, *, w_sup, g_cs, l_ls):
        """
        Find a "c_cs" value using a set of parameters.

        Parameters
        ----------
        w_sup : float
            The angular frequency of a harmonic to suppress.
        g_cs : float
            Parasitic conductance of the series resonant subcircuit's
            capacitor.
        l_ls : float
            Inductance of the series resonant subcircuit.

        Returns
        -------
        result : FoundCcs
            success : bool
                A flag that is "True" if a "c_cs" value has been found and
                "False" otherwise.
            status : int
                An error code.
                0 - a "c_cs" value has been successfully found.
                1 - cannot find a "c_cs" value because a "g_cs" value is too
                    high.
            message : str
                A message that contains the description of a result.
            c_cs : float
                Capacitance of the series resonant subcircuit.
        """
        # The equivalent circuit's capacitance. Note: l_eqv == l_ls.
        c_eqv = 1.0/(l_ls*w_sup**2)
        # Calculation and setting a "c_cs" value.
        radicand = 1.0 - (2.0*g_cs/(w_sup*c_eqv))**2
        if radicand < 0.0:
            message = \
                f"Cannot find a required capacitance value." \
                f"Radicand is less than zero: {radicand}.\n" \
                f"A conductance value of the capacitor is too high: {g_cs}."
            return cls.__FoundCcs(
                success=False, status=1, message=message, c_cs=np.nan)
        c_cs = 0.5*c_eqv*(1.0 + np.sqrt(radicand))
        message = "A capacitance value has been successfully found."
        return cls.__FoundCcs(
            success=True, status=0, message=message, c_cs=c_cs)

    @staticmethod
    def _l_ls_from(*, w_sup, c_cs, g_cs):
        """
        Calculate an "l_ls" value from a set of parameters.

        Parameters
        ----------
        w_sup : float
            The angular frequency of a harmonic to suppress.
        c_cs : float
            Capacitance of the series resonant subcircuit.
        g_cs : float
            Parasitic conductance of the series resonant subcircuit's
            capacitor.

        Returns
        -------
        l_ls : float
            Inductance of the series resonant subcircuit.
        """
        c_eqv = c_cs + ((g_cs/w_sup)**2)/c_cs
        # print(f"SerCirLE._l_ls_from: w_sup={w_sup}, c_cs={c_cs}, " \
        #       f"g_cs={g_cs}, c_eqv={c_eqv}")  # Test.
        return 1.0/(c_eqv*w_sup**2)

    # The end of "SerCirLE" class.


class ParCirLE:
    """
    An add-on that contains parallel resonant subcircuit components: cp, lp.
    """

    def __init__(self):
        raise NotImplementedError

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # The end of "ParCirLE" class.


class NegFbkLE:
    """
    An add-on that contains a negative feedback capacitor: cn.
    It can be used in class A, B, C, F power amplifiers.
    """

    def __init__(self):
        self.__cn = plc.Capacitor()

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # A negative feedback capacitor.
    def set_c_cn(self, val):
        self.__cn.set_cn(val)
        return self

    def get_c_cn(self):
        return self.__cn.get_c()

    def set_g_cn(self, val):
        self.__cn.set_g(val)
        return self

    def get_g_cn(self):
        return self.__cn.get_g()

    def _get_y_cn(self, w):
        return self.__cn.get_y(w=w)

    def _get_z_cn(self, w):
        return self.__cn.get_z(w=w)

    # The end of "NegFbkLE" class.
