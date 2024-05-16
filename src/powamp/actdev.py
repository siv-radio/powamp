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
Models of active devices (ADs).
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

__all__ = ['ADManager']


# If an AD model does not have a linear representation, a common nonlinear
# representation will be used.
class ADManager:
    def __init__(self):
        self.__ad = None  # AD model, object.
        # A public method. The procedure depends on an AD model.
        self.try_get_linear_ij = None  # A method reference.
        # A flag that tells if an AD has a linear representation.
        self.__has_linear_repr = False

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def get_ad(self):
        return self.__ad

    def config_ad(self, **kwargs):
        ad_name = kwargs.pop('name', None)  # Try to extract an AD name.
        if (ad_name is None) and (self.__ad is None):
            raise RuntimeError(
                "No active device (AD) has been set.\n"
                "Cannot set parameters while an AD name is undefined.")
        if ad_name is not None:  # Check if there is an AD name.
            self.__make_ad(ad_name)  # Create a new AD if the name is given.
        self.__ad.set_params(**kwargs)  # Pass parameters into an AD.
        return self

    def get_ad_config(self):
        return self.__ad.get_params()

    def get_name(self):
        if self.__ad is not None:   return self.__ad.get_name()
        else:                       return 'empty'

    def get_ij(self, v_ad_t):
        return self.__ad.get_ij(v_ad_t)

    def is_linear(self):
        return self.__ad.is_linear()

    def has_linear_repr(self):
        return self.__has_linear_repr

    def __make_ad(self, ad_name):
        if isinstance(ad_name, str):  # If "ad_name" name is correct.
            ADClass = actdevs.get(ad_name)  # Try to find an AD by its name.
            if ADClass is None:  # Wrong string content.
                raise ValueError("An unknown active device name: {0}.".format(ad_name))
            self.__ad = ADClass()  # Call a constructor and return the object.
            # Set "try_get_linear_ij" behaviour.
            if hasattr(self.__ad, 'get_linear_ij'):
                self.try_get_linear_ij = self.__ad.get_linear_ij
                self.__has_linear_repr = True
            else:
                self.try_get_linear_ij = self.__ad.get_ij
                self.__has_linear_repr = False
        else:  # Wrong type.
            raise TypeError("An unacceptable type of an active device name: {0}.".format(type(ad_name)))

    # The end of "ADManager" class.


# -- Models of active devices -------------------------------------------------


class ActDev(metaclass=ABCMeta):
    """The abstract base class of all active device models."""

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def get_name(self):
        return self._name

    def is_linear(self):
        return self._is_linear

    # Abstract methods.

    @abstractmethod
    def get_ij(self, v_ad_t):
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_linear_ij(self):
        raise NotImplementedError

    # The end of "ActDev" class.


class Bidirect(ActDev):
    def __init__(self):
        ActDev.__init__(self)

    def get_ij(self, v_ad_t):
        s_len = len(v_ad_t)  # Number of time samples.
        dt_n = 1.0/s_len  # Normalized time sample duration.
        i_ad_t = np.empty((s_len,))  # A current vector in time domain.
        j_ad_t = np.empty((s_len,))  # A Jacobian vector in time domain.
        for s in range(s_len):
            t_n = dt_n*s  # Normalized time / turn.
            i_ad_t[s], j_ad_t[s] = self._law.get_ij(t_n, v_ad_t[s])
        return i_ad_t, j_ad_t

    # The end of "Bidirect" class.


class Forward(Bidirect):
    def __init__(self):
        ActDev.__init__(self)

    def get_ij(self, v_ad_t):
        s_len = len(v_ad_t)  # Number of time samples.
        dt_n = 1.0/s_len  # Normalized time sample duration.
        i_ad_t = np.empty((s_len,))  # A current vector in time domain.
        j_ad_t = np.empty((s_len,))  # A Jacobian vector in time domain.
        for s in range(s_len):
            if v_ad_t[s] >= 0.0:
                # Note: # It is important to use ">=" for better convergence.
                t_n = dt_n*s  # Normalized time / turn.
                i_ad_t[s], j_ad_t[s] = self._law.get_ij(t_n, v_ad_t[s])
            else:
                i_ad_t[s] = 0.0
                j_ad_t[s] = 0.0
        return i_ad_t, j_ad_t

    # The end of "Forward" class.


class Freewheel(Bidirect):
    def __init__(self):
        ActDev.__init__(self)
        self.__r_fwd = 0.1

    def _set_r_fwd(self, val):
        self.__r_fwd = float(val)
        return self

    def _get_r_fwd(self):
        return self.__r_fwd

    def get_ij(self, v_ad_t):
        g_fwd = 1.0/self.__r_fwd
        s_len = len(v_ad_t)  # Number of time samples.
        dt_n = 1.0/s_len  # Normalized time sample duration.
        i_ad_t = np.empty((s_len,))  # A current vector in time domain.
        j_ad_t = np.empty((s_len,))  # A Jacobian vector in time domain.
        for s in range(s_len):
            if v_ad_t[s] < 0.0:  # Freewheeling (anti-parallel) diode works.
                i_ad_t[s] = g_fwd*v_ad_t[s]
                j_ad_t[s] = g_fwd
            else:  # Forward conduction if it is the right time.
                t_n = dt_n*s  # Normalized time / turn.
                i_ad_t[s], j_ad_t[s] = self._law.get_ij(t_n, v_ad_t[s])
        return i_ad_t, j_ad_t

    # The end of "Freewheel" class.


class PulseConductor(metaclass=ABCMeta):
    """An active device with a duty cycle variable."""

    def __init__(self):
        self.__dc = 0.5  # This parameter can be non-common.

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def set_duty_cycle(self, val):
        self.__dc = float(val)
        return self

    def get_duty_cycle(self):
        return self.__dc

    @abstractmethod
    def get_ij(self, t_n, v):
        raise NotImplementedError

    # The end of "PulseConductor" class.


class Switch(PulseConductor):
    """
    This class is common for each AD of switching type. It contains common
    data fields and methods that are used for fast power amplifier (PA) tuning
    when an AD conducts current only in the forward direction (for example,
    class E PA).
    """

    def __init__(self):
        PulseConductor.__init__(self)
        self.__r_ad = 0.1

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def set_r_ad(self, val):
        self.__r_ad = float(val)
        return self

    def get_r_ad(self):
        return self.__r_ad

    # To implement "s" or "t" dependency, it must have info about "sp" or
    # "s_len" to get (s/sp) or (s/s_len) ratio and use it as noramalized time
    # for time dependencies.
    def get_ij(self, t_n, v):
        if t_n < self.get_duty_cycle():
            g_ad = 1.0/self.__r_ad
            return g_ad*v, g_ad
        else:
            return 0.0, 0.0

    # The end of "Switch" class.


class BidirectSwitch(Bidirect):
    """
    A switchmode active device (AD) that can equally conduct current in both
    directions when it is open.

    Equivalent circuit
    ------------------

    ----
       |
       |
       |
       |
     r_ad(t)
       |
       |
       |
       |
    ----

    r_ad(t) - AD resistance law.
    t - time.
    """

    _name = None  # Settable from the outside.
    _is_linear = True
    Params = namedtuple('Params', 'name linear dc r_ad')

    def __init__(self):
        Bidirect.__init__(self)
        self._law = Switch()

    def set_params(self, *, dc=None, r_ad=None):
        if dc is not None:      self._law.set_duty_cycle(dc)
        if r_ad is not None:    self._law.set_r_ad(r_ad)

    def get_params(self):
        return self.Params(name=self._name, linear=self._is_linear,
                           dc=self._law.get_duty_cycle(),
                           r_ad=self._law.get_r_ad())

    def get_linear_ij(self, v_ad_t):
        return self.get_ij(v_ad_t)

    # The end of "BidirectSwitch" class.


class ForwardSwitch(Forward):
    """
    A switchmode active device (AD) that can conduct current only in forward
    direction when it is open.

    Equivalent circuit
    ------------------

    ----
       |
       |
       |
    r_ad(v_ad, t)
       |
       v
       |
       |
       |
    ----

    r_ad(v_ad, t) - AD resistance law.
    v_ad - voltage on the AD.
    t - time.
    """

    _name = None  # Settable from the outside.
    _is_linear = False
    Params = namedtuple('Params', 'name linear dc r_ad')

    def __init__(self):
        Forward.__init__(self)
        self._law = Switch()

    def set_params(self, *, dc=None, r_ad=None):
        if dc is not None:      self._law.set_duty_cycle(dc)
        if r_ad is not None:    self._law.set_r_ad(r_ad)

    def get_params(self):
        return self.Params(name=self._name, linear=self._is_linear,
                           dc=self._law.get_duty_cycle(),
                           r_ad=self._law.get_r_ad())

    def get_linear_ij(self, v_ad_t):
        return Bidirect.get_ij(self, v_ad_t)

    # The end of "ForwardSwitch" class.


class FreewheelSwitch(Freewheel):
    """
    A switchmode active device (AD) that can conduct current in forward
    direction when it is open and also in backward direction at any time.

    Equivalent circuit
    ------------------

    ----
       |
       o------------
       |           |
    r_ad(v_ad, t)  ^
       |           |
       v       r_fwd(v_ad)
       |           |
       o------------
       |
    ----

    r_ad(v_ad, t) - AD resistance law.
    r_fwd(v_ad) - resistance law of a freewheeling diode in its forward
        conductive state.
    v_ad - voltage on the AD.
    t - time.
    """

    _name = None  # Settable from the outside.
    _is_linear = False
    Params = namedtuple('Params', 'name linear dc r_ad r_fwd')

    def __init__(self):
        Freewheel.__init__(self)
        self._law = Switch()

    def set_params(self, *, dc=None, r_ad=None, r_fwd=None):
        if dc is not None:       self._law.set_duty_cycle(dc)
        if r_ad is not None:     self._law.set_r_ad(r_ad)
        if r_fwd is not None:    self._set_r_fwd(r_fwd)

    def get_params(self):
        return self.Params(name=self._name, linear=self._is_linear,
                           dc=self._law.get_duty_cycle(),
                           r_ad=self._law.get_r_ad(),
                           r_fwd=self._get_r_fwd())

    def get_linear_ij(self, v_ad_t):
        return Bidirect.get_ij(self, v_ad_t)

    # The end of "FreewheelSwitch" class.


# It must be defined after the ADs.
actdevs = {
    'bidirect:switch': BidirectSwitch,
    'forward:switch': ForwardSwitch,
    'freewheel:switch': FreewheelSwitch
}

# Write the aliases into the models of the ADs.
# Warning: this technique can cause trouble if the "for" loop will not be
# applied onto the related classes. It can occure if the loop will be placed
# into a separate module and the AD models will be used without using that
# module.
for alias, ADClass in actdevs.items():
    ADClass._name = alias
    # print(alias) # Test.
del alias, ADClass

# -- The end of AD models -----------------------------------------------------
