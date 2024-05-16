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
Some common load networks (LNs).

About the file name.
Some of the models here do not represent any certain circuit and therefore can
be named as abstract. The file also contains abstract base classes for all LNs
in this library. So that this name has been chosen to be ambiguous on purpose.

Variants of LNs can be found in:
1. "Switchmode RF and Microwave Power Amplifiers", Andrei Grebennikov,
   Nathan O. Sokal, Marc J. Franco, 2nd ed., 2012.
2. "Resonant Power Converters", Marian K. Kazimierczuk and Dariusz Czarkowski,
   2nd ed., 2011.
3. "RF Power Amplifiers", Marian K. Kazimierczuk, 2008.

The main types of the LN models:
1. Stable. Models of this type does not have a built-in tuning method
   ("_tune") which provides the ability to tune the LN at a certain frequency.
2. Tunable. Models of this type can be tuned at a certain frequency.
There is also a special case of "Stable" LN called "Custom". It can store a
user defined LN model.
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

# Provide specific internals of this module when importing using:
# from _loadabstract import *
__all__ = [
    'ZBar',
    'ZLaw',
    'Custom'
]


# -- Abstract base classes of load networks -----------------------------------

# Note: it is also the base class of "Stable" load networks.
class LoadNet(metaclass=ABCMeta):
    """
    The common abstract base class of load networks.

    Fields (attributes) that must be in the heirs:
    _name : str
        'name' parameter. It must be defined from the outside without a setter.

    Methods that must be present at the final heirs:
    set_params(self, **kwargs)
    get_params(self)
    get_y_in(self, w)
    get_z_in(self, w)
    """

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def get_name(self):
        """
        Get the name / type / alias of a load network.

        Returns
        -------
        name : str
            The name of a load network.
        """
        return self._name

    @abstractmethod
    def set_params(self, **kwargs):
        """A public method to set parameters of a load network."""
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        """A public method to get parameters of a load network."""
        raise NotImplementedError

    @abstractmethod
    def get_y_in(self, w):
        """A public method to get input admittance of a load network."""
        raise NotImplementedError

    @abstractmethod
    def get_z_in(self, w):
        """A public method to get input impedance of a load network."""
        raise NotImplementedError

    # The end of "LoadNet" class.


# The result of a tuning procedure.
TunRes = namedtuple('TunRes', 'success status message f_tun z_in_req z_in_h1')
# The fields:
# success : bool
#     A flag that tells if a tuning procedure is successful.
# status : int
#     An error code of the result of a tuning procedure.
# message : str
#     A message that tells about the result of a tuning procedure.
# f_tun : float
#     A tuning frequency.
# z_in_req : complex
#     Required input impedance.
# z_in_h1 : complex
#     Actual input impedance at a tuning frequency.


class Tunable(LoadNet):
    """
    The abstract base class of load networks (LNs) that can be tuned at a
    certain frequency to provide a required input impedance value.

    Methods that must be written in the heirs:
    _retrieve_params(self)
    _assign_params(self, **kwargs)
    _eval_y_in(self, w)
    _eval_z_in(self, w)

    Methods that can be overriden in the heirs:
    _tune(self)

    Notes
    -----
    The models of tunable LNs here are based on the concept of lazy
    evaluations. When parameters are changed through "set_params" method,
    the model remember that fact, but a tuning process occurs only on demand
    when it must give some parameters or electrical characteristics which can
    be obtained after the tuning process. I. e. "get_y_in", "get_z_in",
    "get_params" will start the tuning process. After tuning has done, further
    using of these methods will not invoke a tuning procedure again while the
    basic parameters are stable.
    A "get_params" call can be used as a force tuning method to check if a
    tuning procedure is successful.
    """

    def __init__(self):
        """
        A constructor of a "Tunable" object.
        """
        LoadNet.__init__(self)
        self.__changed = True  # A load network has been changed.
        self.__tunres = None  # The result of a tuning.
        self.__f_tun = 0.5/np.pi  # Required tuning frequency at which a load network have to work.
        self.__z_in_req = 1.0 + 1j*0.0  # Required input impedance value at a tuning frequency.

    def set_params(self, **kwargs):
        """
        Set parameters of a load network (LN).

        Parameters
        ----------
        kwargs : dict
            Parameters of a certain LN model. For example,
            f_tun : float, optional
                A tuning frequency.
            z_in_req : complex, optional
                A required input impedance value at a tuning frequency.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__changed = True
        f_tun = kwargs.pop('f_tun', None)
        z_in_req = kwargs.pop('z_in_req', None)
        self.__set_tun_params(f_tun=f_tun, z_in_req=z_in_req)
        self._assign_params(**kwargs)
        return self

    def get_params(self):
        """
        Get parameters of a load network (LN).
        Calling this method will invoke the LN tuning process if some of the LN
        parameters have been changed before.

        Returns
        -------
        result : Params
            A "namedtuple" that contains LN parameters.
        """
        self.__try_tune()
        return self._retrieve_params()

    def get_z_in(self, w):
        """
        Get an input impedance value of a load network at a given angular
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
        self.__try_tune()
        return self._eval_z_in(w)

    def get_y_in(self, w):
        """
        Get an input admittance value of a load network at a given angular
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
        self.__try_tune()
        return self._eval_y_in(w)

    @abstractmethod
    def _retrieve_params(self):
        """A protected method to get parameters of a load network."""
        raise NotImplementedError

    @abstractmethod
    def _assign_params(self, **kwargs):
        """A protected method to set parameters of a load network."""
        raise NotImplementedError

    @abstractmethod
    def _eval_y_in(self, w):
        """A protected method to get input admittance of a load network."""
        raise NotImplementedError

    @abstractmethod
    def _eval_z_in(self, w):
        """A protected method to get input impedance of a load network."""
        raise NotImplementedError

    @abstractmethod
    def _tune(self):
        """
        Tune a load network (LN) for working at a given frequency.
        The method does nothing by default.
        The method must be overriden if a LN requires a specific tuning
        procedure after changing its parameters.

        Returns
        -------
        tunres : TunRes
            The result of a tuning procedure. See "_get_tunres" to get more
            information.
        """
        raise NotImplementedError

    def _get_f_tun(self):
        """
        Get a required tuning frequency.

        Returns
        -------
        f : float
            A required tuning frequency.
        """
        return self.__f_tun

    def _get_w_tun(self):
        """
        Get a required tuning angular frequency.

        Returns
        -------
        w : float
            A required tuning angular frequency.
        """
        return 2*np.pi*self.__f_tun

    def _get_z_in_req(self):
        """
        Get a required input impedance value at a tuning frequency.

        Returns
        -------
        z_in_req : complex
            A required input impedance value at a tuning frequency.
        """
        return self.__z_in_req

    def _get_tunres(self):
        """
        Get the result of a tuning process.

        Returns
        -------
        tunres : TunRes
            A "namedtuple" that contains following fields:
            success : bool
                A flag that is "True" if a tuning procedure is successful, and
                "False" otherwise.
            status : int
                An error code of the result of a tuning procedure. It is "0"
                when a tuning is successful, or a non-zero number otherwise.
            message : str
                A message that tells about the result of a tuning procedure.
            f_tun : float
                A tuning frequency.
            z_in_req : complex
                A required value of input impedance at a tuning frequency.
            z_in_h1 : complex
                An actual value of input impedance at a tuning frequency.
        """
        return self.__tunres

    def __set_tun_params(self, *, f_tun=None, z_in_req=None):
        """
        Set required parameters of a tunable load network (LN).

        Parameters
        ----------
        f_tun : float
            A tuning frequency. It is the main frequency (the 1st
            harmonic frequency) at which a LN have to work.
        z_in_req : complex
            A required input impedance value at a tuning frequency.
        """
        # Can be of types "complex", "float", or "int".
        if z_in_req is not None:    self.__z_in_req = complex(z_in_req)
        # Can be of types "float" or "int".
        if f_tun is not None:       self.__f_tun = float(f_tun)

    def __try_tune(self):
        """
        Try to tune a load network (LN).
        If some of the LN parameters have been changed before, calling of this
        method will invoke a tuning procedure. It does nothing special
        otherwise, except returning of the saved information about the tuning.

        Returns
        -------
        tunres : TunRes
            The result of a tuning procedure. See "_get_tunres" to get more
            information.
        """
        # Note: in this code architecture a user can get info about tuning
        # using "get_params" method. The result of this method call is not
        # required anywhere and useless for now.
        if self.__changed:
            self.__tunres = self._tune()
            self.__changed = False
        return self.__tunres

    # The end of "Tunable" class.

# -- The end of abstract base classes of load networks ------------------------


# -- "Stable" load networks ---------------------------------------------------

class ZLaw(LoadNet):
    """
    Load circuit that is defined by its z(w) law.
    """

    _name = None
    Params = namedtuple('Params', 'name hold')

    def __init__(self):
        """
        A constructor of a "ZLaw" object.
        """
        LoadNet.__init__(self)
        self.__imp_func = None  # z(w) function. It also can be a functional object.

    def set_params(self, *, imp_func=None):
        """
        Set the parameters of the load network.
        In fact, set a law of input impedance dependency on angular frequency.

        Parameters
        ----------
        imp_func : callable z(w), optional
            A function that describes input impedance dependency on angular
            frequency.
        """
        if imp_func is not None:
            if not callable(imp_func):
                raise TypeError(
                    "Cannot set a z(w) impedance function.\n"
                    "An object must be callable.")
            self.__imp_func = imp_func

    def get_params(self):
        """
        Get parameters of the load network (LN).
        In fact, get information whether it contains a law of input impedance
        dependency on angular frequency or not.
        It does not provide any information about the internal parameters of
        the law.

        Returns
        -------
        result : Params
            A "namedtuple" that contains following fields:
            name : str
                A given name of the LN.
            hold : bool
                A flag that shows whether it contains a LN object or not.
        """
        if self.__imp_func is None:     hold = False
        else:                           hold = True
        return self.Params(name=self._name, hold=hold)

    def get_z_in(self, w):
        """
        Get an input impedance value.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        z_in : complex
            An input impedance value.
        """
        return self.__imp_func(w)

    def get_y_in(self, w):
        """
        Get an input admittance value.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y_in : complex
            An input admittance value.
        """
        return 1.0/self.get_z_in(w)

    # The end of "ZLaw" class.


class Custom(LoadNet):
    """
    A holder / wrapper of a custom load network.
    """

    _name = None
    Params = namedtuple('Params', 'name hold')

    def __init__(self):
        """
        A constructor of a "Custom" object.
        """
        LoadNet.__init__(self)
        self.__obj = None  # A load network (LN) object.
        self.__has_z = False  # A LN object has "get_z_in" method.
        self.__has_y = False  # A LN object has "get_y_in" method.

    def get_obj(self):
        """
        Get a reference to a load network object.

        Returns
        -------
        obj : class
            A load network object.
        """
        return self.__obj

    def set_params(self, *, obj=None):
        """
        Set a load network (LN) object.

        Parameters
        ----------
        obj : class, optional
            A load network object.

        Notes
        -----
        A LN object must have at least one of these methods:
            get_y_in(self, w)
            get_z_in(self, w)
        It is presumed that setting and getting operations on LN parameters
        are fulfilled directly through the LN interface.
        """
        if obj is not None:
            # Check that an object has at least one of the necessary methods.
            if hasattr(obj, 'get_z_in'):    self.__has_z = True
            else:                           self.__has_z = False
            if hasattr(obj, 'get_y_in'):    self.__has_y = True
            else:                           self.__has_y = False
            if not (self.__has_z or self.__has_y):
                raise AttributeError(
                    "A load network object must have at least one of these methods:\n"
                    "    get_y_in(self, w)\n"
                    "    get_z_in(self, w)")
            self.__obj = obj

    def get_params(self):
        """
        Get parameters of a load network (LN).
        In fact, get information whether it contains a LN object or not.
        It does not provide any information about the internal parameters of
        the LN object.

        Returns
        -------
        result : Params
            A "namedtuple" that contains following fields:
            name : str
                A given name of the load network.
            hold : bool
                A flag that shows whether it contains a LN object or not.
        """
        if self.__obj is None:  hold = False
        else:                   hold = True
        return self.Params(name=self._name, hold=hold)

    def get_z_in(self, w):
        """
        Get an input impedance value.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        z_in : complex
            An input impedance value.
        """
        if self.__has_z:    return self.__obj.get_z_in(w)
        else:               return 1.0/self.__obj.get_y_in(w)

    def get_y_in(self, w):
        """
        Get an input admittance value.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y_in : complex
            An input admittance value.
        """
        if self.__has_y:    return self.__obj.get_y_in(w)
        else:               return 1.0/self.__obj.get_z_in(w)

    # The end of "Custom" class.


# Note: though it could be a "Tunable" heir, it is more convenient to have it
# separated with some unique parameter names.
class ZBar(LoadNet):
    """
    It is a load network that has a certain value of input impedance only at a
    central working frequency and near it.
    On the other frequencies it gives some other impedance value with zero
    imaginary part.

    If a value of the "highness" parameter is positive and high enought, it can
    be considered as an (almost) ideal filter of the 1st voltage harmonic.
    It has high impedance only at a central working frequency and near it.
    On the other frequencies, it gives very low resistance.

    If a value of the "highness" parameter is negative and low enought, it can
    be considered as an (almost) ideal filter of the 1st current harmonic.
    It has low impedance only at a central working frequency and near it.
    On the other frequencies, it gives very high resistance.

    Real part of the impedance at a central working frequency must be greater
    than zero.

    An equivalent circuit
    ---------------------

    o----
        |
    in  z
        |
    o----

    Notes
    -----
    This load representation is very useful as the first order abstraction for
    power amplifier studies and finding its pure behaviour patterns.

    High positive "highness" mode.
    It selects the 1st harmonic of voltage on it. I. e. it gives an (almost)
    ideal sinusoidal voltage on the input at the main frequency if there is a
    current component at that frequency.
    It also can be associated with an almost ideal parallel resonant circuit
    that has a very high quality factor with some effective load resistance. It
    consumes energy, but it is also an almost ideal voltage bandpass filter.

    Low negative "highness" mode.
    It selects the 1st harmonic of current through it. I. e. it gives an
    (almost) ideal sinusoidal input current at the main frequency if there is a
    voltage component at that frequency.
    It also can be associated with an almost ideal series resonant circuit that
    has a very high quality factor with some effective load resistance. It
    consumes energy, but it is also an almost ideal voltage bandstop filter.
    """

    _name = None  # Settable from the outside.
    Params = namedtuple('Params', 'name highness f_cent z_cent z_side')

    def __init__(self):
        """
        "ZBar" instance constructor.
        """
        # A "z_side" value depends on "z_cent" and "highness" values.
        # It will be recalculated automatically each time when the basic
        # parameters are changed.
        LoadNet.__init__(self)
        self.__highness = 6  # The exponent used to set a "z_side" value.
        self.__f_cent = 0.5/np.pi  # Central working frequency.
        self.__z_cent = 1.0 + 1j*0.0  # Input impedance at a central frequency.
        self.__z_side = None  # Impedance at other (side) frequencies.
        self.__setup_z_side()  # Setup a "z_side" value.

    def set_params(self, *, highness=None, f_cent=None, z_cent=None):
        """
        Set the parameters of the load network.

        Parameters
        ----------
        f_cent : float, optional
            A central frequency.
        z_cent : complex, optional
            An input impedance value at a central frequency.
        highness : int, optional
            An exponent value that is used to calculate a value of
            impedance at non-central frequencies.

        Notes
        -----
        Examples of sets of parameters:
            f_cent=53e3, z_cent=(1.1 - 1j*0.2), highness=6
            z_cent=5.6, f_cent=107e3

        If "highness" and / or "z_cent" values are set, it invokes a "z_side"
        recalculation procedure.
        """
        # Recalculate a "z_side" value if "z_cent" and / or "highness" values
        # are changed.
        recalc_z_side = False

        if f_cent is not None:
            self.__f_cent = float(f_cent)

        if z_cent is not None:
            self.__z_cent = complex(z_cent)
            recalc_z_side = True

        if highness is not None:
            self.__highness = int(highness)
            recalc_z_side = True

        if recalc_z_side:
            self.__setup_z_side()

    def get_params(self):
        """
        Get the parameters of the load network (LN).

        Returns
        -------
        result : Params
            A "namedtuple" that contains parameters of the LN.
        """
        return self.Params(name=self._name, highness=self.__highness,
                           f_cent=self.__f_cent,
                           z_cent=self.__z_cent, z_side=self.__z_side)

    def get_z_in(self, w):
        """
        Get an input impedance value.
        It has a specific impedance value only at a central working frequency
        and near it. On the other frequencies it gives some other impedance
        value with the imaginary part equals zero.
        (z_in == z_cent)  if  (abs(w - w_cent) < 0.5*w_cent),
        (z_in == z_side)  otherwise, where
        z_cent - an impedance value at a central frequency,
        z_side - an impedance value at non-central (side) frequencies,
        w_cent - a central angular frequency,
        w - an angular frequency.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        z_in : complex
            An input impedance value.
        """
        w_cent = 2*np.pi*self.__f_cent
        if abs(w - w_cent) < 0.5*w_cent:  # The central angular frequency.
            return self.__z_cent
        else:  # Other frequencies.
            return self.__z_side  # Very low resistance if "highness" is high.

    def get_y_in(self, w):
        """
        Get an input admittance value.
        It has a specific admittance value only at a central working frequency
        and near it. On the other frequencies it gives some other admittance
        value with the imaginary part equals zero.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y_in : complex
            An input admittance value.
        """
        return 1.0/self.get_z_in(w)

    def __setup_z_side(self):
        """
        Set up a value of the side frequency impedance "z_side". I. e. it
        changes a value of impedance at non-central frequencies.
        It uses "z_cent" and "highness" values. It must be called each
        time when at least one of these values are changed.

        The formula is
        z_side = real(z_cent)*10^(-highness) + 1j*0, where
        z_cent - an impedance value at a central working frequency,
        highness - a coefficient.
        """
        self.__z_side = complex(10**(-self.__highness)*np.real(self.__z_cent))

    # The end of "ZBar" class.


class ZTable(LoadNet):
    """
    To do...

    Load circuit that is defined by z(w) table.
    It uses spline interpolation (Which order?).
    """

    _name = None

    def __init__(self):
        Tunable.__init__(self)
        raise NotImplementedError

    # The end of "ZTable" class.

# -- The end of "Stable" load networks ----------------------------------------
