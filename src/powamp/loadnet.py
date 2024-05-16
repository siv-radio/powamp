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
Some widespread load networks.
"""

from . import _loadabstract as abstract
from . import _loadlumpel as lumpel

__all__ = ['LoadManager']

loads = {
    'custom': abstract.Custom,
    'zbar': abstract.ZBar,
    'zlaw': abstract.ZLaw,
    'sercir:le': lumpel.SerCirLE,
    'parcir:le': lumpel.ParCirLE,
    'pinet:le': lumpel.PiNetLE,
    'teenet:le': lumpel.TeeNetLE
}

# Write aliases into the models of the load networks.
# Warning: this technique can cause trouble if the "for" loop will not be
# applied onto the related classes. It can occure if the LN models will be
# used without using this module.
for alias, LoadClass in loads.items():
    LoadClass._name = alias
    # print(alias) # Test.
del alias, LoadClass


class LoadManager:
    """
    A load network manager.
    It provides a common interface to work with different load network types.
    """

    def __init__(self):
        """
        Constructor of a load manager object.
        """
        self.__load = None

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def get_load(self):
        """
        Get a reference to a load network object.
        It is useful when working with a custom load.
        Avoid to use it with regular loads.

        Returns
        -------
        load : LoadNet heir class or just a user's class.
            A load network object.
        """
        # Useful for a custom load network, useless otherwise.
        # It can give a direct reference to a load network object.
        # If a load network is custom, then it can extract a reference
        # from its wrapper.
        if self.get_name() == 'custom':     return self.__load.get_obj()
        else:                               return self.__load

    # Note: it is not actually necessary to use "**kwargs" here.
    # "OneAD.config_load" also accepts "**kwargs". So that, it is possible
    # to use a "kwargs" "dict" here as an argument.
    def config_load(self, **kwargs):
        """
        Configure a load network.

        Parameters
        ----------
        kwargs : dict
            Parameters related to a certain load network.

        Returns
        -------
        "self" reference to the caller object.

        Examples
        --------
        Some examples of "kwargs":
        1.   {'name': 'zbar', 'mode': 'z', 'highness': 9,
              'f_cent': 100e3, 'z_cent': 5.0}
        1.1. {'f_cent': 200e3, 'z_cent': 4.0}
        2.   {'name': 'parcir:le', 'f_tun': 2e6, 'z_in_req': 4.0 + 0.1j,
              'q_eqv': 2.0, 'g_cp': 1e-6, 'r_lp': 1e-6}
        2.1. {'f_tun': 1.5e6, 'z_in_req': 4.0 + 0.2j, 'q_eqv': 2.5}
        3.   {'name': 'pinet:le', 'f_tun': 500e3, 'z_in_req': 8.0 + 0.5j,
              'midcoef': 0.75, 'g_ci': 1e-6, 'g_co': 1e-6, 'r_lm': 1e-6,
              'r_out': 5.0}
        3.1. {'f_tun': 700e3, 'z_in_req': 5.0 + 0.3j, 'midcoef': 0.6}
        """
        load_name = kwargs.pop('name', None)  # Try to extract a load name.
        if (load_name is None) and (self.__load is None):
            raise RuntimeError(
                "No load has been set.\n"
                "Cannot set parameters while a load type is undefined.")
        if load_name is not None:  # Check if there is a load name.
            self.__make_load(load_name)  # Create a new load if the name is given.
        self.__load.set_params(**kwargs)  # Pass parameters into a load.
        return self

    def get_load_config(self):
        """
        Get a load network configuration.

        Returns
        -------
        params : Params
            A "namedtuple" that contains parameters of a load network.
        """
        return self.__load.get_params()

    def get_z_in(self, w):
        """
        Get the input impedance of a load network.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        z : complex
            An impedance value.
        """
        return self.__load.get_z_in(w)

    def get_y_in(self, w):
        """
        Get the input admittance of a load network.

        Parameters
        ----------
        w : float
            An angular frequency.

        Returns
        -------
        y : complex
            An admittance value.
        """
        return self.__load.get_y_in(w)

    def get_name(self):
        """
        Get the name of a load network.

        Returns
        -------
        name : str
            The name of a load network.
        """
        if self.__load is not None:     return self.__load.get_name()
        else:                           return 'empty'

    def is_tunable(self):
        """
        Check if a load network (LN) is tunable.

        Returns
        -------
        result : bool
            The flag that is "True" if a LN is tunable and "False" otherwise.

        Notes
        -----
        A tunable load network object has a "_tune" method.
        """
        return hasattr(self.__load, '_tune')

    def __make_load(self, load_name):
        """
        Make a load network (LN) object.
        In case of a "custom" load, it makes a wrapper for a LN object.

        Parameters
        ----------
        load_name : str
        """
        if isinstance(load_name, str):  # If "load_name" type is correct.
            LoadClass = loads.get(load_name)  # Try to find a load by its name.
            if LoadClass is None:  # Wrong string content.
                raise ValueError("An unknown load name: {0}.".format(load_name))
            self.__load = LoadClass()  # Call a constructor and return the object.
        elif not isinstance(load_name, None):  # Wrong type.
            raise TypeError("An unacceptable type of a load name: {0}.".format(type(load_name)))

    # The end of "LoadManager" class.
