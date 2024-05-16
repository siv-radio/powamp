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
PowAmp - a library to perform calculations of power amplifiers.

Precaution to the users of this library.
Use only public interface. Do not use protected (names started with an
underscore "_") and private (names started with two underscores "__")
class members. There is no guarantee that they will not be changed in
the future in the same version branch of the library.
"""

from pathlib import Path

from . import _class_e_le
from . import _class_ef_le
from .actdev import actdevs
from .loadnet import loads
from .version import PKG_VERSION

__version__ = PKG_VERSION

__all__ = ['__version__', 'make_powamp', 'show_version', 'show_pa_models']

models = {
    'class-e:le': _class_e_le.Tuner,
    'class-ef:le': _class_ef_le.Tuner
}

# Write the aliases into the models of the power amplifiers.
# Warning: this technique can cause trouble if the "for" loop will not be
# applied onto the related classes. It can occure if the PA models will be
# used without using this module.
for alias, PAClass in models.items():
    PAClass._name = alias
    # print(alias)  # Test.
del alias, PAClass


def make_powamp(name):
    """
    Make a requested power amplifier model.

    Parameters
    ----------
    name : str
        The name of a PA model. Available models:
        'class-e:le' - class E with lumped elements.
        'class-ef:le' - class EFx with lumped elements.

    Returns
    -------
    pa : Tuner
        A PA model with a built-in tuner (a set of methods to tune the PA).
    """
    if not isinstance(name, str):
        raise TypeError("Unexpected power amplifier name type: {0}".format(type(name)))
    PAClass = models.get(name)
    if PAClass is None:
        raise ValueError("Unknown power amplifier name: {0}".format(name))
    return PAClass()
    # The end of "make_powamp" function.


def get_version():
    """
    Get the current version of the package.

    Returns
    -------
    version : str
        The current version of the package.
    """
    return __version__
    # The end of "get_version" function.


def get_path():
    """
    Get an absolute path to the parent directory where the "powamp" regular
    package is located.

    Returns
    -------
    path : pathlib.PurePath
        An absolute path to the regular package.

    Notes
    -----
    It can be useful to quickly locate where the package is in order to get
    the access to the source code and examples.
    """
    return Path(__file__).parents[1]
    # The end of "get_path" function.
