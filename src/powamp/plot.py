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
Plotting functions to display simulation data results.
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['waveform', 'spectrum']


def pretty_pa_name(name):
    """
    Give a pretty power amplifier (PA) name from a given one.
    "Pretty" here means that it is more human-friendly than the original.

    Parameters
    ----------
    name : str
        A given name from "Params.name" field of a PA model.

    Returns
    -------
    pretty_name : str
        A pretty PA name that is made by conversion from a given one.

    Notes
    -----
    The required PA name format at the input:
    class-<letter(s)>:<element base>
    Examples:
    "class-e:le" - class E PA with lumped elements;
    "class-e:tl" - class E PA with transmission lines;
    "class-ef:le" - class EFx PA with lumped elements.
    """
    if name.startswith("class"):
        _, tokens = name.split('-')
        class_name, elem_base = tokens.split(':')
        class_name = class_name.upper()
        if elem_base == "le":
            elem_base = "lumped elements"
        elif elem_base == "tl":
            elem_base = "transmission lines"
        else:
            raise ValueError("Unknown element base: {0}".format(name))
        return " ".join(["class", class_name, "power amplifier with", elem_base])
    else:
        return name
    # The end of "pretty_pa_name" function.


def waveform(simdata):
    """
    Plot time dependencies.

    Parameters
    ----------
    simdata : SimData
        A simulation data object that contains the characteristics of a power
        amplifier.
    """
    name = pretty_pa_name(simdata.get_name())
    # Parameters: 2 rows, 1 column; width 8, height 8.
    fig, (axv, axi) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle("Time domain characteristics of a " + name)
    tsamps = simdata.get_time_samples()
    for probes, tdata in simdata.iterate_characs(charac='v', domain='t', group_by='data'):
        axv.plot(tsamps, tdata, label=", ".join(probes))
    for probes, tdata in simdata.iterate_characs(charac='i', domain='t', group_by='data'):
        axi.plot(tsamps, tdata, label=", ".join(probes))
    axv.set_xlabel("t, s")
    axv.set_ylabel("v, V", rotation='horizontal')
    axv.yaxis.set_label_coords(0.0, 1.05)  # A way to move the "ylabel" a little bit to the right.
    axv.legend(title="Voltages", bbox_to_anchor=(1.16, 1), loc='upper right')
    axv.grid()
    axi.set_xlabel("t, s")
    axi.set_ylabel("i, A", rotation='horizontal')
    axi.yaxis.set_label_coords(0.0, 1.05)
    axi.legend(title="Currents", bbox_to_anchor=(1.16, 1), loc='upper right')
    axi.grid()
    fig.show()  # Show the figure.
    # The end of "waveform" function.


def spectrum(simdata):
    """
    Plot spectrums.

    The "Figure" looks like:
    -------------------
    | v00 v01 i00 i01 |
    | v10 v11 i10 i11 |
    | ... ... ... ... |
    | vx0 vx1 ix0 ix1 |
    -------------------
    where vrc and irc - related "Axes" objects, vrc - voltage spectrum,
    irc - current spectrum, r - row index, c - column index.

    Parameters
    ----------
    simdata : SimData
        A simulation data object that contains the characteristics of a power
        amplifier.
    """
    name = pretty_pa_name(simdata.get_name())
    # Assumption: there are similar numbers of voltages and currents.
    # Voltages go first, then currents follow.
    numv = simdata.size_of(charac='v', group='data')  # Number of voltages.
    numi = simdata.size_of(charac='i', group='data')  # Number of currents.
    cols = 4  # Number of columns in a figure. It must be an even number.
    rows = max(int(np.ceil(numv/2)), int(np.ceil(numi/2)))
    fbins = simdata.get_freq_bins()  # Frequencies.
    # An example of a set of parameters in case of the class E PA:
    # 8 rows, 4 columns; width 16, height 8.
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Frequency domain characteristics of a " + name)

    # Voltages.
    for idx, (probes, fdata) in enumerate(simdata.iterate_characs(charac='v', domain='f', group_by='data')):
        r = idx % rows  # Row in "axs".
        c = idx//rows  # Column in "axs".
        pos = (r, c)
        names = ", ".join(probes)
        mags = np.abs(fdata)
        ax = axs[pos]  # Reference to an "Axis" object.
        ax.stem(fbins, mags)
        if np.all(mags > 0.0):  ax.set_yscale('log')
        else:                   ax.set_yscale('linear')
        ax.set_xlabel("f, Hz")
        ax.set_ylabel("v, V", rotation='horizontal')
        ax.yaxis.set_label_coords(0.0, 1.05)
        ax.set_title("Voltages: " + names, fontsize='medium')
        ax.grid()
    hena = (rows*cols)//2  # Half of the number of elements in "axs".
    for idx in range(numv, hena):  # Remove unused voltage "Axes" in "axs".
        r = idx % rows  # Row in "axs".
        c = idx//rows  # Column in "axs".
        pos = (r, c)
        axs[pos].remove()

    # Currents
    fcc = cols//2  # The first current (i. e. filled with spectrums of the currents) column in "axs".
    for idx, (probes, fdata) in enumerate(simdata.iterate_characs(charac='i', domain='f', group_by='data')):
        r = idx % rows  # Row in "axs".
        c = fcc + idx//rows  # Column in "axs".
        pos = (r, c)
        names = ", ".join(probes)
        mags = np.abs(fdata)
        ax = axs[pos]
        ax.stem(fbins, mags)
        if np.all(mags > 0.0):  ax.set_yscale('log')
        else:                   ax.set_yscale('linear')
        ax.set_xlabel("f, Hz")
        ax.set_ylabel("i, A", rotation='horizontal')
        ax.yaxis.set_label_coords(0.0, 1.05)
        ax.set_title("Currents: " + names, fontsize='medium')
        ax.grid()
    for idx in range(numi, hena):  # Remove unused currents "Axes" in "axs".
        r = idx % rows  # Row in "axs".
        c = fcc + idx//rows  # Column in "axs".
        pos = (r, c)
        axs[pos].remove()
    fig.show()

    # The end of "spectrum" function.
