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
Simulation data basics.

About iterators and "yield" statement, see
"Iterators and Iterables in Python: Run Efficient Iterations",
by Leodanis Pozo Ramos, 2023.03.01,
https://realpython.com/python-iterators-iterables/

INI format to store parameters of a power amplifier.
Python v3.7.3 has a built-in parser for .ini files ("configparser" module).
References:
1. "INI file", on 2023.08.12,
   https://en.wikipedia.org/wiki/INI_file
2. "configparser - Work with Configuration Files", pymotw.com,
   by Doug Hellmann, on 2023.08.13,
   https://pymotw.com/3/configparser/

Comma separated values (CSV) files to store numerical simulation data.
Python v3.7.3 has a built-in parser for .csv files ("csv" module).
References:
1. "Python CSV: Read and Write CSV files", www.programiz.com, on 2023.08.09,
   https://www.programiz.com/python-programming/csv
2. "Working with csv files in Python", www.geeksforgeeks.org, 2023.03.24,
   https://www.geeksforgeeks.org/working-csv-files-python/
3. "Reading and Writing CSV Files in Python", realpython.com, by Jon Fincher,
   on 2023.08.09,
   https://realpython.com/python-csv/

Working with files and directories.
Python v3.7.3 has built-in tools for working with files and directories
("os.path", "tempfile", "shutil", "pathlib" and other modules).
References:
1. "Working With Files in Python", realpython.com, by Vuyisile Ndlovu,
   on 2023.08.11,
   https://realpython.com/working-with-files-in-python/
2. "Python's pathlib Module: Taming the File System", realpython.com,
   by Geir Arne Hjelle, 2023.04.17,
   https://realpython.com/python-pathlib/
3. "Reading and Writing Files in Python (Guide)", realpython.com,
   by James Mertz, on 2023.08.11,
   https://realpython.com/read-write-files-python/
"""

from collections import namedtuple
import configparser
import csv
from pathlib import Path

import numpy as np

from . import hbmath as hbm
from . import utils as pau

__all__ = ['SimDBManager', 'SimDataBase', 'add_charac_getters']


class CharacDB:
    """
    A database to store electrical characteristics.
    It can store voltages or currents in frequency and time domains.
    Each voltage and current data can have several probes (names), while each
    probe refers to only one data set.
    """

    __FTDP = namedtuple('FTDP', 'f t probes')  # Frequency & time data; probes.

    def __init__(self):
        """
        Constructor of an instance of the database.

        Fields
        ------
        __pr2id : dict of (str to int)
            A dictionary with a probe as a key and a data ID as a value.
        __ftdp : list of FTDP
            A list of frequency and time data sets, and probes.
            An index in the list is the data ID.
        """
        self.__pr2id = dict()  # Probe to data ID.
        self.__ftdp = list()  # Frequency and time data, and probes.

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def add(self, *, probes, fdata, tdata):
        """
        Add data into a database.

        Parameters
        ----------
        probes : tuple of str
            The names of a data set.
            For example, ('ad', 'ca'), (lf' 'out'), etc.
        fdata : ndarray of complex
            An array of harmonics.
        tdata : ndarray of float
            An array of time data samples.

        Returns
        -------
        "self" reference to the caller object.
        """
        probes = tuple(probes)
        data_id = len(self.__ftdp)
        self.__ftdp.append(self.__FTDP(f=fdata, t=tdata, probes=probes))
        for pr in probes:
            self.__pr2id[pr] = data_id
        return self

    def get_data(self, *, probe, domain):
        """
        Get a data set by its probe (name) in a selected domain.

        Parameters
        ----------
        probe : str
            The name of a required probe.
            For example, 'ad', 'cb', 'lf' 'out', etc.
        domain : str
            The short name of a required domain.
            'f' is for frequency; 't' is for time.

        Returns
        -------
        result : ndarray
            An array of the requested data.
            It gives a reference to the internal data, not a copy.
        """
        data_id = self.__pr2id.get(probe)
        if domain == 'f':
            return self.__ftdp[data_id].f
        elif domain == 't':
            return self.__ftdp[data_id].t
        else:
            raise ValueError("An unexpected domain: {0}.".format(domain))

    # Unused, unnecessary.
    def get_probes(self, *, probe):
        """
        Get all the probes that refer to the same data as the given probe.

        Parameters
        ----------
        probe : str
            The name of a required probe.
            For example, 'ad', 'cb', 'lf' 'out', etc.

        Returns
        -------
        probes : tuple of str
            A tuple of probes. For example, ('ad', 'ca'), ('lf', 'out'), etc.
        """
        data_id = self.__pr2id.get(probe)
        return self.__ftdp[data_id].probes

    # Unused, Unnecessary.
    def get_row(self, *, probe, domain):
        """
        Get a row in the database for a given probe in a required domain.

        Parameters
        ----------
        probe : str
            The name of a required probe.
            For example, 'ad', 'cb', 'lf' 'out', etc.
        domain : str
            The short name of a required domain.
            'f' is for frequency; 't' is for time.

        Returns
        -------
        row : tuple
            A requested row in the database. It contains two fields:
            probes : tuple of str
                A tuple of probes.
                For example, ('ad', 'ca'), ('lf', 'out'), etc.
            data : ndarray
                Related time or frequency domain data.
        """
        data_id = self.__pr2id.get(probe)
        record = self.__ftdp[data_id]
        if domain == 'f':
            return record.probes, record.f
        elif domain == 't':
            return record.probes, record.t
        else:
            raise ValueError("An unexpected domain: {0}.".format(domain))

    # Notes about Python generator functions:
    # 1. This function can be called several times.
    # 2. A "yield" result can be "return" by a wrapper function.
    def iterate_rows(self, *, domain, group_by):
        """
        A generator function to go through a selected group of electrical
        characteristics in a certain domain.

        Parameters
        ----------
        domain : str
            The short name of a required domain.
            'f' is for frequency; 't' is for time.
        group_by : str
            The name of a required criterion to group rows by.
            'data' is for unique data; 'probe' is for unique probes.

        Returns
        -------
        result : generator
            The generator to go through a requested group of characteristics.
        """
        if group_by == 'data':
            if domain == 'f':
                for data_id in range(len(self.__ftdp)):
                    row = self.__ftdp[data_id]
                    yield row.probes, row.f
            elif domain == 't':
                for data_id in range(len(self.__ftdp)):
                    row = self.__ftdp[data_id]
                    yield row.probes, row.t
            else:
                raise ValueError("An unexpected domain: {0}.".format(domain))
        elif group_by == 'probe':
            if domain == 'f':
                for probe, data_id in self.__pr2id.items():
                    yield (probe,), self.__ftdp[data_id].f
            elif domain == 't':
                for probe, data_id in self.__pr2id.items():
                    yield (probe,), self.__ftdp[data_id].t
            else:
                raise ValueError("An unexpected domain: {0}.".format(domain))
        else:
            raise ValueError("An unexpected grouping: {0}.".format(group_by))

    def size_of(self, *, group):
        """
        Get the number of rows in a selected group of characteristics.

        Parameters
        ----------
        group : str
            The name of a required group.
            'data' is for unique data; 'probe' is for unique probes.

        Returns
        -------
        result : int
            The number of rows in a required group of characteristics.
        """
        if group == 'probes':
            return len(self.__pr2id)
        elif group == 'data':
            return len(self.__ftdp)
        else:
            raise ValueError("An unexpected grouping: {0}.".format(group))

    # The end of "CharacDB" class.


class SimDBManager:
    """
    A simulation database manager.
    An instance of the class is used to create a "CharacDB" database, fill it
    with data, and retrieve it to pass in some other place.
    """

    def __init__(self, *, freq2time):
        """
        Constructor of an instance of the simulation database manager.

        Fileds
        ------
        __charac_db : CharacDB or None
            A database of electrical characteristics.
            It is "None" if a database was not created.
        __freq2time : callable f(fa)
            A function to convert data from frequency to time domain.

        Notes
        -----
        It uses ".utils.realpair2complex" function to convert frequency domain
        "numpy.ndarray" of "float" type values into an "ndarray" of "complex"
        values.
        Iternal parameters of a "__freq2time" function depends on the size of
        an "ndarray" of data to convert. It uses an inverse discrete Fourier
        transform matrix to convert frequency domain data into the time domain.
        It is convenient to use this wrapper instead of using a database
        instance directly because it automatically applies functions to convert
        raw frequency domain data (ndarray of float) into frequency domain data
        (ndarray of complex) and time domain data (ndarray of float).
        """
        self.__charac_db = None
        self.__freq2time = freq2time

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    def create_db(self):
        """
        Create a database of electrical characteristics.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__charac_db = CharacDB()
        return self

    def extract_db(self):
        """
        Extract a database of electrical characteristics.
        After calling this method, the database manager does not contain a
        database.

        Returns
        -------
        charac_db : CharacDB
            A database that contains electrical characteristics (voltages or
            currents) in time and frequency domains.
        """
        result = self.__charac_db
        self.__charac_db = None
        return result

    def add_row(self, *, probes, data):
        """
        Add a data row into a database.

        Parameters
        ----------
        probes : tuple of str
            The names of a data set.
            For example, ('ad', 'ca'), (lf' 'out'), etc.
        data : ndarray of float
            Raw frequency domain data. Each harmonic is represented by a pair
            of floating point numbers.

        Returns
        -------
        "self" reference to the caller object.
        """
        self.__charac_db.add(probes=probes,
                             fdata=hbm.realpair2complex(data),
                             tdata=self.__freq2time(data))
        return self

    # The end of "SimDBManager" class.


# Note: this class is not declaired as abstract since Python v3.7.3 which is
# used for the development does not provide "abc.update_abstractmethods"
# function which is necessary to call after overriding abstract methods outside
# from an heir class. To get more information, see
# https://docs.python.org/3/library/abc.html
class SimDataBase:
    """
    Simulation data abstract base class.
    It is the base class for classes which instances are used to store
    simulation data, i. e. electrical characteristics and the original
    parameters of the simulation.

    After calling "add_charac_getters", an heir of the base class has some
    automatically generated methods for each probe. For more details, see the
    "add_charac_getters" description.

    Fields (attributes) that must be in the heirs (the examples below are
    adduced for a class E power amplifier model):
    Probes : collections.namedtuple
        A "namedtuple" to store electrical characteristics by names of their
        probes. For example,
        Probes = namedtuple('Probes', 'pwr ad ca cb lb cf lf out')
    CLEnergs : collections.namedtuple
        A "namedtuple" to store the maximum energy values of capacitors and
        inductors. For example,
        CLEnergs = namedtuple('CLEnergs', 'ca cb cf lb lf')
    """

    def __init__(self, *, hb_opts, params, db_v, db_i):
        """
        Constructor of the base class.

        Parameters
        ----------
        hb_opts : HBOptions
            A "namedtuple" that contains the parameters of a harmonic balance
            simulation.
        params : Params
            A "namedtuple" that contains the parameters of a power amplifier
            model.
        cdb_v : CharacDB
            A database of voltages in time and frequency domains.
        cdb_i : CharacDB
            A database of currents in time and frequency domains.

        Fields
        ------
        __hb_opts : HBOptions
            Parameters of harmonic balance simulation.
        __params : Params
            Parameters of a power amplifier model.
        __cdb_v : CharacDB
            A database of voltages in time and frequency domains.
        __cdb_i : CharacDB
            A database of currents in time and frequency domains.
        __tsamps : ndarray
            An array of time samples.
        __fbins : ndarray
            An array of frequency bins.
        """
        self.__hb_opts = hb_opts
        self.__params = params
        self.__cdb_v = db_v
        self.__cdb_i = db_i
        self.__tsamps = hbm.make_time_samples(s_len=self.get_s_len(), t_wrk=self.get_t_wrk())
        self.__fbins = hbm.make_freq_bins(h_len=self.get_h_len(), t_wrk=self.get_t_wrk())

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")

    # Get HB options.

    def get_hb_options(self):       return self.__hb_opts

    def get_h_len(self):            return self.__hb_opts.h_len
    def get_s_len(self):            return self.__hb_opts.s_len
    def get_f_wrk(self):            return self.__hb_opts.f_wrk
    def get_t_wrk(self):            return self.__hb_opts.t_wrk
    def get_sample_duration(self):  return self.__hb_opts.dt
    def get_hb_maxiter(self):       return self.__hb_opts.maxiter
    def get_hb_reltol(self):        return self.__hb_opts.reltol

    # Get PA parameters.

    def get_params(self):           return self.__params

    def get_name(self):             return self.__params.name
    def get_ad_name(self):          return self.__params.ad.name
    def get_ad_config(self):        return self.__params.ad
    def get_load_name(self):        return self.__params.load.name
    def get_load_config(self):      return self.__params.load

    # Get characteristics.

    def get_time_samples(self):     return self.__tsamps
    def get_freq_bins(self):        return self.__fbins

    def get_charac(self, *, charac, probe, domain):
        """
        Get a selected electrical characteristic.

        Parameters
        ----------
        charac : str
            The short name of a required characteristic.
            'v' is for voltages; 'i' is for currents.
        probe : str
            The name of a required probe.
            For example, 'ad', 'cb', 'lf' 'out', etc.
        domain : str
            The short name of a required domain.
            'f' is for frequency; 't' is for time.

        Returns
        -------
        result : ndarray
            An array of a required electrical characteristic.
        """
        if charac == 'i':
            return self.__cdb_i.get_data(probe=probe, domain=domain)
        elif charac == 'v':
            return self.__cdb_v.get_data(probe=probe, domain=domain)
        else:
            raise ValueError("An unexpected characteristic: {0}.".format(charac))

    def size_of(self, *, charac, group):
        """
        Get the number of rows in a selected group of characteristics.

        Parameters
        ----------
        charac : str
            The short name of a required characteristic.
            'v' is for voltages; 'i' is for currents.
        group : str
            The name of a required group.
            'data' is for unique data; 'probe' is for unique probes.

        Returns
        -------
        result : int
            The number of rows in a required group of characteristics.
        """
        if charac == 'i':
            return self.__cdb_i.size_of(group=group)
        elif charac == 'v':
            return self.__cdb_v.size_of(group=group)
        else:
            raise ValueError("An unexpected characteristic: {0}.".format(charac))

    # Generators.

    def iterate_characs(self, *, charac, domain, group_by):
        """
        A generator function to go through a selected group of electrical
        characteristics.

        Parameters
        ----------
        charac : str
            The short name of a required characteristic.
            'v' is for voltages; 'i' is for currents.
        domain : str
            The short name of a required domain.
            'f' is for frequency; 't' is for time.
        group_by : str
            The name of a required criterion to group rows by.
            'data' is for unique data; 'probe' is for unique probes.

        Returns
        -------
        result : generator
            The generator to go through a requested group of characteristics.

        Notes
        -----
        This method returns a generator object to go through a certain
        sub-collection of data row by row. It provides the access to electrical
        characteristics and can be used to build plots, for example.
        """
        if charac == 'i':
            return self.__cdb_i.iterate_rows(domain=domain, group_by=group_by)
        elif charac == 'v':
            return self.__cdb_v.iterate_rows(domain=domain, group_by=group_by)
        else:
            raise ValueError("An unexpected characteristic: {0}.".format(charac))

    # Saving.

    def save(self, dirname="simdata"):
        """
        Save simulation data into a memory storage device.
        It produces 3 files:
        "params.ini" which contains parameters of a power amplifier model and
            options of harmonic balance simulation;
        "fdata.csv" which contains frequency characteristics;
        "tdata.csv" which contains time characteristics.
        If the data files already exist in a required directory, then they will
        be rewritten after calling this method.

        Parameters
        ----------
        dirname : str or pathlib.Path
            A directory in which the data will be saved. It can be:
            1. An "str" object that contains an absolute path. For example,
               "/home/username/projects/power_amplifier/simdata".
            2. An "str" object that contains a relative path in the current
               working directory (CWD). For example,
               "simdata".
            3. A "Path" object that contains an absolute path.
            4. A "Path" object that contains a relative path in the CWD.
        """
        # --- Parameters ---
        # Creating data to write.
        # For example, "_class_e_le".
        # Params = namedtuple(
        #     'Params',
        #     'name h_len s_len t_wrk f_wrk dt ad load \
        #      v_pwr c_ca g_ca c_cb g_cb l_lb r_lb c_cf g_cf l_lf r_lf z_out_h1')

        cfg_parser = configparser.ConfigParser()

        # Section: [hb].
        cfg_parser['hb'] = self.get_hb_options()._asdict()

        # Section: [model].
        model_dict = dict()
        params = self.get_params()
        for idx in range(len(params)):
            name = params._fields[idx]
            if not (name == 'ad' or name == 'load'):
                model_dict[name] = params[idx]
        cfg_parser['model'] = model_dict

        # Section: [model.ad].
        cfg_parser['model.ad'] = params.ad._asdict()

        # Section: [model.load].
        load_dict = dict()
        for idx in range(len(params.load)):
            name = params.load._fields[idx]
            if not name == 'tunres':
                load_dict[name] = params.load[idx]
        cfg_parser['model.load'] = load_dict

        # Section: [model.load.tunres].
        if hasattr(params.load, 'tunres'):
            cfg_parser['model.load.tunres'] = params.load.tunres._asdict()

        # Check if a required directory exists and create it if it does not.
        # To learn how to do that, see
        # "Working With Files in Python", realpython.com, by Vuyisile Ndlovu,
        # on 2023.08.11.
        # It can be the name of a directory or a path with subdirectories.
        # To make subdirectories, use "/" symbol.
        # "Python's pathlib Module: Taming the File System", realpython.com,
        # by Geir Arne Hjelle, 2023.04.17.
        # A current working directory (CWD) here is the directory of a project
        # that uses this module.
        # Path.cwd() # Get the current working directory.
        # Path(r"D:\PowAmp\simdata\params.ini") # r - raw string literals.
        # path = Path.cwd().joinpath(dirname) # dirname : str; relative path.
        path = self.__make_path(dirname)
        # Do not raise an exception if the directory already exists.
        path.mkdir(exist_ok=True)

        path_params = path.joinpath("params.ini")
        with path_params.open(mode='w') as params_obj:
            cfg_parser.write(params_obj)
            # Note: a "write" method call does not return anything.
            # If something goes wrong, it will throw an exception.

        # --- Frequency data ---
        # About column-data into row-data conversion, see
        # "convert all rows to columns and columns to rows in Arrays
        # [duplicate]", 2016.09.06,
        # https://stackoverflow.com/questions/39348600/convert-all-rows-to-columns-and-columns-to-rows-in-arrays
        # "Transpose 2D list in Python (swap rows and columns)", 2023.05.07,
        # https://note.nkmk.me/en/python-list-transpose/
        # About how to print complex numbers without parentheses, see
        # "Complex number plotting in python", 2018.03.14,
        # https://stackoverflow.com/questions/49275339/complex-number-plotting-in-python
        # "[Tutor] How to print complex numbers without enclosing parentheses",
        # 2008.09.19,
        # https://mail.python.org/pipermail/tutor/2008-September/064368.html
        path_tdata = path.joinpath("fdata.csv")
        with path_tdata.open(mode='w', newline='') as tdata_obj:
            tdata_writer = csv.writer(tdata_obj)
            names = list()
            names.append("freq")
            cols = list()
            cols.append(self.get_freq_bins())
            # Voltages.
            for probes, tdata in self.iterate_characs(charac='v', domain='f', group_by='data'):
                names.append("v_"+" v_".join(probes))
                cols.append(tdata)
            # Currents.
            for probes, tdata in self.iterate_characs(charac='i', domain='f', group_by='data'):
                names.append("i_"+" i_".join(probes))
                cols.append(tdata)
            # Writing into the file.
            tdata_writer.writerow(names)
            for row in zip(*cols):
                # The problem here is that it uses parentheses for complex
                # numbers.
                row_str = list()
                # The 2nd index is for extraction from an inlaid list.
                row_str.append(str(row[0]))
                for idx in range(1, len(row)):
                    cx = row[idx]
                    # "{cx.imag:+}" means to always use a sign.
                    row_str.append(f"{cx.real}{cx.imag:+}j")
                tdata_writer.writerow(row_str)

        # --- Time data ---
        path_tdata = path.joinpath("tdata.csv")
        with path_tdata.open(mode='w', newline='') as tdata_obj:
            tdata_writer = csv.writer(tdata_obj)
            names = list()
            names.append("time")
            cols = list()
            cols.append(self.get_time_samples())
            # Voltages.
            for probes, tdata in self.iterate_characs(charac='v', domain='t', group_by='data'):
                names.append("v_"+" v_".join(probes))
                cols.append(tdata)
            # Currents.
            for probes, tdata in self.iterate_characs(charac='i', domain='t', group_by='data'):
                names.append("i_"+" i_".join(probes))
                cols.append(tdata)
            # Writing into the file.
            tdata_writer.writerow(names)
            for row in zip(*cols):
                tdata_writer.writerow(row)

        # The end of the "save" method.

    # Derived characteristics.

    VIAmps = namedtuple('VIAmps', 'v i')
    # v : Probes
    # i : Probes

    def get_vi_amps(self):
        """
        Get the amplitudes of voltages and currents of the power amplifier
        components.

        Returns
        -------
        result : VIAmps
            A "collections.namedtuple" object called "VIAmps". It has two
            fields: "v" and "i" related to voltages and currents respectively.
            Each of them contains a "namedtuple" "Probes" object which contains
            the amplitudes.

        Notes
        -----
        An amplitude value has a sign, while a magnitude value does not.
        Here is the maximum deviation from zero.
        """
        # Voltages.
        v_amps = list()
        for (probe,), tdata in self.iterate_characs(charac='v', domain='t', group_by='probe'):
            minimum = tdata.min()
            maximum = tdata.max()
            if np.abs(maximum) < np.abs(minimum):   v_amps.append(minimum)
            else:                                   v_amps.append(maximum)
        # Currents.
        i_amps = list()
        for (probe,), tdata in self.iterate_characs(charac='i', domain='t', group_by='probe'):
            minimum = tdata.min()
            maximum = tdata.max()
            if np.abs(maximum) < np.abs(minimum):   i_amps.append(minimum)
            else:                                   i_amps.append(maximum)
        # Result.
        return self.VIAmps(v=self.Probes._make(v_amps),
                           i=self.Probes._make(i_amps))

    VIMags = namedtuple('VIMags', 'v i')
    # v : Probes
    # i : Probes

    def get_vi_mags(self):
        """
        Get the magnitudes of voltages and currents of the power amplifier
        components.

        Returns
        -------
        result : VIMags
            A "collections.namedtuple" object called "VIMags". It has two
            fields: "v" and "i" related to voltages and currents respectively.
            Each of them contains a "namedtuple" "Probes" object which contains
            the magnitudes.

        Notes
        -----
        It can be used to get to know the peak voltages and currents of the
        electrical components and therefore to define their required ratings.
        However, notice that during a transient process, some of the peak
        voltages and currents can be significantly greater than in the
        respective steady-state response.
        """
        # Voltages.
        v_mags = list()
        for (probe,), tdata in self.iterate_characs(charac='v', domain='t', group_by='probe'):
            v_mags.append(np.abs(tdata).max())
        # Currents.
        i_mags = list()
        for (probe,), tdata in self.iterate_characs(charac='i', domain='t', group_by='probe'):
            i_mags.append(np.abs(tdata).max())
        # Result.
        return self.VIMags(v=self.Probes._make(v_mags),
                           i=self.Probes._make(i_mags))

    def get_cl_energs(self):
        """
        Get the maximum energies of reactive components.

        The maximum energy that is stored in a capacitor
        e_c_max = 0.5*c*v_c_max^2, where
        c - capacitance,
        v_c_max - voltage magnitude on the capacitor.

        The maximum energy that is stored in an inductor
        e_l_max = 0.5*l*i_l_max^2, where
        l - inductance,
        i_l_max - current magnitude through the inductor.

        Returns
        -------
        result : CLEnergs
            A "collections.namedtuple" object called "CLEnergs". Its fields
            have the same names as respective probes. Each field contains an
            energy value.

        Notes
        -----
        If you know values of energy densities (J/m^3) and specific energies
        (J/kg) for each component, you can assess the minimum volume and mass
        of these components.
        """
        # Capasitors go first, inductors go second.
        energs = list()
        for probe in self.CLEnergs._fields:
            if probe[0] == 'c':  # Capacitors.
                c = getattr(self.__params, 'c_'+probe)
                v_max = np.abs(self.get_charac(charac='v', probe=probe, domain='t')).max()
                energs.append(0.5*c*v_max**2)
            elif probe[0] == 'l':  # Inductors.
                l = getattr(self.__params, 'l_'+probe)
                i_max = np.abs(self.get_charac(charac='i', probe=probe, domain='t')).max()
                energs.append(0.5*l*i_max**2)
            else:
                raise RuntimeError(
                    "Unexpected probe name: {0}.\n"
                    "Cannot define whether it is a capacitor or inductor".format(probe))
        # Result.
        return self.CLEnergs._make(energs)

    # Note: it requires to use "v_pwr" constant voltage source as an energy
    # source.
    def get_p_pwr_avg(self):
        """
        Get an average value of power consumed from the voltage source "v_pwr".
        p_pwr_avg = -v_pwr*i_pwr_avg, where
        v_pwr - power supply (constant) voltage,
        i_pwr_avg - average power supply current.

        Returns
        -------
        p_pwr_avg : float
            An average value of consumed power.

        Notes
        -----
        Due to the fact that the current flows into the voltage source, it
        requires to multiply (v_pwr*i_pwr_avg) by "-1" to get a positive
        number.
        "i_pwr_avg" is calculated as the 0th harmonic.
        """
        i_pwr_avg = np.real(self.get_i_pwr_f()[0])
        return -self.__params.v_pwr*i_pwr_avg

    def get_s_out_h1(self):
        """
        Get complex output power of the 1st harmonic.
        s_out_h1 = p_out_h1 + 1i*q_out_h1 = 0.5*z_out_h1*i_out_h1m, where
        p_out_h1 - real (active) output power of the 1st harmonic,
        q_out_h1 - imaginary (reactive) output power of the 1st harmonic,
        z_out_h1 - load impedance at the 1st harmonic frequency,
        i_out_h1m - magnitude of the 1st harmonic of output current.

        Returns
        -------
        s_out_h1 : complex
            A complex value of output power at the working frequency.
        """
        i_out_h1m = np.abs(self.get_i_out_f()[1])
        return 0.5*self.__params.z_out_h1*i_out_h1m**2

    def get_p_ad_avg(self):
        """
        Get an average value of power dissipated in the active device (AD).
        p_ad_avg = sum(v_ad_t*i_ad_t)/s_len, where
        v_ad_t - vector of AD voltage time samples with "s_len" size,
        i_ad_t - vector of AD current time samples with "s_len" size,
        s_len - number of time samples,
        sum() - sum operation,
        * - element-wise multiplication.

        Returns
        -------
        p_ad_avg : float
            An average value of power dissipation in the AD.

        Notes
        -----
        Remember that if a power amplifier (PA) efficiency is about 90 %, then
        improving its efficiency by 1 % reduces power dissipation in the AD by
        10 % (if there are no other losses). Sometimes having 97 % efficiency
        provides the ability to significantly reduce weight, size, and cost of
        the PA in compare with one that has only 90 % efficiency. Of course
        the cost will be reduced if the AD that brings the ability to achieve
        such as efficiency is not too expensive.
        It does not take into account losses in the "ca" anode-cathode
        capacitor.
        """
        return pau.p_avg(v_t=self.get_v_ad_t(), i_t=self.get_i_ad_t())

    def get_p_out_avg(self):
        """
        Get an average value of real (active) output power.
        p_out_avg = sum(v_out_t*i_out_t)/s_len, where
        v_out_t - vector of output voltage time samples with "s_len" size,
        i_out_t - vector of output current time samples with "s_len" size,
        s_len - number of time samples,
        sum() - sum operation,
        * - element-wise multiplication.

        Returns
        -------
        p_out_avg : float
            An average value of active output power.
        """
        return pau.p_avg(v_t=self.get_v_out_t(), i_t=self.get_i_out_t())

    def get_eff_h1(self):
        """
        Get energy conversion efficiency value at the 1st harmonic frequency.
        eff_h1 = p_out_h1/p_pwr_avg, where
        p_out_h1 - average output power of the 1st harmonic,
        p_pwr_avg - average power consumed from the "v_pwr" voltage source.

        Returns
        -------
        eff_h1 : float
            An efficiency value.

        Notes
        -----
        Usually it calls just "efficiency", but the author prefers more certain
        definition.
        """
        return np.real(self.get_s_out_h1())/self.get_p_pwr_avg()

    def get_mpoc(self):
        """
        Get a modified power output capability (MPOC or "cpmr") value.
        mpoc = p_out_h1/(v_ad_max*i_ad_rms), where
        p_out_h1 - output power of the 1st harmonic,
        v_ad_max - the maximum voltage on the active device (AD),
        i_ad_rms - root mean square current through the AD.

        Returns
        -------
        mpoc : float
            An MPOC value.

        Notes
        -----
        Use MPOC to assess comparative efficiency of an AD usage in different
        power amplifiers. Since an AD may cost a considerable amount of money,
        it is important to use it as effective as it is possible.

        References
        ----------
        "High-Efficiency Class E, EF2, and E/F3 Inverters",
        Zbigniew Kaczmarczyk, 2006.
        """
        p_out_h1 = np.real(self.get_s_out_h1())
        v_ad_max = self.get_v_ad_t().max()
        i_ad = self.get_i_ad_t()
        i_ad_rms = pau.rms(arr=i_ad)
        return pau.mpoc(v_ad_max=v_ad_max, i_ad_rms=i_ad_rms,
                        p_out_h1=p_out_h1)

    # Private section.

    @staticmethod
    def __make_path(dirname):
        """
        Make a "pathlib.Path" object.

        Parameters
        ----------
        dirname : str or Path
            A required directory. It can be:
            1. An "str" object that contains an absolute path. For example,
               "/home/username/projects/power_amplifier/simdata".
            2. An "str" object that contains a relative path in the current
               working directory (CWD). For example,
               "simdata".
            3. A "Path" object that contains an absolute path.
            4. A "Path" object that contains a relative path in the CWD.

        Returns
        -------
        path : Path
            The path related to the required directory.
        """
        # Convert "dirname" into "path" variable.
        if isinstance(dirname, str):
            path = Path(dirname)
        elif isinstance(dirname, Path):
            path = dirname
        else:
            raise TypeError("Unexpected 'dirname' type: {0}.".format(type(dirname)))
        # Check if a path is not absolute.
        # Tie a relative path with the current working directory.
        if not path.is_absolute():
            path = Path.cwd().joinpath(path)
        # Return the path.
        return path

    # The end of "SimDataBase" class.


class SimDataLE:
    """
    "SimData" base class for power amplifiers with lumped elements.
    """
    def __init__(self):
        raise NotImplementedError

    def __copy__(self):
        """
        A "copy" operation is disabled.
        """
        raise TypeError("A 'copy' operation is not supported.")


def add_charac_getters(SimDataClass):
    """
    Calling this function after "SimData" class definition will add to the
    class a set of methods to get electrical characteristics.
    It must be called after a certain "SimData" class definition.
    A "SimData" class must contain the "collections.namedtuple" definition
    called "Probes".

    A method definition looks like
    def get_c_pr_d(self), where
    c - characteristic, can be "v" (voltage) or "i" (current);
    pr - probe name, for example, "ad", "cb", "out", etc.;
    d - domain, can be "f" (frequency) or "t" (time).
    Therefore each probe has 4 getters:
    get_i_pr_f, get_i_pr_t, get_v_pr_f, get_v_pr_t.
    Some certain examples of these getters:
    get_v_ad_t (i. e. get the voltage on an active device in time domain),
    get_i_out_f (i. e. get the output current in frequency domain).
    """
    # To learn how to add methods to a class, see
    # 1. "Creating functions (or lambdas) in a loop (or comprehension)",
    #    2010.08.07,
    #    https://stackoverflow.com/questions/3431676/creating-functions-or-lambdas-in-a-loop-or-comprehension
    # 2. "How to Use Python Lambda Functions", realpython.com,
    #    by Andre Burgaud, on 2023.04.26,
    #    https://realpython.com/python-lambda
    for probe in SimDataClass.Probes._fields:
        def make_getter_i_pr_f(probe=probe):
            def get_i_pr_f(self):
                return self.get_charac(charac='i', probe=probe, domain='f')
            return get_i_pr_f
        def make_getter_i_pr_t(probe=probe):
            def get_i_pr_t(self):
                return self.get_charac(charac='i', probe=probe, domain='t')
            return get_i_pr_t
        def make_getter_v_pr_f(probe=probe):
            def get_v_pr_f(self):
                return self.get_charac(charac='v', probe=probe, domain='f')
            return get_v_pr_f
        def make_getter_v_pr_t(probe=probe):
            def get_v_pr_t(self):
                return self.get_charac(charac='v', probe=probe, domain='t')
            return get_v_pr_t

        setattr(SimDataClass, 'get_i_'+probe+'_f', make_getter_i_pr_f())
        setattr(SimDataClass, 'get_i_'+probe+'_t', make_getter_i_pr_t())
        setattr(SimDataClass, 'get_v_'+probe+'_f', make_getter_v_pr_f())
        setattr(SimDataClass, 'get_v_'+probe+'_t', make_getter_v_pr_t())
