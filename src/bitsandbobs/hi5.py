# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-21 11:11:40
# @Last Modified: 2022-07-29 12:20:58
# ------------------------------------------------------------------------------ #
# Helper functions to work conveniently with hdf5 files
#
# Example
# ```
# import hi5 as h5
# fpath = '~/demo/file.hdf5'
# h5.ls(fpath)
# h5.recursive_ls(fpath)
# h5.load(fpath, '/some/dataset/')
# h5.recursive_load(fpath, hot=False)
# ```
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import numbers
import h5py
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s | %(name)-12s | %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M')
log = logging.getLogger(__name__)
log.setLevel("DEBUG")

try:
    from benedict import benedict
    _benedict_is_installed = True
except ImportError:
    _benedict_is_installed = False
    log.debug("benedict is not installed")

try:
    from addict import Dict
    _addict_is_installed = True
except ImportError:
    _addict_is_installed = False
    log.debug("addict is not installed")



def load(filenames, dsetname, keepdim=False, raise_ex=False, silent=False):
    """
        load a h5 dset into an array. opens the h5 file and closes it
        after reading.

        # Parameters
        filenames: str path to h5file(s).
                   if wildcard given, result from globed files is returned
        dsetname:  str, which dset to read
        keepdim:   bool, per default arrays with 1 element will be mapped to scalars
                   set this to `True` to keep them as arrays
        raise_ex: whether to raise exceptions. default false,
                  in this case, np.nan is returned if loading fails
        silent:   if set to true, exceptions will not be reported

        # Returns
        res: ndarray or scalar, depending on loaded datatype
    """

    def local_load(filename):
        try:
            file = h5py.File(filename, "r")
            res = file[dsetname]
            # map 1 element arrays to scalars
            if res.shape == (1,) and not keepdim:
                res = res[0]
            elif res.shape == ():
                res = res[()]
            else:
                res = res[:]
            file.close()
            return res
        except Exception as e:
            if not silent:
                log.error(f"failed to load {dsetname} from {filename}: {e}")
            if raise_ex:
                raise e
            else:
                return np.nan

    files = glob.glob(os.path.expanduser(filenames))
    res = []
    for f in files:
        res.append(local_load(f))

    if len(files) == 1:
        return res[0]
    else:
        return res


def ls(filename, groupname="/"):
    """
        list the keys in a dsetname

        Parameters
        ----------
        filename: path to h5file
        groupname: which dset to list

        Returns
        -------
        list: containing the contained keys as strings
    """
    try:
        file = h5py.File(os.path.expanduser(filename), "r")
        try:
            res = list(file[groupname].keys())
        except Exception as e:
            res = []
        file.close()
    except Exception as e:
        res = []

    return res


_h5_files_currently_open = dict(files=[], filenames=[])


def load_hot(filename, dsetname, keepdim=False):
    """
        sometimes we do not want to hold the whole dataset in RAM, because it is too
        large. Remember to close the file after processing!

        hmmm, two lists where indices have to match seem a bit fragile
    """
    global _h5_files_currently_open
    filename = os.path.expanduser(filename)
    if filename not in _h5_files_currently_open["filenames"]:
        file = h5py.File(filename, "r")
        _h5_files_currently_open["files"].append(file)
        _h5_files_currently_open["filenames"].append(filename)
    else:
        idx = _h5_files_currently_open["filenames"].index(filename)
        file = _h5_files_currently_open["files"][idx]

    try:
        # if its a xsingle value, load it even though this is 'hot'
        if file[dsetname].shape == (1,) and not keepdim:
            return file[dsetname][0]
        elif file[dsetname].shape == ():
            return file[dsetname][()]
        else:
            return file[dsetname]
    except Exception as e:
        log.error(f"Failed to load hot {filename} {dsetname}")
        raise e

def close_hot(which="all"):
    """
        hot files require a bit of care:
        * If a BetterDict is opened from a hot hdf5 file, and `all` hot files are closed, the datasets are no longer accessible.
        * from the outside, currently it is hard to check whether an element of a BeterDict is loaded
    """
    global _h5_files_currently_open
    # everything we opened
    if which == "all":
        for file in _h5_files_currently_open["files"]:
            try:
                file.close()
            except:
                log.debug("File already closed")
        _h5_files_currently_open["files"] = []
        _h5_files_currently_open["filenames"] = []
    # by index
    elif isinstance(which, int):
        del _h5_files_currently_open["files"][which]
        del _h5_files_currently_open["filenames"][which]
        try:
            _h5_files_currently_open["files"][which].close()
        except:
            log.debug("File already closed")
    # by passed hdf5 file handle
    elif isinstance(which, h5py.File):
        _h5_files_currently_open["files"].remove(which)
        _h5_files_currently_open["filenames"].remove(which.filename)
        try:
            which.close()
        except:
            log.debug("File already closed")

def remember_file_is_hot(file):
    # helper to keep a collection of open files
    global _h5_files_currently_open
    _h5_files_currently_open["files"].append(file)
    _h5_files_currently_open["filenames"].append(file.filename)


def recursive_ls(filename, groupname="/"):
    """
    Lists all paths within a h5 file, starting at dsetname.
    the returned paths _do not_ include the path up to `dsetname`.
    """

    # below we assume dsetname as parent node to end with `/`
    try:
        if groupname[-1] != "/":
            groupname += "/"
    except IndexError:
        groupname = "/"

    candidates = ls(filename, groupname)
    res = candidates.copy()
    for c in candidates:
        temp = recursive_ls(filename, groupname + f"{c}/")
        if len(temp) > 0:
            temp = [f"{c}/{el}" for el in temp]
            res += temp
    return res


def recursive_load(filename, groupname="/", skip=None, hot=False, keepdim=False, dtype = None):
    """
        Load a hdf5 file as a nested dict.
        enhenced dicts via benedict or addict are supported via `dtype` argument

        # Paramters:
        filename : str
            path to h5 file
        groupname : str
            where to start loading. think level below which all dsets are loaded.
        skip : list
            names of dsets to exclude
        hot : bool
            if True, does not load dsets to ram, but only links to the hdf5 file. this keeps the file open, call `close_hot()` when done!
            Use this if a dataset in your file is ... big
        keepdim : set to true to preserve original data-set shape for 1d arrays
            instead of casting to numbers
        dtype : str, dtype or None
            which dictionary class to use. Default, None uses a normal dict,
            "benedict" or "addict" use those types,
            if a type is passed, it is assumed to be a subclass of a normal dict and will be called as the constructor

        # Returns:
        dict : nested dict-like of the requested `dtype`
            always contains a `h5` key that holds details about the hdf5 file.
            Note that this is always at the toplevel of the returned dict, even when
            providing a groupname.
    """

    if dtype is None:
        dtype = dict
    elif isinstance(dtype, str):
        if dtype.lower() == "benedict":
            assert _benedict_is_installed, "try `pip install python-benedict`"
            dtype = benedict
        elif dtype.lower() == "addict":
            assert _addict_is_installed, "try `pip install addict`"
            dtype = Dict
        else:
            raise ValueError("unsupported value passed for `dtype`")
    else:
        # assert isinstance(dtype, type), "unsupported value passed for `dtype`"
        assert isinstance(dtype(), dict), "unsupported value passed for `dtype`"

    if skip is not None:
        assert isinstance(skip, list)
    else:
        skip = []

    assert isinstance(hot, bool)
    assert isinstance(keepdim, bool)
    assert isinstance(groupname, str)
    assert isinstance(filename, str)

    # below we assume dsetname as parent node to end with `/`
    try:
        if groupname[-1] != "/":
            groupname += "/"
    except IndexError:
        groupname = "/"

    # get a flat list of all paths. _within_ dsetname.
    # so dsetname is not included in the paths.
    candidates = recursive_ls(filename, groupname)

    res = dtype()
    res["h5"] = dtype()
    res["h5"]["filename"] = filename
    res["h5"]["dsetname"] = groupname
    if hot:
        f = h5py.File(filename, "r")
        # res._set_h5_file(f)
        res["h5"]["file"] = f
        remember_file_is_hot(f)

    # track the number of levels for each paths
    path_lengths = []
    maxdepth = 0
    for path in candidates:
        l = len(path.split("/"))
        path_lengths.append(l)
        if l > maxdepth:
            maxdepth = l

    # iterate by depth, creating hierarchy
    for ddx in range(1, maxdepth + 1):
        # go over the list of flat paths, only act on paths with the right depth
        for ldx, l in enumerate(path_lengths):
            if l == ddx:
                path = candidates[ldx]
                components = path.split("/")
                # now we have components for output dict, but for file access we need
                # to prefix the path with dsetname
                path = groupname + "/".join(components)

                # TODO: also check wildcards
                if len([x for x in skip if x in components]) > 0:
                    continue

                # create the hierarchy, modifying temp also modifies res.
                temp = res
                if ddx > 1:
                    for out_key in components[0:-1]:
                        temp = temp[out_key]

                # last level, load data or create new branch
                out_key = components[-1]
                log.debug(path)
                if len(ls(filename, path)) > 0:
                    temp[out_key] = dtype()
                else:
                    if hot:
                        temp[out_key] = load_hot(filename, path, keepdim)
                    else:
                        temp[out_key] = load(filename, path, keepdim)

    return res


def recursive_write(filename, h5_data, h5_desc=None, **kwargs):
    """
        Write nested dictionaries to hdf5 files.
        Completely overwrites the file at `filename`, purging all previous content.

        Needs python-benedict.
        Dict keys cannot contain "." or "/" characters.

        Does not write subgroups of reserved keypath "/h5/"

        todo: workaround for list of strings, list of tuples etc
        todo: consistency checks between re-loaded an original data.
            due to differing dtypes, so far, this is not ensured.

        # Paramters
        filename : str
            path to (over) write the file to
        h5_data : nested dictionaries
            a dataset is created at each lowest level of the tree
        h5_desc : nested dict
            matching structure of `h5_data`, containing descriptions that will be set
            as the 'desc' attribute of the hdf5 dataset.
            Note: we cannot set descriptions for groups, as parents (groups)
            always correspond to a dict themselves
        **kwargs :
            are passed to `file.create_dataset()`, keys for compression set by default.
    """

    from benedict import benedict

    if h5_desc is None:
        h5_desc = dict()

    h5_data = benedict(h5_data, keypath_separator='/')
    h5_desc = benedict(h5_desc, keypath_separator='/')

    file = h5py.File(filename, "w")

    for key in h5_data.keypaths():
        try:
            if key[0:2] == "h5":
                continue
        except:
            pass

        key_kwargs = kwargs.copy()
        if isinstance(h5_data[key], dict):
            target = file.require_group(f"/{key}")
        else:
            data = h5_data[key]
            key_kwargs.setdefault("data", data)
            key_kwargs.setdefault("compression", "gzip")

            # scalars, and strings do not support compression
            try:
                d_len = len(data)
                if isinstance(data, str):
                    raise TypeError
                if isinstance(data, np.bytes_):
                    raise TypeError
            except TypeError:
                key_kwargs.pop("compression")

            # print(f"{key} {type(data)}")
            # print(key_kwargs)
            # print(h5_data[key])
            try:
                target = file.create_dataset(f"/{key}", **key_kwargs)
            except Exception as e:
                log.error(f"key `{key}`: {data}")
                log.error(f"{e}")

        # try:
        #     target.attrs["description"] = h5_desc[key]
        # except:
        #     # no description specified for this key
        #     pass

    file.close()



# def _check_for_string(data):
    # workaround to store a list of strings (via object array) to hdf5
    # dset = f_tar.create_dataset(
    #     "/meta/axis_overview",
    #     data=np.array(list(d_axes.keys()) + ["repetition"], dtype=object),
    #     dtype=h5py.special_dtype(vlen=str),
    # )


# for key in h5f.keypaths():
#     try:
#         print(f"{key} {np.all(h5f[key] == h5f2[key])}")
#     except Exception as e:
#         print(f"{key} {e}")
