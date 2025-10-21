
import os,pickle, h5py, logging
import scipy.sparse as ssparse
import scipy.io as spio
from typing import Dict
import numpy as np

def load_data(loadPath):

    ext = os.path.splitext(loadPath)[1]
    if ext == ".hdf5":
        ld = load_dict_from_hdf5(loadPath)  # function from CaImAn
    elif ext == ".pkl":
        with open(loadPath, "rb") as f:
            ld = pickle.load(f)
    elif ext == ".mat":
        ld = loadmat(loadPath)
    else:
        assert False, "File extension not yet implemented for loading data!"
    return ld


def save_data(data, savePath):

    ext = os.path.splitext(savePath)[1]
    if ext == ".hdf5":
        save_dict_to_hdf5(data, savePath)  # function from CaImAn
    elif ext == ".pkl":
        with open(savePath, "wb") as f:
            pickle.dump(data, f)
    elif ext == ".mat":
        sv_data = {}
        for key in data:
            sv_data[str(key)] = data[key]
            if isinstance(data[key], dict):
                for keyy in data[key]:
                    if data[key][keyy] is None:
                        sv_data[str(key)][keyy] = np.array([])
        spio.savemat(savePath, sv_data)
    else:
        assert False, "File extension not yet implemented for saving data!"


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    ### get rid of some unnecessary entries
    for key in ["__header__", "__version__", "__globals__"]:
        del data[key]

    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def save_dict_to_hdf5(dic: Dict, filename: str, subdir: str = "/") -> None:
    """Save dictionary to hdf5 file
    Args:
        dic: dictionary
            input (possibly nested) dictionary
        filename: str
            file name to save the dictionary to (in hdf5 format for now)
    """

    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, subdir, dic)


def load_dict_from_hdf5(filename: str) -> Dict:
    """Load dictionary from hdf5 file

    Args:
        filename: str
            input file to load
    Returns:
        dictionary
    """

    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/")


def recursively_save_dict_contents_to_group(
    h5file: h5py.File, path: str, dic: Dict
) -> None:
    """
    Args:
        h5file: hdf5 object
            hdf5 file where to store the dictionary
        path: str
            path within the hdf5 file structure
        dic: dictionary
            dictionary to save
    """
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if key == "g":
            if item is None:
                item = 0
            logging.info(key + " is an object type")
            try:
                item = np.array(list(item))
            except:
                item = np.asarray(item, dtype=float)
        if key == "g_tot":
            item = np.asarray(item, dtype=float)
        if key in [
            "groups",
            "idx_tot",
            "ind_A",
            "Ab_epoch",
            "coordinates",
            "loaded_model",
            "optional_outputs",
            "merged_ROIs",
            "tf_in",
            "tf_out",
            "empty_merged",
        ]:
            logging.info(f"Key {key} is not saved")
            continue

        if isinstance(item, (list, tuple)):
            if len(item) > 0 and all(isinstance(elem, str) for elem in item):
                item = np.string_(item)
            else:
                item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, numpy.int32, and numpy.float64 types
        if isinstance(item, str):
            if path not in h5file:
                h5file.create_group(path)
            h5file[path].attrs[key] = item
        elif isinstance(item, (np.int64, np.int32, np.float64, float, np.float32, int)):
            # TODO In the future we may store all scalars, including these, as attributes too, although strings suffer the most from being stored as datasets
            h5file[path + key] = item
            logging.debug(f"Saving numeric {path + key}")
            if not h5file[path + key][()] == item:
                raise ValueError(
                    f"Error (v {h5py.__version__}) while saving numeric {path + key}: assigned value {h5file[path + key][()]} does not match intended value {item}"
                )
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            logging.debug(f"Saving {key}")
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype("|S32")
                h5file[path + key] = item
            # if not np.array_equal(h5file[path + key][()], item):
            #     raise ValueError(
            #         f"Error while saving ndarray {key} of dtype {item.dtype}"
            #     )
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        elif "sparse" in str(type(item)):
            logging.info(key + " is sparse ****")
            h5file[path + key + "/data"] = item.tocsc().data
            h5file[path + key + "/indptr"] = item.tocsc().indptr
            h5file[path + key + "/indices"] = item.tocsc().indices
            h5file[path + key + "/shape"] = item.tocsc().shape
        # other types cannot be saved and will result in an error
        elif item is None or key == "dview":
            h5file[path + key] = "NoneType"
        elif key in [
            "dims",
            "medw",
            "sigma_smooth_snmf",
            "dxy",
            "max_shifts",
            "strides",
            "overlaps",
            "gSig",
        ]:
            logging.info(key + " is a tuple ****")
            h5file[path + key] = np.array(item)
        elif type(item).__name__ in ["CNMFParams", "Estimates"]:  #  parameter object
            recursively_save_dict_contents_to_group(
                h5file, path + key + "/", item.__dict__
            )
        else:
            raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")


def recursively_load_dict_contents_from_group(h5file: h5py.File, path: str) -> Dict:
    """load dictionary from hdf5 object
    Args:
        h5file: hdf5 object
            object where dictionary is stored
        path: str
            path within the hdf5 file

    Starting with Caiman 1.9.9 we started saving strings as attributes rather than independent datasets,
    which gets us a better syntax and less damage to the strings, at the cost of scanning properly for them
    being a little more involved. In future versions of Caiman we may store all scalars as attributes.

    There's some special casing here that should be solved in a more general way; anything serialised into
    hdf5 and then deserialised should probably go back through the class constructor, and revalidated
    so all the fields end up with appropriate data types.
    """

    ans: Dict = {}
    for akey, aitem in h5file[path].attrs.items():
        ans[akey] = aitem

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            val_set = np.nan
            if isinstance(item[()], str):
                if item[()] == "NoneType":
                    ans[key] = None
                else:
                    ans[key] = item[()]

            elif key in [
                "dims",
                "medw",
                "sigma_smooth_snmf",
                "dxy",
                "max_shifts",
                "strides",
                "overlaps",
            ]:
                if isinstance(item[()], np.ndarray):
                    ans[key] = tuple(item[()])
                else:
                    ans[key] = item[()]
            else:
                if isinstance(item[()], np.bool_):  # sigh
                    ans[key] = bool(item[()])
                else:
                    ans[key] = item[()]
                    if isinstance(ans[key], bytes) and ans[key] == b"NoneType":
                        ans[key] = None

        elif isinstance(item, h5py._hl.group.Group):
            if key in ("A", "W", "Ab", "downscale_matrix", "upscale_matrix"):
                data = item[path + key + "/data"]
                indices = item[path + key + "/indices"]
                indptr = item[path + key + "/indptr"]
                shape = item[path + key + "/shape"]
                ans[key] = ssparse.csc_matrix(
                    (data[:], indices[:], indptr[:]), shape[:]
                )
                if key in ("W", "upscale_matrix"):
                    ans[key] = ans[key].tocsr()
            else:
                ans[key] = recursively_load_dict_contents_from_group(
                    h5file, path + key + "/"
                )
    return ans
