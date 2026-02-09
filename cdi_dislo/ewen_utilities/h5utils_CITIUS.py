import numpy as np
import h5py as h5


def clean_keylists(key_list, string):
    key_list_clean = []
    for key in key_list:
        if string in key:
            key_list_clean.append(key)
    return key_list_clean


def open_citius_data(filename, roi=None, verbose=False):
    with h5.File(filename) as h5f:
        key_list = h5f["entry/"].keys()
        key_list = clean_keylists(key_list, "data_")

        if roi is None:
            data = np.zeros((len(key_list), 728, 384))
        else:
            data = np.zeros((len(key_list), roi[1] - roi[0], roi[3] - roi[2]))

        for n, key in enumerate(key_list):
            if verbose:
                if n % 10 == 0:
                    print(len(key_list) - n, end=" ")

            try:
                key_frames = h5f[f"entry/{key}"].keys()
                key_frames = clean_keylists(
                    key_frames, "frame_"
                )  # Careful, they're not sorted !!
                key_frames = clean_keylists(key_frames, "0")

                for frame in key_frames:
                    if roi is None:
                        data[n] += h5f[f"entry/{key}/{frame}/sensor_image_1"][
                            ()
                        ]
                    else:
                        data[n] += h5f[f"entry/{key}/{frame}/sensor_image_1"][
                            roi[0] : roi[1], roi[2] : roi[3]
                        ]
            except Exception :
                print(f"failed at entry/{key}")
                return data

    return data


#################################################
#####           Master Scan class          ######
#################################################

# Scan class from Ewen Bellec


class Scan:
    def __init__(self, folder_name, verbose=True):
        """
        folder_name = folder path to the scan data
        verbose = show some output
        """
        self.h5file = f"{folder_name}{folder_name.split('/')[-2]}_master.h5"
        self.detector = "CITIUS"
        self.pixel_size = 72.6e-6

        self.verbose = verbose

    def clean_keylists(key_list, string):
        key_list_clean = []
        for key in key_list:
            if string in key:
                key_list_clean.append(key)
        return key_list_clean

    def getImages(self, roi=None, merge=None):
        with h5.File(self.h5file) as h5f:
            key_list = h5f["entry/"].keys()
            key_list = clean_keylists(key_list, "data_")

            if roi is None:
                data = np.zeros((len(key_list), 728, 384))
            else:
                data = np.zeros(
                    (len(key_list), roi[1] - roi[0], roi[3] - roi[2])
                )

            for n, key in enumerate(key_list):
                if self.verbose:
                    if n % 10 == 0:
                        print(len(key_list) - n, end=" ")

                try:
                    key_frames = h5f[f"entry/{key}"].keys()
                    key_frames = clean_keylists(
                        key_frames, "frame_"
                    )  # Careful, they're not sorted !!
                    key_frames = clean_keylists(key_frames, "0")

                    for frame in key_frames:
                        if roi is None:
                            data[n] += h5f[
                                f"entry/{key}/{frame}/sensor_image_1"
                            ][()]
                        else:
                            data[n] += h5f[
                                f"entry/{key}/{frame}/sensor_image_1"
                            ][roi[0] : roi[1], roi[2] : roi[3]]
                except Exception :
                    print(f"failed at entry/{key}")
                    return data

        if merge is not None:
            data = np.sum(
                data.reshape(
                    (
                        data.shape[0] // merge,
                        merge,
                    )
                    + data.shape[1:]
                ),
                axis=1,
            )

        return data
