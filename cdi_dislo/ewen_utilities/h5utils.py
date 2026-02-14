import numpy as np
import pylab as plt
import h5py as h5
from datetime import datetime

### hdf5 utility functions


def get_dataset_hdf5(h5path, filename):
    """
    return a dataset from an hdf5 safely
    """
    try:
        with h5.File(filename, "r") as h5f:
            counter = h5f[h5path][()]
    except Exception:
        print("... %s does not exist ..." % h5path)
        counter = None

    return counter


def get_scan_counters_dict_hdf5(scan_no, filename, counters=[], prefix=""):
    """
    return a list of scan counters tested against an hdf5 file
    """
    h5path_counters = "/%s%i.1/measurement/" % (prefix, scan_no)

    try:
        with h5.File(filename, "r") as h5f:
            counters_list = [key for key in h5f[h5path_counters].keys()]

            print("#################\n Available channels: ")
            for item in counters_list:
                print("\t", item)
            print("#################")

            counter_keys = {}
            if len(counters) == 0:
                counters = counters_list
            else:
                print("Requested: ", counters)

            for counter in counters:
                if counters_list.count(counter):
                    try:
                        counter_keys[counter] = counter
                    except Exception:
                        print("channel:%s not found" % counter)
            print("Found:", counter_keys.keys())
            print("#################\n")

    except Exception:
        print("could not return a list of counters")
    return counter_keys


def get_scans_title_str_hdf5(filename, target_string="", verbose=True):
    """
    return a list of scans whose title contains <target_string> from an hdf5 <filename>
    """
    if verbose:
        print("Available %s scans in hdf5 file:" % target_string)

    with h5.File(filename, "r") as h5f:
        scans = [key for key in h5f.keys()]
        scans = list(map(float, scans))
        scans = list(map(int, scans))
        scans.sort()
        scansList = []
        for scan in scans:
            if str(h5f["%i.1/title" % scan][()]).count(target_string):
                if verbose:
                    print("%i ... %s" % (scan, h5f["%i.1/title" % scan][()]))
                scansList.append(scan)
    return scansList


def get_scan_motor_hdf5(scan_no, filename, motor_name, prefix=""):
    """
    return a list of scan counters tested against an hdf5 file
    """

    h5path = "/%s%i.1/instrument/positioners/%s" % (
        prefix,
        scan_no,
        motor_name,
    )

    with h5.File(filename, "r") as h5f:
        motor_pos = h5f[h5path][()]

    return motor_pos


def get_scan_start_time_hdf5(scan_no, filename, prefix=""):
    """
    return scan start time from scan_no in an hdf5 filename
    """

    h5path = "/%s%i.1/start_time" % (prefix, scan_no)

    with h5.File(filename, "r") as h5f:
        start_time = h5f[h5path][()]
        epoch = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    return epoch


def get_command(scan_no, filename):
    """
    return the scan command/title
    """
    return str(h5.File(filename, "r")["{}.1".format(scan_no)]["title"][()])


def print_scans_list(filename):
    _ = get_scans_title_str_hdf5(filename, target_string="", verbose=True)
    return


#################################################
#####           Master Scan class          ######
#################################################

# Scan class from Ewen Bellec


class Scan:
    def __init__(self, filename, scan_nb, verbose=True):
        """
        filename = path to hdf5 file (dataset level)
        scan_nb = scan number (accepts -n nomenclature)
        verbose = show some output
        """
        self.h5file = filename
        self.keys = self.sortH5Keys()
        self.data_type = "ID01"

        if scan_nb < 0:
            self.scan_string = self.keys[scan_nb]
        else:
            self.scan_string = "{}.1".format(scan_nb)

        self.verbose = verbose
        self.command = str(h5.File(self.h5file, "r")[self.scan_string]["title"][()])

        if verbose:
            print(f"Scan no: {self.scan_string}")
            print(f"{self.command}")

        try:
            self.sample = h5.File(self.h5file, "r")[self.scan_string]["sample/name"][
                ()
            ].decode("UTF-8")
        except Exception:
            pass

        try:
            self.getDetectorShape()
        except Exception:
            pass

        try:
            self.getDetectorName()
        except Exception:
            if self.verbose:
                print("detector name not found. Give it.")

        try:
            self.getDetectorShape()
        except Exception:
            pass

    def show_scaninfo(
        self,
    ):
        print(self.scan_string, self.command)

    def show_info(
        self,
    ):
        outs = ""
        for n, key in enumerate(self.keys):
            with h5.File(self.h5file, "r") as h5f:
                outs += key + " " + h5f[key + "/title"][()] + "\n"
        print(outs)
        return outs

    def sortH5Keys(self):
        with h5.File(self.h5file, "r") as h5f:
            keys = list(h5f.keys())
        index_sort = np.argsort([int(key.split(".")[0]) for key in keys])
        keys_sort = [keys[index] for index in index_sort]
        return keys_sort

    def getMotorPosition(self, motor_name):
        with h5.File(self.h5file, "r") as h5f:
            motor_pos = h5f[self.scan_string][
                "instrument/positioners/{}".format(motor_name)
            ][()]
        return motor_pos

    def getAllMotorDictionary(self):
        with h5.File(self.h5file, "r") as h5f:
            motor_dict = {}
            for motor_name in h5f[self.scan_string]["instrument/positioners/"].keys():
                motor_dict[motor_name] = h5f[self.scan_string][
                    "instrument/positioners/{}".format(motor_name)
                ][()]
        setattr(self, "motor_dict", motor_dict)
        return motor_dict

    def getDetectorName(self):
        with h5.File(self.h5file, "r") as h5f:
            if "mpx1x4" in h5f[self.scan_string]["measurement/"].keys():
                detector = "mpx1x4"
            if "eiger2M" in h5f[self.scan_string]["measurement/"].keys():
                detector = "eiger2M"
            if "mpxgaas" in h5f[self.scan_string]["measurement/"].keys():
                detector = "mpxgaas"

            if "andor_zyla" in h5f[self.scan_string]["measurement/"].keys():
                detector = "andor_zyla"
        if self.verbose:
            print("detector :", detector)
        self.detector = detector
        return detector

    def getDetectorShape(self):
        with h5.File(self.h5file, "r") as h5f:
            shape_0 = h5f[self.scan_string][
                "instrument/{}/dim_j".format(self.detector)
            ][()]
            shape_1 = h5f[self.scan_string][
                "instrument/{}/dim_i".format(self.detector)
            ][()]
        detector_shape = (shape_0, shape_1)

        if self.verbose:
            print("detector shape :", detector_shape)

        self.detector_shape = detector_shape
        return detector_shape

    def getDetCalibInfo(self):
        det_calib = {}
        with h5.File(self.h5file, "r") as h5f:
            det_calib["distance"] = h5f[self.scan_string][
                "instrument/{}/distance".format(self.detector)
            ][()]
            det_calib["beam_center_x"] = h5f[self.scan_string][
                "instrument/{}/beam_center_x".format(self.detector)
            ][()]
            det_calib["beam_center_y"] = h5f[self.scan_string][
                "instrument/{}/beam_center_y".format(self.detector)
            ][()]
            det_calib["x_pixel_size"] = h5f[self.scan_string][
                "instrument/{}/x_pixel_size".format(self.detector)
            ][()]
            det_calib["y_pixel_size"] = h5f[self.scan_string][
                "instrument/{}/y_pixel_size".format(self.detector)
            ][()]
        self.det_calib = det_calib
        return det_calib

    def getEnergy(self):
        with h5.File(self.h5file, "r") as h5f:
            try:
                energy = (
                    12.39842
                    / (
                        h5f[
                            "{}/instrument/monochromator/WaveLength".format(
                                self.scan_string
                            )
                        ][()]
                        * 1e10
                    )
                ) * 1e3  # energy in eV
            except Exception:
                energy = (
                    h5f["{}/instrument/positioners/mononrj".format(self.scan_string)][
                        ()
                    ]
                    * 1e3
                )
        self.energy = energy
        return energy

    def getAllCountersList(self, print_list=False):
        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())
        print(counter_list)
        if print_list:
            print(counter_list)
        return counter_list

    def getCounter(self, counter_name):
        with h5.File(self.h5file, "r") as h5f:
            return h5f[self.scan_string]["measurement/{}".format(counter_name)][()]

    def printCountersList(self, return_list=False):
        import h5py as h5

        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())

        print("available counters:")
        for counter in counter_list:
            print(counter, end="    ")
        print()

        if return_list:
            return counter_list

    def printSumCountersList(self, return_list=False):
        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())

        counter_sum_list = []
        print("available sum ROIs :")
        for counter in counter_list:
            if (
                (self.detector) in counter
                and ("avg" not in counter)
                and ("max" not in counter)
                and ("min" not in counter)
                and ("std" not in counter)
                and (self.detector != counter)
            ):
                print(counter, end="    ")
                counter_sum_list.append(counter)
        if return_list:
            return counter_sum_list

    def getDetectorSum(self, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            nb_img = h5f[self.scan_string][
                "measurement/{}".format(self.detector)
            ].__len__()
            for n in range(nb_img):
                if n == 0:
                    detector_sum = h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ].__getitem__(n)
                else:
                    detector_sum += h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ].__getitem__(n)

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title("log of sum of the detector images", fontsize=15)

        self.detector_sum = detector_sum
        return detector_sum

    def getImageRaw(self, roi=None):
        with h5.File(self.h5file, "r") as h5f:
            if roi is None:
                data = h5f[self.scan_string]["measurement/{}".format(self.detector)][()]
            else:
                if len(roi) == 4:
                    data = h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ][:, roi[0] : roi[1], roi[2] : roi[3]]
                elif len(roi) == 6:
                    data = h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ][roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
                else:
                    print("problem is getImageRaw !!!!")
        #                 #                 data = h5[self.scan_string]['measurement/{}'.format(self.detector)][()][:,roi[0]:roi[1],roi[2]:roi[3]]
        #                 #                 # Might create a memory problem for huge scan. I'll do something else
        #                 nb_img = h5f[self.scan_string][
        #                     "measurement/{}".format(self.detector)
        #                 ].__len__()
        #                 for n in range(nb_img):
        #                     img = h5f[self.scan_string][
        #                         "measurement/{}".format(self.detector)
        #                     ].__getitem__(n)[roi[0] : roi[1], roi[2] : roi[3]]
        #                     if n == 0:
        #                         data = np.zeros((nb_img,) + img.shape)
        #                     data[n] += h5f[self.scan_string][
        #                         "measurement/{}".format(self.detector)
        #                     ].__getitem__((n, np.arange(roi[0], roi[1]), )[roi[0] : roi[1], roi[2] : roi[3]]
        return data

    def getImages(self, roi=None):
        data = self.getImageRaw(roi=roi)

        if self.verbose:
            print("data.shape", data.shape)

        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data

    def getStartEndTime(self, return_seconds=False):
        with h5.File(self.h5file) as h5f:
            start_time = h5f[self.scan_string]["start_time"][()]
            start_time = start_time.decode("utf-8")

            end_time = h5f[self.scan_string]["end_time"][()]
            end_time = end_time.decode("utf-8")

        year = int(start_time.split("-")[0])
        month = int(start_time.split("-")[1])
        day = int(start_time.split("-")[2].split("T")[0])
        hour = int(start_time.split("-")[2].split("T")[1].split(":")[0])
        minutes = int(start_time.split("-")[2].split("T")[1].split(":")[1])
        seconds = int(
            start_time.split("-")[2].split("T")[1].split(":")[2].split(".")[0]
        )
        microseconds = int(
            start_time.split("-")[2]
            .split("T")[1]
            .split(":")[2]
            .split(".")[1]
            .split("+")[0]
        )

        start_time = datetime(year, month, day, hour, minutes, seconds, microseconds)

        year = int(end_time.split("-")[0])
        month = int(end_time.split("-")[1])
        day = int(end_time.split("-")[2].split("T")[0])
        hour = int(end_time.split("-")[2].split("T")[1].split(":")[0])
        minutes = int(end_time.split("-")[2].split("T")[1].split(":")[1])
        seconds = int(end_time.split("-")[2].split("T")[1].split(":")[2].split(".")[0])
        microseconds = int(
            end_time.split("-")[2]
            .split("T")[1]
            .split(":")[2]
            .split(".")[1]
            .split("+")[0]
        )

        end_time = datetime(year, month, day, hour, minutes, seconds, microseconds)

        if return_seconds:
            return start_time.timestamp(), end_time.timestamp()
        else:
            return start_time, end_time

    def getTimeEachPoints(self, return_seconds=False):
        if "SXDM_Scan" in str(type(self)):
            print(
                "Error : getTimeEachPoints doesn't work yet with sxdm scans. There might be a problem with the time counter. To be fixed!"
            )
            return

        start_time, end_time = self.getStartEndTime(return_seconds=False)

        elapsed_time = self.getCounter("elapsed_time")
        elapsed_time = elapsed_time + start_time.timestamp()
        elapsed_time = np.atleast_1d(elapsed_time)

        if return_seconds:
            return np.array(elapsed_time)
        else:
            time = [
                datetime.fromtimestamp(t).strftime("%d-%b-%Y (%H:%M:%S.%f)")
                for t in elapsed_time
            ]
            return np.array(time)

    def print_scan_motors(self, motor_list):
        print("scan no :", self.scan_string)
        for motor in motor_list:
            print(motor, round(np.nanmean(self.getMotorPosition(motor)), 2))
        return


##################################################################################################################################
######################################           Dscan and Ascan          ########################################################
##################################################################################################################################


class StandardScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.getDscanMotorPosition()

    def getDscanMotorPosition(self):
        motor_name = self.command.split()[1]
        motor = self.getMotorPosition(motor_name)

        if self.verbose:
            print("motor : {}".format(motor_name))

        self.motor = motor
        self.motor_name = motor_name

        return motor, motor_name

    def getRoiData(self, roi_name, plot=False, fig_title=""):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        if plot:
            plt.figure()
            plt.plot(self.motor, roidata, ".-")
            plt.xlabel(self.motor_name, fontsize=15)
            plt.ylabel(roi_name, fontsize=15)
            plt.title(fig_title, fontsize=15)
        return roidata


##################################################################################################################################
#########################################           LookupScan          ##########################################################
##################################################################################################################################


class LookupScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.getLookupscanMotorPosition()

    def getLookupscanMotorPosition(self):
        motor_keys = self.command.split()[-1][1:-2].split(",")
        motor_dict = {}
        for motor in motor_keys:
            motor_dict[motor] = self.getMotorPosition(motor)

        if self.verbose:
            print("motors : ", motor_keys)

        self.motor_dict = motor_dict
        return motor_dict


##################################################################################################################################
#####################################           Dmesh scan          #######################################################
##################################################################################################################################


class DmeshScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.getMeshMotorPosition()

    def getMeshMotorPosition(self):
        motor1_name = self.command.split()[1]
        motor2_name = self.command.split()[5]

        motor1 = self.getMotorPosition(motor1_name)
        motor2 = self.getMotorPosition(motor2_name)

        shape = (
            int(self.command.split()[-2]) + 1,
            int(self.command.split()[4]) + 1,
        )
        motor1 = np.reshape(motor1, shape)
        motor2 = np.reshape(motor2, shape)

        if self.verbose:
            print("motor1 : {}".format(motor1_name))
            print("motor2 : {}".format(motor2_name))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor1_name = motor1_name
        self.motor2_name = motor2_name

        return motor1, motor2, motor1_name, motor2_name

    def getRoiData(self, roi_name, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        roidata = roidata.reshape(self.motor1.shape)

        if plot:
            Plot2DMapSXDM_Dmesh(self, roidata, roi_name)

        return roidata

    def getImages(self, roi=None):
        data = self.getImageRaw(roi=roi)
        data = np.reshape(data, (self.motor1.shape) + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {})".format(
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )
        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data


##################################################################################################################################
######################################             2D SXDM scan            #######################################################
##################################################################################################################################


class SXDM_Scan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.getSXDM_MotorsPosition()

    def getSXDM_MotorsPosition(self):
        motor1_name = self.command.split()[1].replace(",", "")  # [:-1]
        motor2_name = self.command.split()[5].replace(",", "")  # [:-1]

        with h5.File(self.h5file, "r") as h5f:
            try:
                motor1 = h5f[self.scan_string][
                    "instrument/{}_position/value".format(motor1_name)
                ][()]
                motor2 = h5f[self.scan_string][
                    "instrument/{}_position/value".format(motor2_name)
                ][()]
            except Exception:
                motor1 = h5f[self.scan_string][
                    "instrument/{}/value".format(motor1_name)
                ][()]
                motor2 = h5f[self.scan_string][
                    "instrument/{}/value".format(motor2_name)
                ][()]

        # Reshape the motor positions into 2D arrays
        dim1 = int(self.command.split()[8].replace(",", ""))  # [:-1])
        dim2 = int(self.command.split()[4].replace(",", ""))  # [:-1])
        motor1 = motor1.reshape(dim1, dim2)
        motor2 = motor2.reshape(dim1, dim2)

        if self.verbose:
            print("motor1 : {}".format(motor1_name))
            print("motor2 : {}".format(motor2_name))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor1_name = motor1_name
        self.motor2_name = motor2_name
        return motor1, motor2, motor1_name, motor2_name

    def getRoiData(self, roi_name, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        roidata = roidata.reshape(self.motor1.shape)

        if plot:
            Plot2DMapSXDM_Dmesh(self, roidata, roi_name)

        return roidata

    def getImages(self, roi=None):
        data = self.getImageRaw(roi=roi)

        data = np.reshape(data, self.motor1.shape + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {})".format(
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )

        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data


##################################################################################################################################
######################################             3D SXDM scan            #######################################################
##################################################################################################################################


class SXDM_3D_Scan:
    def __init__(self, filename, motor3_name=None, verbose=True):
        self.h5file = filename
        self.scan1 = SXDM_Scan(filename, 1)
        self.verbose = verbose
        self.detector = self.scan1.detector

        self.nb_scan = len(
            np.unique(get_scans_title_str_hdf5(filename, "", verbose=False))
        )

        if motor3_name is None:
            self.motor3_name = self.FindThirdMotor()
        else:
            self.motor3_name = motor3_name

        self.getSXDM_3D_MotorsPosition()

        if self.verbose:
            print("detector :", self.detector)
            print("motor3_name :", self.motor3_name)
            print(
                "(if motor3_name is wrong, please put the correct motor as a string in motor3_name argument)"
            )

    def FindThirdMotor(self):
        # I only check if the third motor is eta or phi. Hope that will be enough
        phi = np.zeros(self.nb_scan)
        eta = np.zeros(self.nb_scan)
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            phi[n] += sxdm_scan.getMotorPosition("phi")
            eta[n] += sxdm_scan.getMotorPosition("eta")
        if np.all(eta == eta[0]) and (not np.all(phi == phi[0])):
            motor3_name = "phi"
        if (not np.all(eta == eta[0])) and np.all(phi == phi[0]):
            motor3_name = "eta"
        if (not np.all(eta == eta[0])) and (not np.all(phi == phi[0])):
            print(
                "Error : couldn't find the third motor name."
                + "\nPlease add this motor name in the class argument motor3_name"
            )
        return motor3_name

    def getSXDM_3D_MotorsPosition(self):
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            sxdm_scan.getSXDM_MotorsPosition()

            if n == 0:
                motor1 = np.zeros((self.nb_scan,) + sxdm_scan.motor1.shape)
                motor2 = np.zeros((self.nb_scan,) + sxdm_scan.motor2.shape)
                motor3 = np.zeros(self.nb_scan)

            motor1[n] += sxdm_scan.motor1
            motor2[n] += sxdm_scan.motor2
            motor3[n] += sxdm_scan.getMotorPosition(self.motor3_name)

        motor1_name = sxdm_scan.motor1_name
        motor2_name = sxdm_scan.motor2_name

        if self.verbose:
            print("motor1 : {},   shape : {}".format(motor1_name, motor1.shape))
            print("motor2 : {},   shape : {}".format(motor2_name, motor2.shape))
            print("motor3 : {},   shape : {}".format(self.motor3_name, motor3.shape))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor3 = motor3

        self.motor1_name = motor1_name
        self.motor2_name = motor2_name

        return motor1, motor2, motor3, motor1_name, motor2_name

    def printAllCountersList(self):
        self.scan1.printAllCountersList()

    def printSumCountersList(self):
        self.scan1.printSumCountersList()

    def getRoiData(self, roi_name, plot=False, pcolormesh_plot=False):
        roidata = np.zeros(self.motor1.shape)
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            roidata[n] += sxdm_scan.getRoiData(roi_name)

        if plot:
            self.plotRoiSXDM_3D(roidata, pcolormesh_plot=pcolormesh_plot)

        return roidata

    def plotRoiSXDM_3D(self, roidata, pcolormesh_plot=False):
        nb_rows = int(np.ceil(self.nb_scan / 4))
        fig, ax = plt.subplots(nb_rows, 4, figsize=(14, nb_rows * 4))

        for n, axe in enumerate(ax.flatten()):
            if n < self.nb_scan:
                if pcolormesh_plot:
                    axe.pcolormesh(self.motor1[n], self.motor2[n], roidata[n])
                else:
                    axe.imshow(
                        roidata[n],
                        extent=[
                            np.min(self.motor1[n]),
                            np.max(self.motor1[n]),
                            np.min(self.motor2[n]),
                            np.max(self.motor2[n]),
                        ],
                        origin="lower",
                    )
                axe.set_xlabel(self.motor1_name, fontsize=15)
                axe.set_ylabel(self.motor2_name, fontsize=15)
                axe.set_title("{} : {}".format(self.motor3_name, self.motor3[n]))
            else:
                fig.delaxes(axe)
        fig.tight_layout()

    def getDetectorSum(self, plot=False):
        for n in range(self.nb_scan):
            print(self.nb_scan - n, end=" ")
            scan_nb = n + 1
            sxdm_scan = Scan(self.h5file, scan_nb, verbose=False)
            sxdm_scan.getDetectorSum()
            if n == 0:
                detector_sum = sxdm_scan.detector_sum
            else:
                detector_sum += sxdm_scan.detector_sum
        print("")

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title("log of sum of the detector images", fontsize=15)

        self.detector_sum = detector_sum
        return detector_sum

    def getImages(self, roi=None, plot=False):
        for n in range(self.nb_scan):
            print(self.nb_scan - n, end=" ")
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            data_one_scan = sxdm_scan.getImages(roi=roi)
            if n == 0:
                data = np.zeros((self.nb_scan,) + data_one_scan.shape)
            data[n] += data_one_scan
        print("")

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {}, {} )".format(
                    self.motor3_name,
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )

        #         self.data = data # I wouldn't do that if it makes a copy of data.
        return data


##################################################################################################################################
#######################################             ct scan            #########################################################
##################################################################################################################################


class Scan_ct(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

    def getDetectorSum(self, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            detector_sum = h5f[self.scan_string][
                "measurement/{}".format(self.detector)
            ][()]

        self.detector_sum = detector_sum

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title(
                "log of sum of the detector images\nthis is actually just one image, not a sum",
                fontsize=15,
            )

        return detector_sum


##################################################################################################################################
#######################################             Utilities            #########################################################
##################################################################################################################################


def openScan(filename, scan_nb, verbose=False):
    command = get_command(scan_nb, filename)

    if "scan" in command and "lookupscan" not in command and "loopscan" not in command:
        return StandardScan(filename, scan_nb, verbose=verbose)

    if "lookupscan" in command:
        return LookupScan(filename, scan_nb, verbose=verbose)

    if "mesh" in command:
        return DmeshScan(filename, scan_nb, verbose=verbose)

    if "sxdm" in command or "kmap" in command:
        return SXDM_Scan(filename, scan_nb, verbose=verbose)

    if "ct" in command:
        return Scan_ct(filename, scan_nb, verbose=verbose)

    if "loopscan" in command:
        return Scan(filename, scan_nb, verbose=verbose)


def Plot2DMapSXDM_Dmesh(scan, roidata, roi_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].pcolormesh(scan.motor1, scan.motor2, roidata)
    ax[0].set_title("pcolormesh", fontsize=20)
    ax[1].imshow(
        roidata,
        origin="lower",
        extent=[
            np.min(scan.motor1),
            np.max(scan.motor1),
            np.min(scan.motor2),
            np.max(scan.motor2),
        ],
        aspect="auto",
    )
    ax[1].set_title("imshow", fontsize=20)
    for axe in ax:
        axe.set_xlabel(scan.motor1_name, fontsize=15)
        axe.set_ylabel(scan.motor2_name, fontsize=15)

    fig.suptitle(roi_name, fontsize=20)
    fig.tight_layout()
