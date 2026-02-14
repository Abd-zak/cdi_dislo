import numpy as np
from silx.io.specfile import SpecFile as SF
import fabio
from silx.io.spech5 import SpecH5
import h5py


# def openSpec(specfile, printing=True):
def get_scans_title_str_hdf5(specfile, printing=True):
    spec = SF(specfile)

    if printing:
        for scan_no in spec.keys():
            for line in spec[scan_no].header:
                if "#S" in line:
                    print(line)
    return spec


def get_command_spec(scan_no, specfile):
    """
    return the scan command/title
    """
    spec = SF(specfile)
    for line in spec[scan_no - 1].header:
        if "#S" in line:
            break
    command = "".join([l_ + " " for l_ in line.split()[2:]])
    return command[:-1]


class Scan_spec:
    def __init__(
        self,
        specfile,
        scan_nb,
        path_imgs=None,
        search_img_first_file=True,
        search_detector=True,
        verbose=True,
    ):
        """
        filename = path to hdf5 file (dataset level)
        scan_nb = scan number (accepts -n nomenclature)
        verbose = show some output
        """
        self.specfile = specfile
        self.h5file = self.specfile[:-5] + ".h5"  # Dummy h5file for saving function
        self.spec = SF(self.specfile)
        self.scan_string = "{}.1".format(scan_nb)
        self.path_imgs = path_imgs
        self.verbose = verbose

        self.getCommand()
        self.getSampleName()

        if search_img_first_file:
            self.getFirstImageFile()
            if search_detector:
                self.detector = self.img_first_file.split("_")[-2]

        if verbose:
            print(f"Scan no: {self.scan_string}")
            print(f"command: {self.command}")

    def getCommand(self):
        for line in self.spec[self.scan_string].scan_header:
            if "#S" in line:
                self.command = line
        return self.command

    def getSampleName(self):
        for line in self.spec[self.scan_string].header:
            if "#F" in line:
                break
        sample = line.split("/")[-4]
        self.sample = sample
        return sample

    def getEnergy(self):
        for line in self.spec[self.scan_string].scan_header:
            if "#UMONO" in line:
                break
        self.energy = (
            float(line.split("mononrj=")[-1].split("keV")[0]) * 1e3
        )  # energy in eV
        return self.energy

    def getMotorPosition(self, motor_name):
        try:
            motor_pos = SpecH5(self.specfile)[self.scan_string]["measurement"][
                motor_name
            ][()]
        except Exception:
            motor_pos = self.spec[self.scan_string].motor_position_by_name(motor_name)
        return motor_pos

    #######################################################################################################################################
    ##########################################          Detector images functions          ################################################
    #######################################################################################################################################

    def getFirstImageFile(self):
        for line in self.spec[self.scan_string].scan_header:
            if "ULIMA" in line:
                img_first_file = line.split()[-1]
        self.img_first_file = img_first_file
        return

    def getImagesNumbers(self):
        counter_dict = {
            "eiger2M": "ei2minr",
            "mpx4": "mpx4inr",
            "mpx22": "mpx22ir",
        }

        counter = counter_dict[self.detector]

        img_nb_list = self.spec[self.scan_string].data_column_by_name(counter)
        return img_nb_list

    def getImagesFilesList(self):
        #         self.getFirstImageFile()

        prefix = self.img_first_file.split(".")[0][:-5]
        if self.path_imgs is not None:
            prefix = self.path_imgs + prefix.split("/")[-1]

        if ".edf.gz" in self.img_first_file:
            suffix = ".edf.gz"
        elif ".edf" in self.img_first_file:
            suffix = ".edf"
        elif ".h5" in self.img_first_file:
            suffix = ".h5"
        elif ".h5bs" in self.img_first_file:
            suffix = ".h5bs"
        else:
            suffix = ""

        img_path_list = []
        img_nb_list = self.getImagesNumbers()

        # Test if there's an error
        first_nb = int(self.img_first_file.split(".")[0].split("_")[-1])
        if first_nb != img_nb_list[0]:
            print("Error : first image number from GetImageNumbers seems wrong !")

        for img_nb in img_nb_list:
            img_path_list.append(prefix + "%05d" % img_nb + suffix)
        self.img_path_list = img_path_list
        self.suffix = suffix
        return

    def getImagesRaw(self):
        self.getImagesFilesList()

        for n, img_path in enumerate(self.img_path_list):
            if (
                self.suffix == ".edf"
                or self.suffix == ".edf.gz"
                or self.suffix == ""
                or self.suffix == ".h5bs"
            ):
                img = fabio.open(img_path).data
            elif self.suffix == ".h5":
                hdf5 = h5py.File(img_path)
                img = hdf5[list(hdf5.keys())[0]]["measurement"]["data"].__array__()[0]

            if n == 0:
                data = np.zeros((len(self.img_path_list),) + img.shape)
            data[n] += img

        if self.verbose:
            print("data.shape", data.shape)

        return data

    def getImages(self):
        data = self.getImagesRaw()

        if self.verbose:
            print("data.shape", data.shape)
        return data

    #######################################################################################################################################
    ##########################################                    ################################################
    #######################################################################################################################################

    def getDetCalibInfo(self):
        det_calib = {}
        for line in self.spec[self.scan_string].scan_header:
            if "#UDETCALIB" in line:
                break

        det_calib["beam_center_x"] = float(line.split()[1].split(",")[0].split("=")[1])
        det_calib["beam_center_y"] = float(line.split()[1].split(",")[1].split("=")[1])
        #         pixperdeg = float(line.split()[1].split(',')[2].split('=')[1])
        # det_distance_CC = float(line.split()[1].split(",")[3].split("=")[1])
        det_distance_COM = float(line.split()[1].split(",")[4].split("=")[1])
        det_calib["distance"] = det_distance_COM
        #         timestamp = line.split()[1].split(',')[5].split('=')[1]

        if self.detector == "mpx4" or self.detector == "mpx22":
            pixel_sizes = 55e-6
        elif self.detector == "eiger2M":
            pixel_sizes = 75e-6
        else:
            raise ValueError("detector's pixel sizes unknown")
        det_calib["x_pixel_size"] = pixel_sizes
        det_calib["y_pixel_size"] = pixel_sizes

        self.det_calib = det_calib
        if self.verbose:
            print(self.det_calib)
        return det_calib


#######################################################################################################################################
##########################################          Standard Scan class          ################################################
#######################################################################################################################################


class StandardScan_spec(Scan_spec):
    def __init__(self, filename, scan_nb, path_imgs=None, verbose=False):
        super().__init__(filename, scan_nb, path_imgs=path_imgs, verbose=verbose)

        _ = self.getDscanMotorPosition()

    def getDscanMotorPosition(self):
        motor_name = self.command.split()[3]
        motor = self.spec[self.scan_string].data_column_by_name(motor_name)

        if self.verbose:
            print("motor : {}".format(motor_name))

        self.motor = motor
        self.motor_name = motor_name

        return motor, motor_name


#     def getRoiData(self, roi_name, plot=False, fig_title=''):
#         with h5.File(self.h5file, "r") as h5f:
#             roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
#         if plot:
#             plt.figure()
#             plt.plot(self.motor, roidata, ".-")
#             plt.xlabel(self.motor_name, fontsize=15)
#             plt.ylabel(roi_name, fontsize=15)
#             plt.title(fig_title, fontsize=15)
#         return roidata


##################################################################################################################################
#####################################           Dmesh scan          #######################################################
##################################################################################################################################


class DmeshScan_spec(Scan_spec):
    def __init__(self, filename, scan_nb, path_imgs=None, verbose=True):
        super().__init__(filename, scan_nb, path_imgs=path_imgs, verbose=verbose)

        _ = self.getMeshMotorPosition()

    def getMeshMotorPosition(self):
        motor1_name = self.command.split()[3]
        motor2_name = self.command.split()[7]

        motor1 = self.spec[self.scan_string].data_column_by_name(motor1_name)
        motor2 = self.spec[self.scan_string].data_column_by_name(motor2_name)

        shape = (
            int(self.command.split()[-2]) + 1,
            int(self.command.split()[6]) + 1,
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

    def getImages(self):
        data = self.getImagesRaw()
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
        return data


##################################################################################################################################
#####################################           SXDM scan          #######################################################
##################################################################################################################################


class SXDM_Scan_spec(Scan_spec):
    def __init__(self, filename, scan_nb, path_imgs=None, verbose=True):
        super().__init__(
            filename,
            scan_nb,
            path_imgs=path_imgs,
            verbose=verbose,
            search_img_first_file=False,
        )

        self.getSXDMShape()
        self.getSXDMMotorPosition()

    def getSXDMShape(self):
        self.sxdm_shape = (
            int(self.command.split()[-2]),
            int(self.command.split()[6]),
        )
        return

    def getSXDMMotorPosition(self):
        """
        Not great. I do that through the command
        """
        self.motor1_name = self.command.split()[3]
        self.motor2_name = self.command.split()[7]

        motor1_start = float(self.command.split()[4])
        motor1_end = float(self.command.split()[5])
        motor1 = np.linspace(motor1_start, motor1_end, self.sxdm_shape[0])

        motor2_start = float(self.command.split()[8])
        motor2_end = float(self.command.split()[9])
        motor2 = np.linspace(motor2_start, motor2_end, self.sxdm_shape[1])

        motor1, motor2 = np.meshgrid(motor1, motor2, indexing="ij")
        self.motor1 = motor1
        self.motor2 = motor2
        return

    def getEDFFile(self):
        for line in self.spec[self.scan_string].scan_header:
            if "imageFile" in line:
                break

        if self.path_imgs is None:
            self.path_imgs = line.split("dir[")[1].split("]")[0]
        prefix = line.split("prefix[")[1].split("]")[0]
        # idxFmt = line.split("idxFmt[")[1].split("]")[0]
        nextNr = int(line.split("nextNr[")[1].split("]")[0])
        suffix = line.split("suffix[")[1].split("]")[0]

        self.edf_file = f"{self.path_imgs}{prefix}{nextNr:05d}{suffix}"
        return

    def getImages(self):
        self.getEDFFile()
        edf = fabio.open(self.edf_file)
        for n in range(edf.nframes):
            img = edf.getframe(n).data

            if n == 0:
                data = np.zeros((edf.nframes,) + img.shape)

            data[n] += img

        data = np.reshape(data, self.sxdm_shape + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
        return data


##################################################################################################################################
#################################           General open scan function          ##################################################
##################################################################################################################################


def openScan(filename, scan_nb, verbose=False, path_imgs=None):
    command = get_command_spec(scan_nb, filename)

    if (
        "scan" in command
        and "lookupscan" not in command
        and "loopscan" not in command
        and "_pscando" not in command
    ):
        return StandardScan_spec(
            filename, scan_nb, verbose=verbose, path_imgs=path_imgs
        )

    #     if "lookupscan" in command:
    #         return LookupScan(filename, scan_nb, verbose=verbose)

    if "mesh" in command:
        return DmeshScan_spec(filename, scan_nb, verbose=verbose, path_imgs=path_imgs)

    if "_pscando" in command:
        return SXDM_Scan_spec(filename, scan_nb, verbose=verbose, path_imgs=path_imgs)


#     if "ct" in command:
#         return Scan_ct(filename, scan_nb, verbose=verbose)

#     if "loopscan" in command:
#         return Scan(filename, scan_nb, verbose=verbose)
