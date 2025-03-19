import numpy as np
import pylab as plt
import os
import h5py as h5
import hdf5plugin
from datetime import datetime

from matplotlib.colors import LogNorm

############################################################################################################################
#######################################    Main scan class                ##################################################
############################################################################################################################


class Scan:
#     def __init__(self, folder_name, scan_nb, verbose=True):
    def __init__(self, folder_name, verbose=True):

        self.data_type = 'PETRA'
        self.folder_name = folder_name
        if self.folder_name[-1] == '/':
            self.folder_name = self.folder_name[:-1]
#         self.scan_nb = scan_nb
        self.sample = self.folder_name.split('/')[-1][:-6]
        self.scan_nb = int(self.folder_name.split('/')[-1][-5:])
        self.verbose = verbose

        self.get_master_h5file()
        self.h5file = self.master_h5file # to avoid issues later during BCDI processing
        self.get_data_h5file_list()
        self.get_fio_file()
        self.get_command()
        
        if self.verbose:
            print('fio file :', self.fiofile)
            print('master h5 :', self.master_h5file)
            print('data h5 :', self.data_h5file_list)
            print('scan command :', self.command)

    def get_master_h5file(self):
        filename = f'{self.folder_name}/e4m/{self.folder_name.split("/")[-1]}_master.h5'
        self.master_h5file = filename
        return
    
    def get_data_h5file_list(self):
        filename_list_all = sorted(os.listdir(f'{self.folder_name}/e4m/'))
        filename_list = []
        for filename in filename_list_all:
            if '_data_' in filename:
                filename_list.append(f'{self.folder_name}/e4m/' + filename)
#         filename = f'{self.folder_name}/e4m/{self.folder_name.split("/")[-1]}_data_{1:06d}.h5'
        self.data_h5file_list = filename_list
        return

    def get_fio_file(self):
        filename = f'{self.folder_name}/{self.folder_name.split("/")[-1]}.fio'
        self.fiofile = filename
        return

    def get_command(self):
        file = open(self.fiofile)
        lines = file.readlines()
        for n in range(len(lines)):
            if '%c' in lines[n]:
                break
        self.command = lines[n+1]
        return

    def getEnergy(self):
        with h5.File(self.master_h5file, "r") as h5f:
            energy = (
                12.39842
                /
                h5f["entry/instrument/beam/incident_wavelength"][()]) * 1e3  # energy in eV
        self.energy = energy
        return energy

    def getMotorStartPosition(self, motor_name):
        file = open(self.fiofile)
        for line in file.readlines():
            if f'{motor_name} =' in line:
                break
        motor_pos = float(line.split('= ')[-1])
        return motor_pos

    def getAllMotorDictionary(self):
        file = open(self.fiofile)
        motor_dict = {}
        column_dict = {}
        # Get column position
        for line in file.readlines():
            if 'Col ' in line:
                motor = line.split()[2]
                column = line.split()[1]
                column_dict[motor] = int(column)

        # get motors values
        file = open(self.fiofile)
        lines = file.readlines()
        for n in range(len(lines)):
            if '%d' in lines[n]:
                break
        n += len(column_dict.keys())+1

        for motor in column_dict.keys():
            motor_dict[motor] = []

        for line in lines[n:] :
            if '! Acquisition ended' not in line:
                for motor in column_dict.keys():
                    motor_dict[motor].append(float(line.split()[column_dict[motor]-1]))
        return motor_dict
    
    def getMotorPosition(self, motor_name):
        try:
            motor = self.getAllMotorDictionary()[motor_name]
        except:
            motor = self.getMotorStartPosition(motor_name)
        return motor

    def getDetCalibInfo(self):
        det_calib = {}
        with h5.File(self.master_h5file, "r") as h5f:
            det_calib["distance"] = h5f[
                "entry/instrument/detector/detector_distance"
            ][()]
            det_calib["beam_center_x"] = h5f[
                "entry/instrument/detector/beam_center_x"
            ][()]
            det_calib["beam_center_y"] = h5f[
                "entry/instrument/detector/beam_center_y"
            ][()]
            det_calib["x_pixel_size"] = h5f[
                "entry/instrument/detector/x_pixel_size"
            ][()]
            det_calib["y_pixel_size"] = h5f[
                "entry/instrument/detector/y_pixel_size"
            ][()]
        self.det_calib = det_calib
        return det_calib
    
    def getImageRaw(self, apply_mask = True, 
                  roi=None):
        detector_shape = (2167, 2070)
        if roi is None:
            roi = [0, detector_shape[0], 0, detector_shape[1]]
        else:
            detector_shape = (roi[1]-roi[0], roi[3]-roi[2]) 
        if 'dmesh' in self.command:
            nb_elements = (int(self.command.split()[4]) + 1) * (int(self.command.split()[8]) + 1) 
        if 'scan' in self.command and not '2scan' in self.command:
            nb_elements = int(self.command.split()[4])
        if '2scan' in self.command:
            nb_elements = int(self.command.split()[-2]) + 1

        if len(self.data_h5file_list)==1:
            with h5.File(self.data_h5file_list[0], "r") as h5f:
                data = h5f['entry/data/data'][:, roi[0]:roi[1], roi[2]:roi[3]]
        else:
            data = np.zeros((nb_elements,) + detector_shape) 
            for ii, data_h5file in enumerate(self.data_h5file_list):
                with h5.File(data_h5file, "r") as h5f:
                    if ii == len(self.data_h5file_list)-1:
                        data[ii*2000:] += h5f['entry/data/data'][:, roi[0]:roi[1], roi[2]:roi[3]]
                    else:
                        data[ii*2000:(ii+1)*2000] += h5f['entry/data/data'][:, roi[0]:roi[1], roi[2]:roi[3]]
        if apply_mask:
            mask = (data[0]==np.max(data[0]))
            data = data * (1-mask[None])
            self.mask = mask
        return data
    
############################################################################################################################
#######################################    Standard scan                ####################################################
############################################################################################################################
    
class StandardScan(Scan):
#     def __init__(self, folder_name, scan_nb, verbose=False):
#         super().__init__(folder_name, scan_nb, verbose=verbose)
    def __init__(self, folder_name, verbose=False):
        super().__init__(folder_name, verbose=verbose)

        _ = self.getDscanMotorPosition()

    def getDscanMotorPosition(self):
        motor_name = self.command.split()[1]
        motor = self.getMotorPosition(motor_name)

        if self.verbose:
            print("motor : {}".format(motor_name))

        self.motor = np.array(motor)
        self.motor_name = motor_name

        return motor, motor_name
    
    def getImages(self, apply_mask=True, roi=None):
        return self.getImageRaw(apply_mask=apply_mask, roi=roi)
    
############################################################################################################################
#####################################            D2scan                 ####################################################
############################################################################################################################    
    
class D2Scan(Scan):
#     def __init__(self, folder_name, scan_nb, verbose=False):
#         super().__init__(folder_name, scan_nb, verbose=verbose)
    def __init__(self, folder_name, verbose=False):
        super().__init__(folder_name, verbose=verbose)

        _ = self.getD2scanMotorsPosition()

    def getD2scanMotorsPosition(self):
        self.motor_name1 = self.command.split()[1]
        self.motor1 = self.getMotorPosition(self.motor_name1)
        
        self.motor_name2 = self.command.split()[4]
        self.motor2 = self.getMotorPosition(self.motor_name2)

        if self.verbose:
            print("motor1 : {}".format(self.motor_name1))
            print("motor2 : {}".format(self.motor_name2))
        return
    
    def getImages(self, apply_mask=True, roi=None):
        return self.getImageRaw(apply_mask=apply_mask, roi=roi)
    
    
def openScan(folder_name, verbose=False):
    scan = Scan(folder_name, verbose=False)
    command = scan.command

    if "scan" in command and not '2scan':
        return StandardScan(folder_name, verbose=verbose)

    if "mesh" in command:
        return DmeshScan(folder_name, verbose=verbose)
    
    if '2scan' in command:
        return D2Scan(folder_name, verbose=verbose)
    
    return
    
    
############################################################################################################################
#######################################       Dmesh scan                ####################################################
############################################################################################################################


class DmeshScan(Scan):
    def __init__(self, folder_name, verbose=False):
        super().__init__(folder_name, verbose=verbose)

        _ = self.getMeshMotorPosition()

    def getMeshMotorPosition(self):
        motor1_name = self.command.split()[1]
        motor2_name = self.command.split()[5]

        motor1 = self.getMotorPosition(motor1_name)
        motor2 = self.getMotorPosition(motor2_name)

        shape = (int(self.command.split()[4]) + 1, int(self.command.split()[8]) + 1)
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
        mesh_map = np.nansum(data,axis=(-1,-2))
        det_sum = np.nansum(data,axis=(0,1))
        return data, mesh_map, det_sum
    
    def plot_dmesh_map_det_sum(self, mesh_map, det_sum, fw=8,
                              return_fig=False):
        fig,ax = plt.subplots(1,2, figsize=(2*fw,fw))

        extent = [np.min(self.motor1), np.max(self.motor1),
                 np.max(self.motor2), np.min(self.motor2)]
        ax[0].matshow(mesh_map, norm=LogNorm(), extent=extent)
        ax[0].set_xlabel(f'{self.motor1_name}', fontsize=3*fw)
        ax[0].set_ylabel(f'{self.motor2_name}', fontsize=3*fw)

        ax[1].matshow(det_sum, norm=LogNorm())
        
        if return_fig:
            return fig,ax
        else:
            return
    
############################################################################################################################
#################################           Other functions                   ##############################################
############################################################################################################################

def openScan(folder_name, verbose=False):
    scan = Scan(folder_name, verbose=False)
    command = scan.command

    if "scan" in command:
        return StandardScan(folder_name, verbose=verbose)

    if "mesh" in command:
        return DmeshScan(folder_name, verbose=verbose)
    
    return
