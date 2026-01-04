#####################################################################################################################
# Summary: Utility Functions for Reciprocal and Real Space Optimization, Data Processing, and Visualization
#####################################################################################################################

# 1. Voxel Optimization in Reciprocal and Real Space:
#    - `optimize_voxel_reciproque_space`: Finds optimal voxel size along a given direction in reciprocal space.
#    - `optimize_voxel_real_space`: Adjusts voxel size along a single direction in real space.
#    - `optimize_voxel_real_space_all_directions`: Optimizes voxel size in all three real-space directions.

# 2. Data Resampling and Adjustments:
#    - `adjust_voxel_size`: Rescales voxel size in a 3D array to match a target voxel size.
#    - `array_to_dict`: Converts a NumPy array into a dictionary.

# 3. Data Reading and Processing:
#    - `orth_ID27_gridder_def_new`: Computes reciprocal space coordinates for ID27 beamline data.
#    - `orth_ID27_gridder_def`: Alternative version for ID27 beamline data orthonormalization.
#    - `get_abc_direct_space`: Computes direct-space coordinates from reciprocal-space data.
#    - `get_abc_direct_space_sixs2019`: Specialized function for direct-space computation in SIXS 2019 data.
#    - `orth_sixs2019_gridder_def`: Converts SIXS 2019 experimental data to reciprocal space.

# 4. Data Visualization:
#    - `plot_qxqyqzI`: Plots 3D intensity maps in reciprocal space.
#    - `plotqxqyqzi_imshow`: Displays 3D intensity slices using `imshow`.
#    - `ortho_data_phaserho_sixs2019`: Processes and plots SIXS 2019 orthogonalized phase/intensity data.

# 5. Utility Functions for X-ray Diffraction Analysis:
#    - `detectframe_to_labframe`: Converts detector frame data to lab frame.
#    - `orth_SIXS2019`: Orthonormalization for SIXS 2019 data.
#    - `orth_SIXS2019_gridder_def`: Gridder function for SIXS 2019 data processing.

#####################################################################################################################
# Latest Updates:
# - Added optimized voxel search for both reciprocal and real space.
# - Improved gridder-based data transformation.
# - Updated plotting functions for better visualization.
#####################################################################################################################



import numpy as np
# from numpy import array
import matplotlib.pyplot as plt
import xrayutilities as xu
from cdi_dislo.ewen_utilities.plot_utilities import plot_3D_projections, plot_2D_slices_middle_one_array3D

#####################################################################################################################
#####################################################################################################################
def optimize_voxel_reciproque_space(
        a, b, c, density, direction, start, end,
        voxel_to_found=60 * 1e-5, round_data=7
        ):
    import numpy as np
    import xrayutilities as xu
    
    n_list = np.round(np.arange(start, end, 1), 1)
    result = []
    nx, ny, nz = density.shape
    for indice in n_list:
        if direction == 0:
            gridder = xu.FuzzyGridder3D(int(indice), int(ny), int(nz))
        elif direction == 1:
            gridder = xu.FuzzyGridder3D(int(nx), int(indice), int(nz))
        elif direction == 2:
            gridder = xu.FuzzyGridder3D(int(nx), int(ny), int(indice))
        else:
            print("wrong indice")
            break
        gridder(a, b, c, density)
        x = gridder.xaxis
        y = gridder.yaxis
        z = gridder.zaxis
        voxel_size = np.array(
            [(x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y), (z[-1] - z[0]) / len(z)]
        )
        # print(voxel_size)
        result.append(
            np.around(abs(voxel_size[direction] - voxel_to_found), round_data)
        )
    result = np.array(result)
    return (
        n_list[np.where(result == result.min())[0][0]],
        result[np.where(result == result.min())[0][0]],
    )
def adjust_voxel_size(data, qx_lin, qy_lin, qz_lin, target_voxel_size):
    """
    Adjust the voxel size of a 3D numpy array by applying zoom, to match a target voxel size.

    :param data: 3D numpy array (transformed data after rotation)
    :param qx_lin, qy_lin, qz_lin: 1D numpy arrays representing the axis values along each dimension
    :param target_voxel_size: Desired voxel size (e.g., 1 / dq) for uniform voxel size
    :return: Adjusted data array, and the rescaled axis values qx_lin, qy_lin, qz_lin
    """
    import numpy as np
    from scipy.ndimage import zoom
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")

    # Calculate current step sizes in reciprocal space
    current_voxel_size_x = np.abs(qx_lin[1] - qx_lin[0])
    current_voxel_size_y = np.abs(qy_lin[1] - qy_lin[0])
    current_voxel_size_z = np.abs(qz_lin[1] - qz_lin[0])

    # Calculate scaling factors to match target voxel size
    scale_x = current_voxel_size_x / target_voxel_size
    scale_y = current_voxel_size_y / target_voxel_size
    scale_z = current_voxel_size_z / target_voxel_size

    # Verify that we have exactly three scaling factors
    scaling_factors = (scale_x, scale_y, scale_z)
    if len(scaling_factors) != data.ndim:
        raise RuntimeError("Scaling factors must match the number of data dimensions.")

    # Apply zoom with the scaling factors
    adjusted_data = zoom(
        data, scaling_factors, order=1
    )  # order=1 for linear interpolation (minimal)

    # Rescale the axis values based on the new voxel size
    new_qx_lin = np.linspace(qx_lin[0], qx_lin[-1], adjusted_data.shape[0]) # pyright: ignore[reportCallIssue, reportArgumentType]
    new_qy_lin = np.linspace(qy_lin[0], qy_lin[-1], adjusted_data.shape[1]) # pyright: ignore[reportCallIssue, reportArgumentType]
    new_qz_lin = np.linspace(qz_lin[0], qz_lin[-1], adjusted_data.shape[2]) # pyright: ignore[reportArgumentType, reportCallIssue]

    return adjusted_data, new_qx_lin, new_qy_lin, new_qz_lin
def optimize_voxel_real_space_all_directions(
        a, b, c, density, start, end,
        voxel_to_found=1e-5, round_data=7
        ):
    import numpy as np
    import xrayutilities as xu
    n_list = np.round(np.arange(start, end, 1), 1)
    result = []
    nx, ny, nz = density.shape
    # Iterate over all potential values of nx, ny, and nz
    for nx_val in n_list:
        for ny_val in n_list:
            for nz_val in n_list:
                gridder = xu.FuzzyGridder3D(int(nx_val), int(ny_val), int(nz_val))
                gridder(a, b, c, density)
                x = gridder.xaxis
                y = gridder.yaxis
                z = gridder.zaxis
                voxel_size = (
                    np.array(
                        [
                            (x[-1] - x[0]) / len(x),
                            (y[-1] - y[0]) / len(y),
                            (z[-1] - z[0]) / len(z),
                        ]
                    )
                    / 10
                )
                # Calculate the deviation from the target voxel size for each direction
                deviation = np.around(np.abs(voxel_size - voxel_to_found), round_data)

                # Append the result with voxel sizes and the deviation sum (to minimize total deviation)
                result.append((nx_val, ny_val, nz_val, voxel_size, np.sum(deviation)))

    # Convert to array and find the configuration with the minimum deviation
    result = np.array(result, dtype=object)
    min_idx = np.argmin(result[:, -1])  # Get the index of the minimum deviation
    optimal_values = result[min_idx]

    return {
        "optimal_nx": int(optimal_values[0]),
        "optimal_ny": int(optimal_values[1]),
        "optimal_nz": int(optimal_values[2]),
        "voxel_sizes": optimal_values[3],
        "total_deviation": optimal_values[4],
    }
def optimize_voxel_real_space(
        a, b, c, density, direction, start, end, 
        voxel_to_found=10, round_data=2
        ):
    import numpy as np
    import xrayutilities as xu
    n_list = np.round(np.arange(start, end, 1), 1)
    result = []
    nx, ny, nz = density.shape
    for indice in n_list:
        if direction == 0:
            gridder = xu.FuzzyGridder3D(int(indice), int(ny), int(nz))
        elif direction == 1:
            gridder = xu.FuzzyGridder3D(int(nx), int(indice), int(nz))
        elif direction == 2:
            gridder = xu.FuzzyGridder3D(int(nx), int(ny), int(indice))
        else:
            print("wrong indice")
            break
        gridder(a, b, c, density)
        x = gridder.xaxis
        y = gridder.yaxis
        z = gridder.zaxis
        voxel_size = (
            np.array(
                [
                    (x[-1] - x[0]) / len(x),
                    (y[-1] - y[0]) / len(y),
                    (z[-1] - z[0]) / len(z),
                ]
            )
            / 10
        )
        # print(voxel_size)
        result.append(
            np.around(np.abs(voxel_size[direction] - voxel_to_found), round_data)
        )
    result = np.array(result)
    return (
        n_list[np.where(result == result.min())[0][0]],
        result[np.where(result == result.min())[0][0]],
    )
def array_to_dict(array):
    """Converts a NumPy array to a dictionary.

    Args:
      array: A NumPy array.

    Returns:
      A dictionary.
    """

    if array.dtype.names is None:
        array = array.flatten()
        dictionary = {}
        for index, value in enumerate(array):
            dictionary[index] = value
    else:
        dictionary = {}
        for key, value in zip(array.dtype.names, array.tolist()):
            dictionary[key] = value
    return dictionary
#####################################################################################################################
#####################################################################################################################
######################################          plotting                    #########################################
#####################################################################################################################
#####################################################################################################################
def plot_qxqyqzI(qx_, qy_, qz_, Int, i_scan):
    import xrayutilities as xu
    import matplotlib.pyplot as plt
    # affichage donnees interpolees
    fig1 = plt.figure(1, figsize=(20, 4))
    plt.subplot(1, 3, 1)

    plt.contourf(qx_, qy_, xu.maplog(Int.sum(axis=2)).T, 150, cmap="jet")
    plt.xlabel(r"Q$_x$ ($1/\AA$)")
    plt.ylabel(r"Q$_y$ ($1/\AA$)")
    plt.colorbar()
    plt.axis("tight")
    plt.subplot(1, 3, 2)
    plt.contourf(qx_, qz_, xu.maplog(Int.sum(axis=1)).T, 150, cmap="jet")
    plt.xlabel(r"Q$_X$ ($1/\AA$)")
    plt.ylabel(r"Q$_Z$ ($1/\AA$)")
    plt.colorbar()
    plt.axis("tight")
    plt.subplot(1, 3, 3)
    plt.contourf(qy_, qz_, xu.maplog(Int.sum(axis=0)).T, 150, cmap="jet")
    plt.xlabel(r"Q$_Y$ ($1/\AA$)")
    plt.ylabel(r"Q$_Z$ ($1/\AA$)")
    plt.colorbar()
    plt.axis("tight")
    fig1.suptitle(r"Scan_" + str(i_scan))
    plt.show()
def plotqxqyqzi_imshow(Int, i_scan, vmax=5):
    import xrayutilities as xu
    import matplotlib.pyplot as plt
    # affichage donnees interpolees
    f_s = 16
    fig2 = plt.figure(1, figsize=(20, 4))
    ax = plt.subplot(1, 3, 1)
    im = ax.imshow(xu.maplog(Int.sum(axis=2)), vmin=0, vmax=vmax, cmap="jet")
    plt.xlabel("indice[1]", fontsize=f_s)
    plt.ylabel("indice[0]", fontsize=f_s)
    plt.axis("tight")
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)

    plt.grid(alpha=0.01)
    ax = plt.subplot(1, 3, 2)
    im = ax.imshow(xu.maplog(Int.sum(axis=1)), cmap="jet", vmin=0, vmax=vmax)
    plt.xlabel("indice[2]", fontsize=f_s)
    plt.ylabel("indice[0]", fontsize=f_s)
    plt.axis("tight")
    plt.grid(alpha=0.01)
    ax.invert_yaxis()
    ax.tick_params(labelsize=f_s)

    ax = plt.subplot(1, 3, 3)
    im = ax.imshow(xu.maplog(Int.sum(axis=0)), cmap="jet", vmin=0, vmax=vmax)
    plt.xlabel("indice[2]", fontsize=f_s)
    plt.ylabel("indice[1]", fontsize=f_s)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.invert_yaxis()
    plt.axis("tight")
    ax.tick_params(labelsize=f_s)

    plt.grid(alpha=0.01)
    fig2.suptitle(r"Scan " + str(i_scan), fontsize=f_s)
    return fig2
#####################################################################################################################
#####################################################################################################################
###################################### Latest for id27      utility 2025    #########################################
#####################################################################################################################
#####################################################################################################################
def orth_ID27_gridder_def_new(
        ndata, 
        methode="Xrayutility", omgstep=9.99991e-03,
        cch_detec=[700, 300]
        ):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage  import center_of_mass as C_O_M
    import xrayutilities as xu


    cx, cy, cz = np.round(np.array(C_O_M(ndata))).astype("int")
    print(cx, cy, cz)
    ##### orthonormalisation d'une solution
    # vecteur du réseau réciproque
    lamb = 12.398 / 33.169
    dd = 2.24e6
    taypix = 74
    pixang = taypix / dd
    a0 = 3.924
    h, k, l = 1, 1, 1
    d = a0 / np.sqrt(h * h + k * k + l * l)
    theta = np.arcsin(lamb / (2 * d))
    omgrange = omgstep * len(ndata)
    print(f"nath range is {omgrange}")

    cx, cy, cz = np.round(np.array(C_O_M(ndata))).astype("int")
    print(cx, cy, cz)

    if methode == "Xrayutility":
        cch = [cy, cz]
        chpdeg = [522.0, 522.0]
        nrj = 33169.0  # x-ray energy in eV
        nav = [
            1,
            1,
        ]  # reduce data: number of pixels to average in each detector direction
        roi = [0, 200, 0, 200]
        # npixel=200

        size_t = len(ndata)
        eta = np.rad2deg(theta) + np.linspace(-omgrange / 2, omgrange / 2, size_t)
        phi = np.zeros(size_t) + 0.0

        L = 2.24
        delta = np.zeros(size_t) + 2 * np.rad2deg(theta) - 0.57
        nu = np.zeros((size_t)) - 3.25

        qconv = xu.experiment.QConversion(
            ["y-", "z-"], ["z-", "y-"], [1, 0, 0]
        )  # 2S+2D goniometer (simplified ID01 goniometer, sample: eta, phi detector nu,del
        nx, ny, nz = ndata.shape

        hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1], en=nrj, qconv=qconv)
        hxrd.Ang2Q.init_area( # type: ignore
            "z-",
            "y+",
            cch1=cch[0],
            cch2=cch[1],
            Nch1=ny,
            Nch2=nz,
            chpdeg1=chpdeg[0],
            chpdeg2=chpdeg[1],
            detrot=90.0,
        )
        cx, cy, cz = hxrd.Ang2Q.area(eta, phi, nu, delta) # type: ignore
        return cx, cy, cz

    else:
        #  NOUVELLE ORTHOGONALISATION PLUS JUSTE

        lamb = 12.398 / 33.169
        a0 = 3.924
        h, k, l = 1, 1, 1
        d = a0 / np.sqrt(h * h + k * k + l * l)
        theta = np.arcsin(lamb / (2 * d))

        dd = 2.24e6
        taypix = 75
        cch = cch_detec
        dbeam = [2230, 5040]
        t0, t1, t2 = ndata.shape
        C0, C1, C2 = np.indices(ndata.shape)
        size_t = len(ndata)
        X = np.zeros(ndata.shape) + dd
        Y = (dbeam[1] - cch[1] + t2 // 2 - C2) * taypix
        Z = (dbeam[0] - cch[0] + t1 // 2 - C1) * taypix

        eta = np.rad2deg(theta) + np.linspace(
            -omgrange / 2, omgrange / 2, size_t
        )  # -19.47
        ii = 0
        Xp = np.zeros(X.shape)
        Yp = np.zeros(Y.shape)
        Np = np.zeros(Y.shape)
        for angle in eta:
            nangle = angle * np.pi / 180
            Xp[ii] = X[ii] * np.cos(nangle) + Y[ii] * np.sin(nangle)
            Yp[ii] = -X[ii] * np.sin(nangle) + Y[ii] * np.cos(nangle)
            Np[ii] = np.sqrt(Xp[ii] ** 2 + Yp[ii] ** 2 + Z[ii] ** 2)
            ii += 1

        # dirmax=np.array([Xp[t0//2,t1//2,t2//2],Yp[t0//2,t1//2,t2//2],Z[t0//2,t1//2,t2//2]])
        Qx = np.zeros(X.shape)
        Qy = np.zeros(Y.shape)
        Qz = np.zeros(Z.shape)
        ii = 0
        for angle in eta:
            nangle = angle * np.pi / 180
            Qx[ii] = (2 * np.pi / lamb) * (Xp[ii] / Np[ii] - np.cos(nangle))
            Qy[ii] = (2 * np.pi / lamb) * (Yp[ii] / Np[ii] + np.sin(nangle))
            Qz[ii] = (2 * np.pi / lamb) * Z[ii] / Np[ii]
            ii += 1

        return Qx, Qy, Qz
def orth_ID27_gridder_def(ndata, methode="Xrayutility"):
    import numpy as np
    from scipy.ndimage  import center_of_mass as C_O_M
    import xrayutilities as xu
    cx, cy, cz = np.round(np.array(C_O_M(ndata))).astype("int")
    print(cx, cy, cz)
    ##### orthonormalisation d'une solution
    # vecteur du réseau réciproque
    lamb = 12.398 / 33.169
    dd = 2.24e6
    taypix = 74
    pixang = taypix / dd
    a0 = 3.924
    h, k, l = 1, 1, 1
    d = a0 / np.sqrt(h * h + k * k + l * l)
    theta = np.arcsin(lamb / (2 * d))
    omgrange = 1.0

    cx, cy, cz = np.round(np.array(C_O_M(ndata))).astype("int")
    print(cx, cy, cz)

    if methode == "Xrayutility":
        cch = [cy, cz]
        chpdeg = [522.0, 522.0]
        nrj = 33169.0  # x-ray energy in eV
        nav = [
            1,
            1,
        ]  # reduce data: number of pixels to average in each detector direction
        roi = [0, 200, 0, 200]
        # npixel=200

        size_t = len(ndata)
        eta = np.rad2deg(theta) + np.linspace(-omgrange / 2, omgrange / 2, size_t)
        phi = np.zeros(size_t) + 0.0

        L = 2.24
        delta = np.zeros(size_t) + 2 * np.rad2deg(theta) - 0.57
        nu = np.zeros((size_t)) - 3.25

        qconv = xu.experiment.QConversion(
            ["y-", "z-"], ["z-", "y-"], [1, 0, 0]
        )  # 2S+2D goniometer (simplified ID01 goniometer, sample: eta, phi detector nu,del
        nx, ny, nz = ndata.shape

        hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1], en=nrj, qconv=qconv)
        hxrd.Ang2Q.init_area( # type: ignore
            "z-",
            "y+",
            cch1=cch[0],
            cch2=cch[1],
            Nch1=ny,
            Nch2=nz,
            chpdeg1=chpdeg[0],
            chpdeg2=chpdeg[1],
            detrot=90.0,
        )
        cx, cy, cz = hxrd.Ang2Q.area(eta, phi, nu, delta) # type: ignore
        return cx, cy, cz

    else:
        # %% NOUVELLE ORTHOGONALISATION PLUS JUSTE
        taypix = 0.075
        posy = 300
        posz = 700
        ray = np.array([2240, (5040 - posy) * taypix, (2230 - posz) * taypix])
        raynorm = ray / np.linalg.norm(ray)
        tth = np.arccos(raynorm[0]) * 180 / np.pi
        print(tth)
        nath = 15 * np.pi / 180
        matchg = np.array(
            [
                [np.cos(nath), -np.sin(nath), 0],
                [np.sin(nath), np.cos(nath), 0],
                [0, 0, 1],
            ]
        ).T
        dira = np.array([1, 1, 1])
        newcc = np.dot(matchg, dira)
        print(newcc / np.linalg.norm(dira))

        # vecteur du réseau réciproque
        lamb = 12.398 / 33.169
        dd = 2.24e6
        taypix = 75
        a0 = 3.924
        h, k, l = 1, 1, 1
        d = a0 / np.sqrt(h * h + k * k + l * l)
        theta = np.arcsin(lamb / (2 * d))
        omgrange = 1.0

        cch = [700, 300]
        dbeam = [2230, 5040]
        t0, t1, t2 = ndata.shape
        C0, C1, C2 = np.indices(ndata.shape)
        size_t = len(ndata)
        X = np.zeros(ndata.shape) + dd
        Y = (dbeam[1] - cch[1] - C2) * taypix
        Z = (dbeam[0] - cch[0] - C1) * taypix

        eta = np.rad2deg(theta) + np.linspace(
            -omgrange / 2, omgrange / 2, size_t
        )  # -19.47
        ii = 0
        Xp = np.zeros(X.shape)
        Yp = np.zeros(Y.shape)
        Np = np.zeros(Y.shape)
        for angle in eta:
            nangle = angle * np.pi / 180
            Xp[ii] = X[ii] * np.cos(nangle) + Y[ii] * np.sin(nangle)
            Yp[ii] = -X[ii] * np.sin(nangle) + Y[ii] * np.cos(nangle)
            Np[ii] = np.sqrt(Xp[ii] ** 2 + Yp[ii] ** 2 + Z[ii] ** 2)
            ii += 1

        # dirmax=np.array([Xp[t0//2,t1//2,t2//2],Yp[t0//2,t1//2,t2//2],Z[t0//2,t1//2,t2//2]])
        Qx = np.zeros(X.shape)
        Qy = np.zeros(Y.shape)
        Qz = np.zeros(Z.shape)
        ii = 0
        for angle in eta:
            nangle = angle * np.pi / 180
            Qx[ii] = (2 * np.pi / lamb) * (Xp[ii] / Np[ii] - np.cos(nangle))
            Qy[ii] = (2 * np.pi / lamb) * (Yp[ii] / Np[ii] + np.sin(nangle))
            Qz[ii] = (2 * np.pi / lamb) * Z[ii] / Np[ii]
            ii += 1

        return Qx, Qy, Qz
def get_abc_direct_space(cx, cy, cz, cx0, cy0, cz0, wanted_shape=None):
    import numpy as np

    degree = 180 / np.pi
    qxCOM = cx[cx0, cy0, cz0]
    qyCOM = cy[cx0, cy0, cz0]
    qzCOM = cz[cx0, cy0, cz0]

    tayx, tayy, tayz = cx.shape
    qpeak = [qxCOM, qyCOM, qzCOM]
    # print(f"qpeak: {qpeak}")
    dhkl = 2 * np.pi / np.linalg.norm(qpeak)
    print("dhkl: ", dhkl, 3.924 / np.sqrt(3))

    p0_x, p0_y, p0_z = 0, 0, 0  # p0_x,p0_y,p0_z=cx0_-tayx//2,cy0_-tayy//2,cz0_-tayz//2
    p1_x, p1_y, p1_z = -1, 0, 0  # p1_x,p1_y,p1_z=cx0_+tayx//2,cy0_-tayy//2,cz0_-tayz//2
    p2_x, p2_y, p2_z = 0, -1, 0  # p2_x,p2_y,p2_z=cx0_-tayx//2,cy0_+tayy//2,cz0_-tayz//2
    p3_x, p3_y, p3_z = 0, 0, -1  # p3_x,p3_y,p3_z=cx0_-tayx//2,cy0_-tayy//2,cz0_+tayz//2

    P0 = np.array([cx[p0_x, p0_y, p0_z], cy[p0_x, p0_y, p0_z], cz[p0_x, p0_y, p0_z]])
    P1 = np.array([cx[p1_x, p1_y, p1_z], cy[p1_x, p1_y, p1_z], cz[p1_x, p1_y, p1_z]])
    P2 = np.array([cx[p2_x, p2_y, p2_z], cy[p2_x, p2_y, p2_z], cz[p2_x, p2_y, p2_z]])
    P3 = np.array([cx[p3_x, p3_y, p3_z], cy[p3_x, p3_y, p3_z], cz[p3_x, p3_y, p3_z]])
    dqx = P1 - P0
    dqy = P2 - P0
    dqz = P3 - P0
    print("voxel size based on initial coordinate")
    print(
        2 * np.pi / np.linalg.norm(dqx),
        2 * np.pi / np.linalg.norm(dqy),
        2 * np.pi / np.linalg.norm(dqz),
    )
    vol = np.dot(dqx, np.cross(dqy, dqz))
    xr = 2 * np.pi * np.cross(dqy, dqz) / vol
    yr = 2 * np.pi * np.cross(dqz, dqx) / vol
    zr = 2 * np.pi * np.cross(dqx, dqy) / vol
    alp = np.arccos(np.dot(yr, zr) / (np.linalg.norm(yr) * np.linalg.norm(zr))) * degree
    bet = np.arccos(np.dot(xr, zr) / (np.linalg.norm(xr) * np.linalg.norm(zr))) * degree
    gam = np.arccos(np.dot(xr, yr) / (np.linalg.norm(xr) * np.linalg.norm(yr))) * degree

    print(np.linalg.norm(xr), np.linalg.norm(yr), np.linalg.norm(zr))
    print(f"real space voxel angles: {alp,bet,gam}")
    if wanted_shape:
        print(f"wanted shape of data is {wanted_shape}")
    else:
        wanted_shape = cx.shape

    frame = np.indices(wanted_shape)
    real_x_, real_y_, real_z_ = frame[0], frame[1], frame[2]
    a = real_x_ * xr[0] + real_y_ * yr[0] + real_z_ * zr[0]
    b = real_x_ * xr[1] + real_y_ * yr[1] + real_z_ * zr[1]
    c = real_x_ * xr[2] + real_y_ * yr[2] + real_z_ * zr[2]
    return a, b, c
#####################################################################################################################
#####################################################################################################################
###################################### Latest for sixs 2019 utility 2025    #########################################
#####################################################################################################################
#####################################################################################################################
def get_abc_direct_space_sixs2019(
        cx, cy, cz, cx0, cy0, cz0,
        mu_range_trigger=False, wanted_shape=None, print_log=True
        ):
    import numpy as np
    degree = 180 / np.pi
    tayx, tayy, tayz = wanted_shape # type: ignore

    qxCOM = cx[cx0, cy0, cz0]
    qyCOM = cy[cx0, cy0, cz0]
    qzCOM = cz[cx0, cy0, cz0]
    qpeak = [qxCOM, qyCOM, qzCOM]
    dhkl = 2 * np.pi / np.linalg.norm(qpeak)
    if print_log:
        print(f"qpeak: {qpeak}")
        print("dhkl: ", dhkl, 3.924 / np.sqrt(3))

    p0_x = cx0 - tayx // 2
    p0_y = cy0 - tayy // 2
    p0_z = cz0 - tayz // 2

    p1_x = cx0 + tayx // 2
    p1_y = cy0 - tayy // 2
    p1_z = cz0 - tayz // 2

    p2_x = cx0 - tayx // 2
    p2_y = cy0 + tayy // 2
    p2_z = cz0 - tayz // 2

    p3_x = cx0 - tayx // 2
    p3_y = cy0 - tayy // 2
    p3_z = cz0 + tayz // 2

    if print_log:
        print(
            f"points selected: {(p0_x,p0_y,p0_z)} {(p1_x,p1_y,p1_z)} {(p2_x,p2_y,p2_z)} {(p3_x,p3_y,p3_z)}"
        )
        print(f"position of COM: {(cx0,cy0,cz0)} ")

    P0 = np.array([cx[p0_x, p0_y, p0_z], cy[p0_x, p0_y, p0_z], cz[p0_x, p0_y, p0_z]])
    P1 = np.array([cx[p1_x, p1_y, p1_z], cy[p1_x, p1_y, p1_z], cz[p1_x, p1_y, p1_z]])
    P2 = np.array([cx[p2_x, p2_y, p2_z], cy[p2_x, p2_y, p2_z], cz[p2_x, p2_y, p2_z]])
    P3 = np.array([cx[p3_x, p3_y, p3_z], cy[p3_x, p3_y, p3_z], cz[p3_x, p3_y, p3_z]])
    dqx = P1 - P0
    dqy = P2 - P0
    dqz = P3 - P0
    if print_log:
        print(f"base vector reciprocal space: {dqx,dqy,dqz}")
        print(
            f"voxel size based on initial coordinate: {2*np.pi/np.linalg.norm(dqx),2*np.pi/np.linalg.norm(dqy),2*np.pi/np.linalg.norm(dqz)}"
        )

    vol = np.dot(dqx, np.cross(dqy, dqz))
    xr = 2 * np.pi * np.cross(dqy, dqz) / vol
    yr = 2 * np.pi * np.cross(dqz, dqx) / vol
    zr = 2 * np.pi * np.cross(dqx, dqy) / vol
    alp = np.arccos(np.dot(yr, zr) / (np.linalg.norm(yr) * np.linalg.norm(zr))) * degree
    bet = np.arccos(np.dot(xr, zr) / (np.linalg.norm(xr) * np.linalg.norm(zr))) * degree
    gam = np.arccos(np.dot(xr, yr) / (np.linalg.norm(xr) * np.linalg.norm(yr))) * degree

    if print_log:
        print(f"base vector real space: {xr,yr,zr}")
        print(np.linalg.norm(xr), np.linalg.norm(yr), np.linalg.norm(zr))
        print(f"real space voxel angles: {alp,bet,gam}")

    if wanted_shape is None:
        wanted_shape = cx.shape
    print(
        f"wanted shape of data is {wanted_shape} | experrimental data shape: {cx.shape}"
    )
    frame = np.indices(wanted_shape)
    real_x_, real_y_, real_z_ = frame[0], frame[1], frame[2]
    a = real_x_ * xr[0] + real_y_ * yr[0] + real_z_ * zr[0]
    b = real_x_ * xr[1] + real_y_ * yr[1] + real_z_ * zr[1]
    c = real_x_ * xr[2] + real_y_ * yr[2] + real_z_ * zr[2]
    return a, b, c
def orth_sixs2019_gridder_def(dim, delta, gamma, mu, cch=[193, 201], wanted_shape=None,
                             nrj = 8.5 # energie du faisceau de rayons X
                             ):
    import numpy as np
    ####
    # orthonormalisation d'une solution
    degree = np.pi / 180.0
    beta = 2 * degree
    dd = 1.215e6  # distance detecteur-echantillon en micron
    tp = 55  # taille pixel detecteur en micron
    cch = cch  # position du faisceau direct pour delta et gamma nuls
    lamb = 12.398 / nrj  # longueur d'onde correspondante en angstrom

    ############ application du masque
    X, Y = np.indices((dim[1], dim[2]))

    xp = np.array(
        [np.sin(delta) * np.cos(gamma), np.sin(delta) * np.sin(gamma), -np.cos(delta)]
    )
    yp = np.array([-np.sin(gamma), np.cos(gamma), 0])
    rdet = dd * np.array(
        [np.cos(delta) * np.cos(gamma), np.cos(delta) * np.sin(gamma), np.sin(delta)]
    )

    cx = tp * (X - cch[0]) * xp[0] + tp * (Y - cch[1]) * yp[0] + rdet[0]
    cy = tp * (X - cch[0]) * xp[1] + tp * (Y - cch[1]) * yp[1] + rdet[1]
    cz = tp * (X - cch[0]) * xp[2] + tp * (Y - cch[1]) * yp[2] + rdet[2]

    """
    TM = np.matrix.transpose(np.array([np.array([np.sin(delta) * np.cos(gamma)      ,  np.sin(delta) * np.sin(gamma)   , -np.cos(delta)]),
                                       np.array([-np.sin(gamma * degree)            ,  np.cos(gamma  * degree)         , 0             ]), 
                                       dd * np.array([np.cos(delta) * np.cos(gamma) ,  np.cos(delta) * np.sin(gamma)   , np.sin(delta) ])
                                      ]))
    print(TM)
    cx = tp * (X - cch[0]) * TM[0, 0] + tp * (Y - cch[1]) * TM[0, 1] + TM[0, 2]
    cy = tp * (X - cch[0]) * TM[1, 0] + tp * (Y - cch[1]) * TM[1, 1] + TM[1, 2]
    cz = tp * (X - cch[0]) * TM[2, 0] + tp * (Y - cch[1]) * TM[2, 1] + TM[2, 2]
    """
    r = np.sqrt(cx**2 + cy**2 + cz**2)
    qx = (2 * np.pi / lamb) * (cx / r - np.cos(beta))
    qy = (2 * np.pi / lamb) * (cy / r)
    qz = (2 * np.pi / lamb) * (cz / r + np.sin(beta))
    if wanted_shape:
        len_mu = wanted_shape[0]
    else:
        len_mu = len(mu)
    mu_ref_center = mu[len(mu) // 2]
    step_mu = np.round(np.mean(np.diff(mu)), 7)
    mu_new = np.arange(-len_mu // 2, -len_mu // 2 + len_mu, 1) * step_mu + mu_ref_center
    print(
        f" the step of rocking angle is {np.mean(np.diff(np.round(mu,5)))}  and the rounded one is {step_mu}"
    )

    # in the sample frame
    CX = np.zeros((len(mu_new), 516, 516))
    CY = np.zeros((len(mu_new), 516, 516))
    CZ = np.zeros((len(mu_new), 516, 516))
    ii = 0
    for ang in mu_new:
        CX[ii] = qx * np.cos(ang) + qy * np.sin(ang)
        CY[ii] = -qx * np.sin(ang) + qy * np.cos(ang)
        CZ[ii] = qz
        ii = ii + 1
    return CX, CY, CZ
#####################################################################################################################
#####################################################################################################################
######################################              old                     #########################################
#####################################################################################################################
#####################################################################################################################
def ortho_data_phaserho_sixs2019(
    data_allscan,
    mypath="/home/abdelrahman/data_sixs_2019/",
    save__orth="/home/abdelrahman/data_sixs_2019/results/3D/3d_updated_code_8-12-2023/",
    plot_=True,
    linecut_plotting=True,
    defined_scan=None,
    ):
    from cdi_dislo.general_utilities.cdi_dislo_utils import (
        crop_data_and_update_coordinates,
        crop_3d_obj_pos,
        pad_to_shape,   
        nan_to_zero,
        zero_to_nan,       
    )
    from cdi_dislo.orthogonalisation_handler.cdi_dislo_ortho_handler import (
        remove_phase_ramp_abd,
        getting_strain_mapvti,   
        get_het_normal_strain,       
    )

    from cdi_dislo.plotutilities.cdi_dislo_plotutilities import (
        summary_slice_plot_abd,
    )
    import matplotlib.pyplot as plt
    import time
    import os
    import h5py
    from os import listdir
    from os.path import isfile, join
    import numpy as np
    import xrayutilities as xu
    from scipy.ndimage import center_of_mass as C_O_M
    from bcdi.graph import graph_utils as gu # type: ignore
    from functools import reduce
    from bcdi.graph import linecut as linecut # type: ignore
    from cdiutils.analysis import find_isosurface
    from cdiutils.utils import fill_up_support


    d0 = 2.2655224563000917

    tick_direction = "inout"
    tick_length = 3
    tick_width = 1
    tick_spacing = 50  # for plots, in nm
    hwidth = 1
    avg_counter = 1
    isosurface_strain = 0.01
    hkl = np.array([1, 1, 1])
    degree = 180 / np.pi
    # mypath = '/home/abdelrahman/data_sixs_2019/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files = np.array([str(mypath + item) for item in onlyfiles if "ascan_mu" in item])
    files_mask = np.array(
        [str(mypath + item) for item in onlyfiles if "hotpixels" in item]
    )
    files.sort()
    scan_nb = [
        "S" + files[i][files[i].find("_mu_") + 6 : files[i].find("_mu_") + 9]
        for i in range(len(files))
    ]
    mask_diff = np.array(np.load(files_mask[0])["data"])
    mask_diff = 1 - mask_diff[np.newaxis, :, :]
    mask_diff[300:] = 0
    mask_diff[:, 300:] = 0
    degree = np.pi / 180.0
    delta_scans, gamma_scans, mu_scans, data_diff = {}, {}, {}, {}
    ii = 0
    for name in files:
        f = h5py.File(name, mode="r")
        list_elmts = [key for key in f["/com/SIXS"].keys()] # type: ignore
        data_ = np.array(f["/com/scan_data/data_02"]) * mask_diff
        delta = np.array(f["/com/SIXS/" + list_elmts[32] + "/position_pre"])[0] * degree
        gamma = np.array(f["/com/SIXS/" + list_elmts[40] + "/position_pre"])[0] * degree
        mu = np.array(f["/com/scan_data/actuator_1_1"]) * degree
        f.close()
        delta_scans = {**delta_scans, scan_nb[ii]: delta}
        gamma_scans = {**gamma_scans, scan_nb[ii]: gamma}
        mu_scans = {**mu_scans, scan_nb[ii]: mu}
        data_diff = {**data_diff, scan_nb[ii]: data_}

        ii += 1
    h_w = 40
    data_allscan_complexe = (
        np.zeros((len(data_diff), int(2 * h_w), int(2 * h_w), int(2 * h_w))) * 1j
    )
    data_allscan_displacement = np.zeros(
        (len(data_diff), int(2 * h_w), int(2 * h_w), int(2 * h_w))
    )
    data_allscan_a_spacing = np.zeros(
        (len(data_diff), int(2 * h_w), int(2 * h_w), int(2 * h_w))
    )
    data_allscan_strain_from_dspacing = np.zeros(
        (len(data_diff), int(2 * h_w), int(2 * h_w), int(2 * h_w))
    )
    data_allscan_Int = np.zeros((len(data_diff), 100, 100, 100))
    voxel_size_allscan = np.zeros((len(data_diff), 3))

    x_lin_orth_allscan, y_lin_orth_allscan, z_lin_orth_allscan, data_orth_allscan = (
        {},
        {},
        {},
        {},
    )
    qxs_cut, qys_cut, qzs_cut = [], [], []
    ii__ = 0
    best_voxel_size = None
    print(f"will process those scans {scan_nb} ")
    for i_scan in scan_nb:
        os.makedirs(save__orth + i_scan, exist_ok=True)

        scan = i_scan
        print(
            "******************************"
            + str(i_scan)
            + "******************************"
        )
        start = time.time()
        if defined_scan:
            if i_scan != defined_scan:
                continue

        cx, cy, cz = orth_sixs2019_gridder_def(
            data_diff[i_scan].shape,
            delta_scans[i_scan],
            gamma_scans[i_scan],
            mu_scans[i_scan],
        )

        gridder = xu.FuzzyGridder3D(500, 400, 400)
        gridder(cx, cy, cz, data_diff[i_scan])
        qx_lin = gridder.xaxis
        qy_lin = gridder.yaxis
        qz_lin = gridder.zaxis
        ndata = gridder.data
        del gridder
        max_int_pos = [i[0] for i in np.where(ndata == ndata.max())]
        print(max_int_pos)
        ndata, qx_lin, qy_lin, qz_lin = crop_data_and_update_coordinates(
            qx_lin, qy_lin, qz_lin, ndata, (100, 100, 100)
        )

        # %% Normalisation vecteurs du reseau reciproque
        delta_x, delta_y, delta_z = (
            2 * np.pi / (qx_lin[-1] - qx_lin[0]),
            2 * np.pi / (qy_lin[-1] - qy_lin[0]),
            2 * np.pi / (qz_lin[-1] - qz_lin[0]),
        )
        # print('Voxel Size :',delta_x,delta_y,delta_z)
        # %% Normalisation vecteurs du reseau reciproque
        delta_dx_, delta_dy_, delta_dz_ = (
            np.mean(np.diff(qx_lin)),
            np.mean(np.diff(qy_lin)),
            np.mean(np.diff(qz_lin)),
        )
        print("Reciprocal space voxel size :", delta_dx_, delta_dy_, delta_dz_)
        dqX, dqY, dqZ = (
            2 * np.pi / delta_dx_,
            2 * np.pi / delta_dy_,
            2 * np.pi / delta_dz_,
        )
        print("Inverse Step Size :", dqX, dqY, dqZ)
        cx0, cy0, cz0 = np.round(np.array(C_O_M(data_diff[i_scan]))).astype("int")
        qxCOM = cx[cx0, cy0, cz0]
        qyCOM = cy[cx0, cy0, cz0]
        qzCOM = cz[cx0, cy0, cz0]
        g_vector = [qxCOM, qyCOM, qzCOM]
        print(f"COM[qx, qz, qy] = {qxCOM:.2f}, {qzCOM:.2f}, {qyCOM:.2f}")
        distances_q = np.sqrt(
            (cx - qxCOM) ** 2 + (cy - qyCOM) ** 2 + (cz - qzCOM) ** 2
        )  # if reconstructions are centered #  and of the same shape q values will be identical
        max_z = ndata.sum(axis=0).max()
        #### plotting #######################
        if plot_:
            fig0, _, _ = gu.contour_slices(
                ndata,
                (qx_lin, qy_lin, qz_lin),
                sum_frames=True,
                title="Regridded data",
                levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=True),
                plot_colorbar=True,
                scale="log",
                is_orthogonal=True,
                reciprocal_space=True,
            )
            fig0.savefig(
                save__orth + i_scan + f"/S{i_scan}_orthogonal_intensity_qxqyqz.png"
            )
            plt.close()
            fig1 = gu.combined_plots(
                tuple_array=(cx, cy, cz),
                tuple_sum_frames=False,
                tuple_sum_axis=(0, 1, 2),
                tuple_width_v=None,
                tuple_width_h=None,
                tuple_colorbar=True,
                tuple_vmin=np.nan,
                tuple_vmax=np.nan,
                tuple_title=("qz", "qy", "qx"),
                tuple_scale="linear",
                position=[131, 132, 133],
            )
            fig1.savefig(save__orth + i_scan + f"/S{scan}_grid_q.png")
            plt.close()
            fig2, _, _ = gu.multislices_plot(
                distances_q,
                sum_frames=False,
                plot_colorbar=True,
                title="distances_q",
                scale="linear",
                vmin=np.nan,
                vmax=np.nan,
                reciprocal_space=True,
                save_as=save__orth + scan + f"/S{scan}_distances_q.png",
                ipynb_layout=True,
            )
            plt.close()

        ##############################################################################################################
        # orthonormalisation d'une solution direct space                                                             #
        ##############################################################################################################
        density = abs(data_allscan[i_scan])
        density = density * (density > 0.05 * density.max())
        nx, ny, nz = np.shape(density)
        phase = 0.0 - np.angle(np.exp(1j * np.angle(data_allscan[i_scan]))) * (
            density > 0.05 * density.max()
        )
        plot_3D_projections((density), cmap="jet")
        plot_3D_projections((phase), cmap="jet")
        plt.show()
        phase = zero_to_nan(phase)
        phase, _ = remove_phase_ramp_abd(phase)
        # phase=unwrap_phase(phase)   *(density>10).astype(float)
        density = density / np.nanmax(density)
        phase = nan_to_zero(phase)

        wanted_shape = density.shape
        c, b, a = get_abc_direct_space(
            cx, cy, cz, cx0, cy0, cz0, wanted_shape=wanted_shape
        )
        if best_voxel_size is None:
            list_voxel_to_test = np.arange(11.1, 12.1, 0.02)
            list_of_error_voxel = []

            for i, voxel in enumerate(list_voxel_to_test):
                time_start = time.time()
                nx, dif_x = optimize_voxel_real_space(
                    a, b, c, density, 0, 60, 85, voxel_to_found=voxel
                )
                ny, dif_y = optimize_voxel_real_space(
                    a, b, c, density, 1, 115, 140, voxel_to_found=voxel
                )
                nz, dif_z = optimize_voxel_real_space(
                    a, b, c, density, 2, 80, 110, voxel_to_found=voxel
                )
                list_of_error_voxel.append(dif_x + dif_y + dif_z)

                elapsed_time = np.round((time.time() - time_start) * (i + 1) / 60, 3)
                total_estimated_time = np.round(
                    len(list_voxel_to_test) * (time.time() - time_start) / 60, 3
                )
                error = dif_x + dif_y + dif_z

                print(
                    f"\rError for {voxel:.2f} nm is {error:.6f} | {elapsed_time:.2f}/ {total_estimated_time:.2f} min",
                    end="",
                    flush=True,
                )
            list_of_error_voxel = np.array(list_of_error_voxel)
            best_voxel_size = list_voxel_to_test[(np.argmin(list_of_error_voxel))]
        # best is 11.5

        nx, dif_x = optimize_voxel_real_space(
            a, b, c, density, 0, 89 - 80, 89 + 5, voxel_to_found=best_voxel_size
        )
        ny, dif_y = optimize_voxel_real_space(
            a, b, c, density, 1, 149 - 80, 149 + 5, voxel_to_found=best_voxel_size
        )
        nz, dif_z = optimize_voxel_real_space(
            a, b, c, density, 2, 107 - 80, 107 + 5, voxel_to_found=best_voxel_size
        )
        gridder = xu.FuzzyGridder3D(int(nx), int(ny), int(nz))
        gridder(a, b, c, density)
        x = gridder.xaxis
        y = gridder.yaxis
        z = gridder.zaxis
        voxel_size = (
            np.array(
                [
                    (x[-1] - x[0]) / len(x),
                    (y[-1] - y[0]) / len(y),
                    (z[-1] - z[0]) / len(z),
                ]
            )
            / 10
        )
        print("voxel size real space:")
        print(voxel_size)
        density_ortho = gridder.data
        del gridder

        gridder = xu.FuzzyGridder3D(int(nx), int(ny), int(nz))
        gridder(a, b, c, phase)
        x = gridder.xaxis
        y = gridder.yaxis
        z = gridder.zaxis
        phase_ortho = gridder.data

        obj_ortho = density_ortho * np.exp(1j * phase_ortho)
        obj_ortho = crop_3d_obj_pos(
            obj_ortho, output_shape=(2 * h_w, 2 * h_w, 2 * h_w), methods="com",
        )

        density_ortho = np.abs(obj_ortho)
        mask_ortho = np.array(density_ortho != 0).astype(float)
        density_ortho *= mask_ortho
        phase_ortho = 0.0 - np.angle(np.exp(1j * np.angle(obj_ortho))) * mask_ortho
        phase_ortho, _ = remove_phase_ramp_abd(zero_to_nan(phase_ortho)) * mask_ortho

        isosurface, fig00 = find_isosurface(density_ortho, sigma_criterion=5, plot=True)
        if isosurface < 0 or isosurface > 0.3:
            isosurface = 0.05
        fig00.savefig(save__orth + scan + f"/S{scan}_amp_dist_isosurface.png") # type: ignore

        ######################################
        # estimate the volume of the crystal #
        ######################################
        a__ = np.array(phase_ortho)
        b__ = np.array(density_ortho)
        a__ = zero_to_nan(a__)
        amp = density_ortho
        amp = amp / amp.max()
        temp_amp = np.copy(amp)
        temp_amp[amp < isosurface_strain] = np.nan
        temp_amp[np.nonzero(temp_amp)] = 1
        volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in A3
        ####################################################################################################################
        # estimate the bulk, displacement, strain, dspacing ,lattice parameter and strain based on dspacing of the crystal #
        ####################################################################################################################
        bulk = fill_up_support(density_ortho > isosurface)
        piz, piy, pix = np.round(np.array(C_O_M(bulk)), 1).astype("int")
        print(
            f"phase.max() = {phase_ortho[np.nonzero(bulk)].max():.2f} at voxel ({piz}, {piy}, {pix})"
        )
        pixel_spacing = [tick_spacing / vox for vox in voxel_size]
        displacement = a__ / np.linalg.norm(g_vector)
        strain = get_het_normal_strain(
            displacement, g_vector, voxel_size, gradient_method="hybrid"
        )
        dspacing = 2 * np.pi / np.linalg.norm(g_vector) * (1 + strain)
        lattice_parameter = np.sqrt(hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2) * dspacing
        lattice_parameter_mean = np.nanmean(lattice_parameter)
        dspacing_mean = np.nanmean(dspacing)
        strain_from_dspacing = (dspacing - dspacing_mean) / d0

        final_plots = {
            "amplitude": amp,
            "phase": a__,
            "displacement": displacement,
            "het_strain": strain,
            "lattice_parameter": lattice_parameter,
        }

        ######################################
        # Plot results of orthogonalisation  #
        ######################################
        if plot_:
            fig3, _, _ = gu.multislices_plot(
                bulk,
                sum_frames=False,
                title="Orthogonal bulk",
                vmin=0,
                vmax=1,
                is_orthogonal=True,
                reciprocal_space=False,
            )
            fig3.text(0.60, 0.45, "Scan " + str(scan), size=20)
            fig3.text(
                0.60, 0.40, "Bulk - isosurface=" + str("{:.2f}".format(0.5)), size=20
            )
            fig3.savefig(save__orth + i_scan + f"/S{i_scan}_bulk.png")
            plt.close()

            # amplitude
            fig4, _, _ = gu.multislices_plot(
                zero_to_nan(amp),
                sum_frames=False,
                title="Normalized orthogonal amp",
                vmin=0,
                vmax=1,
                tick_direction=tick_direction,
                tick_width=tick_width,
                tick_length=tick_length,
                pixel_spacing=pixel_spacing,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                ipynb_layout=True,
            )
            fig4.text(0.60, 0.45, f"Scan {scan}", size=20)
            fig4.text(
                0.60,
                0.40,
                f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
                size=20,
            )
            fig4.text(0.60, 0.35, f"Ticks spacing={tick_spacing} nm", size=20)
            fig4.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
            fig4.text(
                0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20
            )
            fig4.savefig(save__orth + i_scan + f"/S{i_scan}_amp.png")
            plt.close()

            # amplitude histogram
            fig5, ax = plt.subplots(1, 1)
            ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
            ax.set_ylim(bottom=1)
            ax.tick_params(
                labelbottom=True,
                labelleft=True,
                direction="out",
                length=tick_length,
                width=tick_width,
            )
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            fig5.savefig(save__orth + i_scan + f"/S{i_scan}_histo_amp.png")
            plt.close()

            # phase
            fig6, _, _ = gu.multislices_plot(
                zero_to_nan(displacement * bulk),
                sum_frames=False,
                title="Orthogonal displacement (A)",
                vmin=-2 * np.pi // np.linalg.norm(g_vector),
                vmax=2 * np.pi // np.linalg.norm(g_vector),
                # vmin=-0.5,vmax=0.5,
                tick_direction=tick_direction,
                tick_width=tick_width,
                tick_length=tick_length,
                pixel_spacing=pixel_spacing,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                ipynb_layout=True,
            )
            fig6.text(0.60, 0.30, f"Scan {i_scan}", size=20)
            fig6.text(
                0.60,
                0.25,
                f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
                size=20,
            )
            fig6.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
            fig6.text(
                0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20
            )
            fig6.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
            fig6.savefig(save__orth + i_scan + f"/S{i_scan}_displacement.png")
            plt.close()

            fig7, _, _ = gu.multislices_plot(
                zero_to_nan(strain * 100 * bulk),
                sum_frames=False,
                title="Orthogonal strain (%)",
                vmin=-15,
                vmax=15,
                tick_direction=tick_direction,
                tick_width=tick_width,
                tick_length=tick_length,
                pixel_spacing=pixel_spacing,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                ipynb_layout=True,
            )
            fig7.text(0.60, 0.30, f"Scan {i_scan}", size=20)
            fig7.text(
                0.60,
                0.25,
                f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
                size=20,
            )
            fig7.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
            fig7.text(
                0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20
            )
            fig7.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
            fig7.savefig(save__orth + scan + f"/S{i_scan}_strain.png")
            plt.close()

            fig8, _, _ = gu.multislices_plot(
                lattice_parameter,
                sum_frames=False,
                title="Lattice parameter (A)",
                vmin=3.9,
                vmax=3.924,
                tick_direction=tick_direction,
                tick_width=tick_width,
                tick_length=tick_length,
                pixel_spacing=pixel_spacing,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                ipynb_layout=True,
            )
            fig8.text(0.60, 0.30, f"Scan {scan}", size=20)
            fig8.text(
                0.60,
                0.25,
                f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
                size=20,
            )
            fig8.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
            fig8.text(
                0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20
            )
            fig8.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
            fig8.savefig(save__orth + scan + f"/S{scan}_lattice_parameter.png")
            plt.close()

            fig9, _, _ = gu.multislices_plot(
                strain_from_dspacing,
                sum_frames=False,
                title="Orthogonal strain d-spacing based calculation",
                vmin=-0.05,
                vmax=0.05,
                tick_direction=tick_direction,
                tick_width=tick_width,
                tick_length=tick_length,
                pixel_spacing=pixel_spacing,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                ipynb_layout=True,
            )
            fig9.text(0.60, 0.30, f"Scan {scan}", size=20)
            fig9.text(
                0.60,
                0.25,
                f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
                size=20,
            )
            fig9.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
            fig9.text(
                0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20
            )
            fig9.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
            fig9.savefig(save__orth + scan + f"/S{scan}_strain_from_dspacing.png")
            plt.close()
            fig_summary = summary_slice_plot_abd(
                title=f"Summary figure Pt 3 ID27 {scan}",
                support=zero_to_nan(bulk),
                dpi=150,
                voxel_size=voxel_size,
                isosurface=isosurface,
                det_reference_voxel=g_vector,
                averaged_dspacing=dspacing_mean, # type: ignore
                averaged_lattice_parameter=dspacing_mean, # type: ignore
                **final_plots,
            )
            # plt.tight_layout()
            fig_summary.savefig(save__orth + scan + f"/S{scan}_Summary_slice_plot.png")
            plt.close()

        np.savez(
            save__orth + scan + f"/S" + str(scan) + "s_results.npz",
            **{
                "amp": density_ortho,
                "phase": phase_ortho,
                "bulk": bulk,
                "voxel_size": voxel_size,
            },
        )
        gu.save_to_vti(
            filename=save__orth + scan + f"/S" + str(scan) + "_amp-phase_bulk.vti",
            voxel_size=[i for i in voxel_size],
            tuple_array=(density_ortho, amp, phase_ortho, bulk, displacement, strain),
            tuple_fieldnames=(
                "density",
                f"Amp*{int(density_ortho.max())}",
                "phase",
                "bulk",
                "displacement",
                "strain",
            ),
            amplitude_threshold=0.01,
        )
        gu.save_to_vti(
            filename=save__orth + scan + f"/S" + str(scan) + "_ortho_int.vti",
            voxel_size=[delta_dx_, delta_dy_, delta_dz_],
            tuple_array=(ndata),
            tuple_fieldnames=("int"),
            amplitude_threshold=0.01,
        )

        voxel_size_allscan[ii__] = voxel_size
        data_allscan_Int[ii__] = pad_to_shape(ndata, (100, 100, 100), pad_value=0)
        data_allscan_complexe[ii__] = pad_to_shape(
            (density_ortho * np.exp(1j * a__)) * bulk, (60, 60, 60), pad_value=0 * 1j # type: ignore
        )
        data_allscan_displacement[ii__] = pad_to_shape(
            nan_to_zero(displacement * bulk), (60, 60, 60), pad_value=0
        )
        data_allscan_a_spacing[ii__] = pad_to_shape(
            nan_to_zero(lattice_parameter * bulk), (60, 60, 60), pad_value=0
        )
        data_allscan_strain_from_dspacing[ii__] = pad_to_shape(
            nan_to_zero(strain * bulk), (60, 60, 60), pad_value=0
        )

        getting_strain_mapvti(
            obj=data_allscan_complexe[ii__],
            path_to_save=save__orth + i_scan + f"/S{i_scan}_",
            save_filename_vti="gradient_map.vti",
            voxel_size=voxel_size,
            plot_debug=True,
        )
        if linecut_plotting:
            output = linecut.fit_linecut(
                array=density_ortho,
                fit_derivative=True,
                filename=save__orth + scan + f"/S{scan}_LINE_CUT.png",
                voxel_sizes=list(voxel_size),
                label="modulus",
                support_threshold=0.2,
            )
            plt.show()
            maskkk = np.where(np.array(output["dimension_0"]["linecut"][1]) > 0.015)[0]
            maskkk1 = np.where(np.array(output["dimension_1"]["linecut"][1]) > 0.015)[0]
            maskkk2 = np.where(np.array(output["dimension_2"]["linecut"][1]) > 0.015)[0]
            plt.figure()
            plt.plot(
                np.array(output["dimension_0"]["linecut"][0])[maskkk],
                np.array(output["dimension_0"]["linecut"][1])[maskkk],
            )
            plt.plot(
                np.array(output["dimension_1"]["linecut"][0])[maskkk1],
                np.array(output["dimension_1"]["linecut"][1])[maskkk1],
            )
            plt.plot(
                np.array(output["dimension_2"]["linecut"][0])[maskkk2],
                np.array(output["dimension_2"]["linecut"][1])[maskkk2],
            )
            dimension = np.array(
                [
                    np.array(output["dimension_0"]["linecut"][0])[maskkk].max()
                    - np.array(output["dimension_0"]["linecut"][0])[maskkk].min(),
                    np.array(output["dimension_1"]["linecut"][0])[maskkk1].max()
                    - np.array(output["dimension_1"]["linecut"][0])[maskkk1].min(),
                    np.array(output["dimension_2"]["linecut"][0])[maskkk2].max()
                    - np.array(output["dimension_2"]["linecut"][0])[maskkk2].min(),
                ]
            )
            plt.title(
                f" S{scan} Line cut: dimention of particle {np.rint(dimension*voxel_size)} nm"
            )
            plt.savefig(save__orth + scan + f"/S{scan}dimension_Linecutbased.png")
            plt.show()
        end = time.time() - start
        print("**********" + "scan time: " + str(int(end)) + "s **********")

        ii__ += 1

    final_data = {
        "dat_cmp": data_allscan_complexe,
        "disp": data_allscan_displacement,
        "a_spacing": data_allscan_a_spacing,
        "het_strain": data_allscan_strain_from_dspacing,
        "int": data_allscan_Int,
        "voxel_size": voxel_size, # type: ignore
    }
    r = [x, y, z] # type: ignore
    starting_pixel_val_nm = [
        r[0][0] - 100 * voxel_size[i] for i in range(len(voxel_size)) # type: ignore
    ]

    np.savez(
        save__orth + "real_space_data",
        **{
            "data_cm": data_allscan_complexe,
            "displacement": data_allscan_displacement,
            "a_spacing": data_allscan_a_spacing,
            "strain_from_dspacing": data_allscan_strain_from_dspacing,
            "scan": scan_nb,
            "start_": starting_pixel_val_nm,
            "voxel_size": voxel_size_allscan,
        },
    )
    np.savez(
        save__orth + "int_data",
        **{
            "data_cm": data_allscan_Int,
            "scan": scan_nb,
            "qx": qx_lin, # type: ignore
            "qy": qy_lin, # type: ignore
            "qz": qz_lin, # type: ignore
        },
    )
    return final_data
def orth_SIXS2019(ndata):
    import numpy as np
    import xrayutilities as xu
    import scipy.ndimage as nd
    cx, cy, cz = np.round(np.array(nd.center_of_mass(ndata))).astype("int")
    print(cx, cy, cz)
    ##### orthonormalisation d'une solution
    # vecteur du réseau réciproque
    lamb = 12.398 / 33.169
    dd = 2.24e6
    taypix = 55
    pixang = taypix / dd
    a0 = 3.924
    h, k, l = 1, 1, 1
    d = a0 / np.sqrt(h * h + k * k + l * l)
    theta = np.arcsin(lamb / (2 * d))
    omgrange = 1.2
    DQx = (ndata.shape[0] - 1) * np.deg2rad(omgrange) / (ndata.shape[0] - 1)
    resx = 1 / DQx
    print(resx)
    DQy = (ndata.shape[1] - 1) * pixang / lamb
    resy = 1 / DQy
    print(resy)
    DQz = (ndata.shape[2] - 1) * pixang / lamb
    resz = 1 / DQz
    print(resz)
    vectQX = np.array([DQx, 0, 0])
    vectQY = np.array([0, DQy, 0])
    vectQZ = np.array([-DQz * np.sin(theta), 0, DQz * np.cos(theta)])

    X, Y, Z = np.indices(np.shape(ndata))
    cx = X * vectQX[0] + Y * vectQY[0] + Z * vectQZ[0]
    cy = X * vectQX[1] + Y * vectQY[1] + Z * vectQZ[1]
    cz = X * vectQX[2] + Y * vectQY[2] + Z * vectQZ[2]
    nx, ny, nz = np.shape(ndata)
    nz = nz + 10
    gridder = xu.FuzzyGridder3D(nx, ny, nz)
    gridder(cx, cy, cz, ndata)
    qx_lin = gridder.xaxis
    qy_lin = gridder.yaxis
    qz_lin = gridder.zaxis
    int_ortho = gridder.data
    # %% Normalisation vecteurs du reseau reciproque
    delta_x = 2 * np.pi / (qx_lin[-1] - qx_lin[0])
    delta_y = 2 * np.pi / (qy_lin[-1] - qy_lin[0])
    delta_z = 2 * np.pi / (qz_lin[-1] - qz_lin[0])
    print("Voxel Size :", delta_x, delta_y, delta_z)
    # %% Normalisation vecteurs du reseau reciproque
    delta_dx_ = (2 * np.pi * (1 / X)) / len(qx_lin)
    delta_dy_ = (2 * np.pi * (1 / Y)) / len(qy_lin)
    delta_dz_ = (2 * np.pi * (1 / Z)) / len(qz_lin)
    print("Reciprocal space voxel size :", delta_dx_, delta_dy_, delta_dz_)

    dqX = 2 * np.pi / (qx_lin[10] - qx_lin[9])
    dqY = 2 * np.pi / (qy_lin[20] - qy_lin[19])
    dqZ = 2 * np.pi / (qz_lin[30] - qz_lin[29])
    print("Inverse Step Size :", dqX, dqY, dqZ)

    return int_ortho, delta_dx_, delta_dy_, delta_dz_
def detectframe_to_labframe(data, mask, delta, gamma, mu, shape_after_ref):
    import numpy as np
    import xrayutilities as xu  
    
    #  experimental parameters
    degree = np.pi / 180.0
    beta = 1.7 * degree
    dd = 1.215e6  # distance detecteur-echantillon en micron
    tp = 55  # taille pixel detecteur en micron
    cch = [193, 201]  # position du faisceau direct pour delta et gamma nuls
    nrj = 8.5  # energie du faisceau de rayons X
    lamb = 12.398 / nrj  # longueur d'onde correspondante en angstrom

    dim = np.shape(data)
    ############ application du masque

    ndata = np.zeros(dim)
    ndata = data * (1 - mask)

    X, Y = np.indices((dim[1], dim[2]))

    TM = np.matrix.transpose(
        np.array(
            [
                np.array(
                    [
                        np.sin(delta) * np.cos(gamma),
                        np.sin(delta) * np.sin(gamma),
                        -np.cos(delta),
                    ]
                ),
                np.array([-np.sin(gamma * degree), np.cos(gamma * degree), 0]),
                dd
                * np.array(
                    [
                        np.cos(delta) * np.cos(gamma),
                        np.cos(delta) * np.sin(gamma),
                        np.sin(delta),
                    ]
                ),
            ]
        ) # type: ignore
    )

    cx = tp * (X - cch[0]) * TM[0, 0] + tp * (Y - cch[1]) * TM[0, 1] + TM[0, 2]
    cy = tp * (X - cch[0]) * TM[1, 0] + tp * (Y - cch[1]) * TM[1, 1] + TM[1, 2]
    cz = tp * (X - cch[0]) * TM[2, 0] + tp * (Y - cch[1]) * TM[2, 1] + TM[2, 2]
    r = np.sqrt(cx**2 + cy**2 + cz**2)
    qx = (2 * np.pi / lamb) * (cx / r - np.cos(beta))
    qy = (2 * np.pi / lamb) * (cy / r)
    qz = (2 * np.pi / lamb) * (cz / r + np.sin(beta))
    # %% in the sample frame
    qxs = np.zeros(dim)
    qys = np.zeros(dim)
    qzs = np.zeros(dim)

    qxs = np.array([qx * np.cos(ang) + qy * np.sin(ang) for ang in mu])
    qys = np.array([-qx * np.sin(ang) + qy * np.cos(ang) for ang in mu])
    qzs = np.array([qz for ang in mu])

    # %%
    nx, ny, nz = shape_after_ref

    gridder = xu.Gridder3D(nx, ny, nz)
    gridder(qxs, qys, qzs, ndata)

    qx_lin = gridder.xaxis
    qy_lin = gridder.yaxis
    qz_lin = gridder.zaxis
    int_ortho = gridder.data

    # %% Normalisation vecteurs du reseau reciproque
    X = 2 * np.pi / (qx_lin[-1] - qx_lin[0])
    Y = 2 * np.pi / (qy_lin[-1] - qy_lin[0])
    Z = 2 * np.pi / (qz_lin[-1] - qz_lin[0])

    print("Voxel Size :", X, Y, Z)

    dqX = 2 * np.pi / (qx_lin[10] - qx_lin[9])
    dqY = 2 * np.pi / (qy_lin[20] - qy_lin[19])
    dqZ = 2 * np.pi / (qz_lin[30] - qz_lin[29])
    print("Inverse Step Size :", dqX, dqY, dqZ)

    return qxs, qys, qzs, qx_lin, qy_lin, qz_lin, int_ortho
