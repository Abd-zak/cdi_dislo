import numpy as np
import pylab as plt
from scipy.optimize import curve_fit


###########################################################################################################################################
############################################          Homemade Gaussian fit functions           ###########################################
###########################################################################################################################################


def calculate_center_of_mass(array1d, x=None):
    if x is None:
        x = np.arange(len(array1d))

    proba = array1d / np.sum(array1d)
    return np.sum(proba * x)


def calculate_std(array1d, x=None):
    if x is None:
        x = np.arange(len(array1d))

    proba = array1d / np.sum(array1d)

    x_cen = calculate_center_of_mass(array1d, x=x)

    return np.sqrt(np.sum(proba * ((x - x_cen) ** 2.0)))


def GaussianAndLinearBackground(
    x, A, x0, sig, background_slope, background_const
):
    return (
        A * np.exp(-0.5 * (x - x0) ** 2.0 / sig**2.0)
        + background_slope * x
        + background_const
    )


def Gaussian(x, A, x0, sig):
    return A * np.exp(-0.5 * (x - x0) ** 2.0 / sig**2.0)


def GaussianAndLinearBackgroundFit(
    array1d,
    x=None,
    sig=None,
    return_popt_pcov=False,
    background=True,
    plot=False,
):
    if x is None:
        x = np.arange(len(array1d))

    #     x0 = x[np.argmax(array1d)]
    x0 = calculate_center_of_mass(array1d, x=x)

    if sig is None:
        sig = calculate_std(array1d, x=x)

    if background:
        N_background = 5
        background_const = np.nanmean(
            array1d[np.argsort(array1d)[:N_background]]
        )
        background_slope = 0.0
        A = np.max(array1d) - background_const
    else:
        A = np.max(array1d)

    #     popt, pcov = curve_fit(GaussianAndLinearBackground, x, array1d, p0=[A,x0,sig,background_slope, background_const],
    # #                           bounds = ((-np.inf, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
    #                           bounds = ((-np.inf, -np.inf, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf)))
    if background:
        popt, pcov = curve_fit(
            GaussianAndLinearBackground,
            x,
            array1d,
            p0=[A, x0, sig, background_slope, background_const],
            #                           bounds = ((-np.inf, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
            bounds=(
                (-np.inf, np.min(x), 0, -np.inf, -np.inf),
                (np.inf, np.max(x), np.inf, np.inf, np.inf),
            ),
        )
    else:
        popt, pcov = curve_fit(
            Gaussian,
            x,
            array1d,
            p0=[A, x0, sig],
            #                           bounds = ((-np.inf, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
            bounds=((-np.inf, np.min(x), 0), (np.inf, np.max(x), np.inf)),
        )

    if background:
        fit = GaussianAndLinearBackground(x, *popt)
    else:
        fit = Gaussian(x, *popt)

    if plot:
        plt.figure()
        plt.plot(x, array1d, "b.-")
        if background:
            plt.plot(
                x,
                GaussianAndLinearBackground(
                    x, A, x0, sig, background_slope, background_const
                ),
                "g-",
                label="guess",
            )
        else:
            plt.plot(x, Gaussian(x, A, x0, sig), "g-", label="guess")

        plt.plot(x, fit, "r-", label="fit")
        plt.legend()

    if return_popt_pcov:
        return fit, popt, pcov
    else:
        return fit
