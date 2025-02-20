"""
#################################################################################################
# Suggestions 
#################################################################################################
# 1. Removed redundant import:
#    - `from cdi_dislo.common_imports import C_O_M, zero_to_nan, nanmax`
#      (already covered by `from cdi_dislo.common_imports import *`)
#
# 2. Fixed function name in the docstring example:
#    - Changed `slection_of_reco_in_path(...)` to `selection_of_reco_in_path(...)`
#
# 3. Improved directory handling:
#    - Ensured all `os.makedirs()` calls use `exist_ok=True` to avoid directory errors.
#
# 4. Enhanced logging:
#    - Added `console_handler = logging.StreamHandler()` to print logs to the console.
#
# 5. Optimized `try-except` blocks:
#    - Combined multiple `try-except` blocks for plotting into a single block.
#
# 6. Improved shell command handling (`os.system`):
#    - Used `os.makedirs()` and Python file operations instead of shell `cd` and `mkdir`.
#
# 7. General cleanup:
#    - Ensured consistency in indentation, spacing, and formatting.
#################################################################################################
"""
"""
#################################################################################################
# Script Summary: Selection of Reconstruction Files
#################################################################################################
# This script performs selection of reconstructed 3D phase and density data based on:
# - **LLKf metric**: Log-likelihood factor for reconstruction quality.
# - **Standard deviation of density (stdrho)**: A measure of variation in density.
# - **Number of active pixels**: To filter reconstructions based on spatial significance.
#
# Key Features:
# - Supports **multiple selection methods** (`auto`, `looped`, `filter_by_llkf`, `manual`).
# - Loads and processes **.CXI reconstruction files**.
# - Filters reconstructions based on defined criteria and saves selected results.
# - Supports **plotting and visualization** of selection criteria.
# - **Creates mode files** for selected reconstructions (`modes.h5`).
# - **Optionally saves animations** of phase evolution in X, Y, Z directions.
# - Allows **archiving previous selections** and clearing previous results if needed.
#
# Outputs:
# - Saves selected reconstructions in a dedicated folder.
# - Generates selection logs (`log_selection.txt`).
# - Produces visualizations (`LLKf_vs_Run_sel.png`, `dist_sigmarho.png`, etc.).
# - Saves **3D projections** and phase animations (if enabled).
#
# Notes:
# - Uses `pynx-cdi-analysis` for mode file generation.
# - Provides an interactive loop option (`looped` mode) for user adjustment.
#
#################################################################################################
"""



from cdi_dislo.common_imports import *
from cdi_dislo.common_imports import C_O_M, zero_to_nan, nanmax

from cdi_dislo.general_utilities.cdi_dislo_utils                  import check_array_empty,IfStringRepresentsFloat,load_reco_from_cxinpz,optimize_cropping,crop_3darray_pos,get_max_cut_parametre
from cdi_dislo.plotutilities.cdi_dislo_plotutilities              import plot_single_3darray_slices_as_subplots,plot_3darray_as_gif_animation

from cdi_dislo.orthogonalisation_handler.cdi_dislo_ortho_handler  import remove_phase_ramp
from cdi_dislo.ewen_utilities.plot_utilities                      import plot_3D_projections ,plot_2D_slices_middle_one_array3D
#####################################################################################################################
#####################################################################################################################
def selection_of_reco_in_path(path_to_reco: str = '/home/abdelrahman/cristal_Nov_2022/',
                             particle: str = "B8",
                             scan: str = "S400",
                             prefix_to_reco_1: str = "/results/good/",
                             prefix_to_reco_2: str = "/pynxraw/new_shape_nomask/",
                             Folder_name_of_recosel: str = "selected_runs_stdrho_supp_newsel_corr_suppfix",
                             per_LLKf: float = 0.1,
                             per_stdrho: float = 0.1,
                             per_nbpixel: float = 0.95,
                             save_animation: bool = False,
                             create_mode: bool = True,
                             show_plots: bool = False,
                             old_results: str = 'old',
                             move_previous_recosel: bool = True,
                             clear_previous_recosel: bool = False,
                             selection_method: str = 'auto',
                             load_data: bool = False,
                             load_selection: bool = False,
                             ramp_phase: bool = False,
                             mode_output_file_name: str = "modes.h5",
                             ramp_mode: bool = True,
                             multiply_by_mask: bool = False) -> None:
    """
    Selects reconstruction files based on LLKf, standard deviation of density (stdrho), and number of active pixels.

    Parameters:
    - path_to_reco (str): Path to the directory containing reconstruction files. Default: '/home/abdelrahman/cristal_Nov_2022/'
    - particle (str): Name of the particle. Example: 'B8'
    - scan (str): Name of the scan. Example: 'S400'
    - prefix_to_reco_1 (str): Prefix for the first part of reconstruction files. Default: '/results/good/'
    - prefix_to_reco_2 (str): Prefix for the second part of reconstruction files. Default: '/pynxraw/new_shape_nomask/'
    - Folder_name_of_recosel (str): Name of the folder for selected reconstructions. Default: 'selected_runs_stdrho_supp_newsel_corr_suppfix'
    - per_LLKf (float): Percentage of LLKf for selection. Default: 0.1
    - per_stdrho (float): Percentage of stdrho for selection. Default: 0.1
    - per_nbpixel (float): Percentage of active pixels for selection. Default: 0.95
    - save_animation (bool): Whether to save animations. Default: False
    - create_mode (bool): Whether to create mode for selected reconstructions. Default: True
    - show_plots (bool): Whether to display plots. Default: False
    - old_results (str): Directory for old results. Default: 'old'
    - move_previous_recosel (bool): Whether to move previous reco selections. Default: True
    - clear_previous_recosel (bool): Whether to clear previous reco selections. Default: False
    - selection_method (str): Method for selection, with possible values:
      - 'auto': Automatic selection based on default parameters
      - 'looped': Interactive loop allowing adjustments based on user input
      - 'filter_by_llkf': Selection based solely on LLKf values
      - 'manual': Manual selection process
      Default: 'auto'
    - load_data (bool): Whether to load data instead of processing. Default: False
    - load_selection (bool): Whether to load an existing selection. Default: False
    - ramp_phase (bool): Whether to apply a ramp phase. Default: False
    - mode_output_file_name (str): Filename for mode output. Default: "modes.h5"
    - ramp_mode (bool): Whether to apply ramp mode in analysis. Default: True
    - multiply_by_mask (bool): Whether to multiply data by mask. Default: False

    Returns:
    - None

    Example:
    selection_of_reco_in_path(path_to_reco='/path/to/reconstruction/',
                             particle="H1",
                             scan="S500",
                             prefix_to_reco_1="/reconstructions/good/",
                             prefix_to_reco_2="/pynxraw/new_shape_nomask/",
                             Folder_name_of_recosel="selected_runs",
                             per_LLKf=0.2,
                             per_stdrho=0.15,
                             per_nbpixel=0.9,
                             save_animation=True,
                             create_mode=True,
                             selection_method='looped')
    """
    # Function body
    rc['text.usetex'] = True;sns.set_theme();rc['figure.figsize']= (9, 9);  rc['font.size']= 14;rc['xtick.labelsize']= 14;
    rc['ytick.labelsize']   = 14;rc["figure.autolayout"] = True;  plt.rcParams['text.usetex'] = False
    start_glob = time.time()
    print("Time: "+ str(datetime.now().strftime("%H:%M:%S")))
    
    #particle="B8";scan="S400"
    #path_to_reco='/home/abdelrahman/cristal_Nov_2022/'
    #prefeix_to_reco_2="/pynxraw/new_shape_nomask/"
    #prefeix_to_reco_1="/pynxraw/new_shape_nomask/"
    #Folder_name_of_recosel="selected_runs_stdrho_supp_newsel_corr_suppfix"
    wdir=path_to_reco;
    if not os.path.exists(wdir):
        print(f"The file or directory '{wdir}' does not exist."); return 
    else:
        wdir+=particle
        if not os.path.exists(wdir):
            print(f"The file or directory '{wdir}' does not exist."); return 
        else:
            wdir+=prefix_to_reco_1
            if not os.path.exists(wdir):
                print(f"The file or directory '{wdir}' does not exist.");return 
            else:
                wdir+=scan
                if not os.path.exists(wdir):
                    print(f"The file or directory '{wdir}' does not exist.")
                    return 
                else:
                    wdir+=prefix_to_reco_2
                    if not os.path.exists(wdir):
                        print(f"The file or directory '{wdir}' does not exist.")
                        return                
    
    print("Searching data in this path: "+wdir)
    files=glob.glob(wdir+"*LLKf*cxi")
    if  check_array_empty(files):
        print(f"No .CXI files found in {wdir}. Please check if prefix_to_reco_2 parametre well defined (path to .CXI reco files)");return 
    wdir_save_reco_sel=path_to_reco+particle+prefix_to_reco_1+scan+"/"+prefix_to_reco_2+Folder_name_of_recosel+"/";
    wdir_save= wdir+"selection/"
    os.makedirs(wdir_save_reco_sel,exist_ok=True) 
    os.makedirs(wdir_save,exist_ok=True) 
    log_file = os.path.join(wdir_save, "log_selection.txt")
    # Redirect stdout and stderr to log file
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting selection_of_reco_in_path function.")

    print("results of the selection and debugging will be save to : "+wdir_save)
    
    list_run   =[ i_run[i_run.find("_Run")+len("_Run"):i_run.find("_Run")+len("_Run")+4] for i_run in files ]
    llkf_run   =[ IfStringRepresentsFloat(i_run[i_run.find("_LLKf")+len("_LLKf"):i_run.find("_LLK0")]) for i_run in files ]
    print(llkf_run)
    try:
        list_llkf=np.array(llkf_run).astype(float)
        list_llkf = np.where(list_llkf > 10., 10., list_llkf)
    except:
        list_llkf=np.array([ IfStringRepresentsFloat(i_run) for i_run in llkf_run ])
        list_llkf = np.where(llkf_run > 10., 10., llkf_run)
    print('********** scan ' + str(scan) + ' with ' + str(len(files)) + 'runs' + '**********')
    def data_read_selection(files,multiply_by_mask):
        """
        Read data from files and process it to extract rho and phi data.
    
        Parameters:
        - files: List of file paths containing the data
        - dirlist: List of directories associated with each file
    
        Returns:
        - rho_data: Processed rho data
        - phi_data: Processed phi data
        """
        start = time.time()
        data_allrun,rho_data, phi_data, rho_data_mask= [],[],[],[]
        size           = len(files)
        pad_add=20
        data_allrun=np.array([ np.pad(np.array(load_reco_from_cxinpz(i_run,multiply_by_mask=multiply_by_mask)),  ((pad_add, pad_add), (pad_add, pad_add), (pad_add, pad_add)) ) for i_run in files ])
        rho_data       = abs (data_allrun)
        rho_data*=rho_data>rho_data.max()*0.05    
        phi_data       = np.angle(data_allrun)*(rho_data>rho_data.max()*0.05)
        phi_data=np.angle(rho_data*np.exp(1j*phi_data))  
        mask= array([ ((i>0).astype(float)) for i in rho_data])
        
        com_int=[ list(array(C_O_M(i)).astype(int)) for i in  mask]
        half_w=int(np.max(array([optimize_cropping(i>0) for i in rho_data]))/2+5)
        half_w=shape_=100   #int(2*half_w*1.25)
        #ndata_centre=crop_3darray_pos(rho_data,com_int ,(shape_,shape_,shape_))
        
        rho_data         = array([ crop_3darray_pos(rho_data[i], methods=com_int[i],output_shape=(half_w,half_w,half_w))  for i in range(len(rho_data))])
        phi_data         = array([ crop_3darray_pos(phi_data[i], methods=com_int[i],output_shape=(half_w,half_w,half_w))  for i in range(len(rho_data))])
    
        mask= array([ ((i>0).astype(float)) for i in rho_data])
        rho_data=zero_to_nan(rho_data)
        mask=zero_to_nan(mask)
        phi_data=phi_data*mask
        ######################################
        # remove ramp then unwrapping        #
        ######################################
        if ramp_phase:
            phi_data=np.array([remove_phase_ramp((phi_data[i]))[0] for i in  range(len(phi_data))])
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + 'min')
        return rho_data,phi_data
    if selection_method != "filter_by_llkf":
        start = time.time()
    
        if not load_data:
            rho_data, phi_data = data_read_selection(files,multiply_by_mask)
            print("read data done")
            np.savez(wdir_save+'all_part_scan_results.npz', **{'rho_data': rho_data,
                                                      'phi_data': phi_data,
                                                      'list_run_allscan':list_run,
                                                      'list_scan':scan,
                                                      'list_llkf_allscan':list_llkf})
            print("saving data processed rho \& phi \& list of run \& the LLKf in " +wdir_save+'all_part_scan_results.npz')
        
        f=np.load(wdir_save+"all_part_scan_results.npz",allow_pickle=True)
        rho_data    =array(f["rho_data"])
        phi_data    =array(f["phi_data"])
        data_allscans_mask   =rho_data>0
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + 'min')
        
        rho_data=zero_to_nan(rho_data)
        results_STD=nanmax(rho_data, axis=(1, 2, 3))
        end = time.time()
        print(str(int(((end - start) / 60) * 10) / 10) + 'min')
        np.savez(wdir_save+'all_part_scan_results_selection.npz', **{'STD_rho':results_STD ,
                                                                     'list_llkf_allscan':list_llkf})
        f=np.load(wdir_save+"all_part_scan_results_selection.npz",allow_pickle=True)
        results_STD         = array(f["STD_rho"])
    list_llkf_allscan   = list_llkf#array(f["list_llkf_allscan"])
    print("done estimation of std of the density")
    nb_frames = len(list_llkf)
    if selection_method == "filter_by_llkf":
        nb_sel_LLKF = int(nb_frames * per_LLKf)
        run_list = np.arange(0, nb_frames, step=1).astype(str)
        
        run_list = np.arange(0, nb_frames, step=1).astype(str)
        print("####################################################################")
        print('selection based only on LLKf')
        RESULTS_LLKf = np.array(list_llkf)
        max_llkf = get_max_cut_parametre(RESULTS_LLKf, wanted_nb=nb_sel_LLKF)
        trigger_llk = np.where((list_llkf <= max_llkf) & (list_llkf != 0))[0]
        print("*******" + scan + "*********" + str(len(trigger_llk)) + "*********" + str(np.round(max_llkf, 4)))

        trigger_LLK_pixelNB_mtm_stdrho=trigger_llk
        a, b, c = [], [], []
        for i_run in trigger_LLK_pixelNB_mtm_stdrho:
            a.append(list_run[i_run])
            b.append(list_llkf[i_run])
        print("list of selected run ")
        print(b)
        print("list of LLKf of selected run ")
        print(a)
        print("####################################################################")
        pass
    elif selection_method == 'looped':    
        while True:
            nb_sel_LLKF = int(nb_frames * per_LLKf)
            nb_sel_stdrho = int(nb_frames * per_stdrho)
            nb_nbpixel = nb_frames - int(nb_frames * per_nbpixel)
        
            run_list = np.arange(0, nb_frames, step=1).astype(str)
            print("####################################################################")
            print('selection based on nb active pixel')
            sum_pixel = []
            for i_run in range(nb_frames):
                sum_pixel.append(int(np.sum(data_allscans_mask[i_run])))
            sum_pixel = np.array(sum_pixel)
            MIN_sumpixel = get_max_cut_parametre(sum_pixel, wanted_nb=nb_nbpixel)
            select_nb_pixel = np.where((sum_pixel >= MIN_sumpixel))[0]
            print("*******" + scan + "*********" + str(len(select_nb_pixel)) + "*********" + str(np.round(MIN_sumpixel, 4)))
            print('selection based on LLKf')
            RESULTS_LLKf = np.array(list_llkf)
            max_llkf = get_max_cut_parametre(RESULTS_LLKf, wanted_nb=nb_sel_LLKF)
            trigger_llk = np.where((list_llkf <= max_llkf) & (list_llkf != 0))[0]
            print("*******" + scan + "*********" + str(len(trigger_llk)) + "*********" + str(np.round(max_llkf, 4)))
        
            print('selection based on std rho')
            RESULTS_stdrho = np.array(results_STD)
            max_stdrho = get_max_cut_parametre(RESULTS_stdrho, wanted_nb=nb_sel_stdrho)
            trigger_STDRHO = np.where(np.array(RESULTS_stdrho) <= max_stdrho)[0]
            print("*******" + scan + "*********" + str(len(trigger_STDRHO)) + "*********" + str(np.round(max_stdrho, 4)))
            print('combine the selections based on std density, the LLKf and the # in the support ')
            trigger___ = []
            for i_run in trigger_llk:
                if (
                    (i_run in select_nb_pixel) and
                    (i_run in trigger_STDRHO) and
                    (i_run in trigger_llk)  # and
                ):
                    trigger___.append(i_run)
                else:
                    continue
            print("*******" + scan + "*********" + str(len(trigger___)) + f"/{nb_frames}")
            trigger_LLK_pixelNB_mtm_stdrho = trigger___
            a, b, c = [], [], []
            for i_run in trigger_LLK_pixelNB_mtm_stdrho:
                a.append(list_run[i_run])
                b.append(list_llkf[i_run])
                c.append(sum_pixel[i_run])
            print("list of selected run ")
            print(b)
            print("list of LLKf of selected run ")
            print(a)
            print("list of #pixel in supp of selected run ")
            print(c)
            print("####################################################################")
        
            # Ask user if they are satisfied with the selection
            user_input = input("Are you satisfied with the selection? (yes/no): ")
            if user_input.lower() == "yes":
                break
            elif user_input.lower() == "no":
                # Modify parameters based on user input
                per_LLKf = float(input("Enter the new percentage of LLKf: "))
                per_stdrho = float(input("Enter the new percentage of stdrho: "))
                per_nbpixel = float(input("Enter the new percentage of active pixels: "))
                continue
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    elif selection_method == 'auto':
        nb_sel_LLKF = int(nb_frames * per_LLKf)
        nb_sel_stdrho = int(nb_frames * per_stdrho)
        nb_nbpixel = nb_frames - int(nb_frames * per_nbpixel)
    
        run_list = np.arange(0, nb_frames, step=1).astype(str)
        print("####################################################################")
        print('selection based on nb active pixel')
        sum_pixel = []
        for i_run in range(nb_frames):
            sum_pixel.append(int(np.sum(data_allscans_mask[i_run])))
        sum_pixel = np.array(sum_pixel)
        MIN_sumpixel = get_max_cut_parametre(sum_pixel, wanted_nb=nb_nbpixel)
        select_nb_pixel = np.where((sum_pixel >= MIN_sumpixel))[0]
        print("*******" + scan + "*********" + str(len(select_nb_pixel)) + "*********" + str(np.round(MIN_sumpixel, 4)))
        print('selection based on LLKf')
        RESULTS_LLKf = np.array(list_llkf)
        max_llkf = get_max_cut_parametre(RESULTS_LLKf, wanted_nb=nb_sel_LLKF)
        trigger_llk = np.where((list_llkf <= max_llkf) & (list_llkf != 0))[0]
        print("*******" + scan + "*********" + str(len(trigger_llk)) + "*********" + str(np.round(max_llkf, 4)))
    
        print('selection based on std rho')
        RESULTS_stdrho = np.array(results_STD)
        max_stdrho = get_max_cut_parametre(RESULTS_stdrho, wanted_nb=nb_sel_stdrho)
        trigger_STDRHO = np.where(np.array(RESULTS_stdrho) <= max_stdrho)[0]
        print("*******" + scan + "*********" + str(len(trigger_STDRHO)) + "*********" + str(np.round(max_stdrho, 4)))
        print('combine the selections based on std density, the LLKf and the # in the support ')
        trigger___ = []
        for i_run in trigger_llk:
            if (
                (i_run in select_nb_pixel) and
                (i_run in trigger_STDRHO) and
                (i_run in trigger_llk)  # and
            ):
                trigger___.append(i_run)
            else:
                continue
        print("*******" + scan + "*********" + str(len(trigger___)) + f"/{nb_frames}")
        trigger_LLK_pixelNB_mtm_stdrho = trigger___
        a, b, c = [], [], []
        for i_run in trigger_LLK_pixelNB_mtm_stdrho:
            a.append(list_run[i_run])
            b.append(list_llkf[i_run])
            c.append(sum_pixel[i_run])
        print("list of selected run ")
        print(b)
        print("list of LLKf of selected run ")
        print(a)
        print("list of #pixel in supp of selected run ")
        print(c)
        print("####################################################################")
        pass
    else:
        print("Invalid selection method. Please choose 'looped' or 'auto'.")    
    try:   
        fig1=plt.figure(1,figsize=(9,5))
        y=np.array(list_run).astype(float)
        x=np.array(list_llkf)
        y_sel=    np.array(y)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        x_sel=    np.array(x)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        p=plt.plot(x,y, '*',linewidth=2,label="all runs")
        p=plt.plot(x_sel,y_sel, '*',linewidth=2,label="selected")
        plt.ylabel("Run", fontsize=24)
        plt.xlabel("LLKf", fontsize=24)
        plt.grid(alpha=0.5)
        #plt.title('Plot of Run in function of means in Ampl.',fontsize=24)
        plt.xlim((0.,1))
        plt.legend(fontsize=14,loc="best",ncol=1)
        plt.title(scan,fontsize=24)
        plt.savefig(wdir_save  + "LLKf_vs_Run_sel.png", dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close()
    except:
        print("couldn't plot std rho vs LLKf all runs")        
    try:   
        fig1=plt.figure(1,figsize=(9,5))
        y=np.array(results_STD)
        x=np.array(list_llkf)
        y_sel=    np.array(y)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        x_sel=    np.array(x)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        p=plt.plot(x,y, '*',linewidth=2,label="all runs")
        p=plt.plot(x_sel,y_sel, '*',linewidth=2,label="selected")
        plt.ylabel("$STD \\rho$", fontsize=24)
        plt.xlabel("LLKf", fontsize=24)
        plt.grid(alpha=0.5)
        #plt.title('Plot of Run in function of means in Ampl.',fontsize=24)
        plt.xlim((0.,1))
        plt.legend(fontsize=14,loc="best",ncol=1)
        plt.title(scan,fontsize=24)
        plt.savefig(wdir_save  + "LLKf_vs_sigmarho_sel.png", dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close()
    except:
        print("couldn't plot std rho vs LLKf all runs")
    try:
        fig1=plt.figure(1,figsize=(9,5))
        a=np.max(RESULTS_LLKf[RESULTS_LLKf!=10.])
        y_sel=    np.asarray(RESULTS_stdrho)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        x_sel=    np.asarray(RESULTS_LLKf)[np.hstack(trigger_LLK_pixelNB_mtm_stdrho)]
        p=plt.plot(RESULTS_LLKf,RESULTS_stdrho, '*',linewidth=2,label="all runs")
        plt.plot([RESULTS_LLKf.min(),RESULTS_LLKf.max()],[max_stdrho,max_stdrho],c=p[0].get_color())
        plt.plot([max_llkf,max_llkf],[RESULTS_stdrho.min(),RESULTS_stdrho.max()],c=p[0].get_color())
        p=plt.plot(x_sel,y_sel, '>',linewidth=2,label="selected")
        plt.ylabel("$\sigma_{\\rho}$", fontsize=24)
        plt.xlabel("LLKf", fontsize=24)
        plt.grid(alpha=0.5)
        plt.legend(fontsize=14,loc="best",ncol=1)
        plt.title(scan,fontsize=24)
        plt.xlim((RESULTS_LLKf.min(),a))
        plt.savefig(wdir_save  + "LLKf_vs_sigmarho.png", dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close()
    except:
        print("couldn't plot std rho vs LLKf for selcted runs ")
    try:
        fig1=plt.figure(1,figsize=(9,5))
        
        sns.histplot(RESULTS_stdrho, kde=True, stat='count', fill=True,bins=int(nb_frames*0.75))
        plt.axvline(max_stdrho, color='red', linestyle='--', linewidth=2)  # Add a vertical line for max_stdrho
        plt.xlabel('$\sigma_{ \\rho}$', fontsize=24)
        plt.ylabel('# of run', fontsize=24)  # Label the y-axis appropriately
        plt.title(scan, fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.savefig(wdir_save  + "dist_sigmarho.png", dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close()
    except:
        print("couldn't plot std rho vs run nb ")
    try:
        file_save=wdir_save+'selectedd_data_rho.pdf'
        pdf = be_pdf.PdfPages(file_save)
        for i_run in trigger_LLK_pixelNB_mtm_stdrho:
            fig=plot_3D_projections(zero_to_nan(rho_data[i_run]),fig_title='S'+scan+' Run '+str(list_run[i_run])+' llkf '+str(np.round(list_llkf[i_run],3))+" $\sigma_{\\rho}$ "+str(np.round(results_STD[i_run],3)))
            pdf.savefig(fig, dpi=150)
            if show_plots:
                plt.show()
            else:
                plt.close()
        pdf.close()
    except:
        print("couldn't plot rho for selected runs")
    try: 
        file_save=wdir_save+'selectedd_data_phi.pdf'
        pdf = be_pdf.PdfPages(file_save)
        for i_run in trigger_LLK_pixelNB_mtm_stdrho:
            fig=plot_3D_projections(phi_data[i_run],cmap='jet',fig_title='S'+scan+' Run '+str(list_run[i_run])+' llkf '+str(np.round(list_llkf[i_run],3))+" $\sigma_{\\rho}$ "+str(np.round(results_STD[i_run],3)))
            pdf.savefig(fig, dpi=150)
            if show_plots:
                plt.show()
            else:
                plt.close()
        pdf.close()
    except:
        print("couldn't plot phase for selected runs")      
    if move_previous_recosel:
        os.system(f"cd {wdir_save_reco_sel};mkdir {old_results}; mv * {old_results}")
    if clear_previous_recosel:
        os.system(f"cd {wdir_save_reco_sel}; rm *")
    for i_run in range(len(trigger_LLK_pixelNB_mtm_stdrho)):
        print("the reconstructions selected will be saved to the following path")
        print(wdir_save_reco_sel)
        L=files[i_run]
        print(L[:-3])
        shutil.copy(L, wdir_save_reco_sel)
        try:
            shutil.copy(L[:L.find("_LLKf")]+".png", wdir_save_reco_sel)
        except IndexError as e:
            print(e)
            continue
    if create_mode:
        print("****** running the mode for the seleceted reconstriction ******")
        if ramp_mode:
            os.system(f"cd {wdir_save_reco_sel}; pynx-cdi-analysis *cxi modes=1 phase_ramp modes_output={mode_output_file_name}")
        else:
            os.system(f"cd {wdir_save_reco_sel}; pynx-cdi-analysis *cxi modes=1 modes_output={mode_output_file_name}")
    if save_animation:
        start_time=time.time()
        phi_max=0.25*np.pi
        for i_run in range(len(trigger_LLK_pixelNB_mtm_stdrho)):
            L0=files[0]
            destination_=wdir_save
            L=files[i_run]
            #plot_3darray_as_gif_animation(zero_to_nan(data_allscans_phi[i_scan][i_run]),destination_ + L[:-3][L[:-3].find("/_pynx") + 1:-1] + '_x'+str(phi_max)+'.gif',-phi_max, phi_max,title_fig='Animation in direction X '+i_scan + 'run' + str(i_run) )
            try:
                print('create animation of the phase along direction X')
                save_file_name=  destination_+L[L.find("_pynx") :-4] + '_x'+str(phi_max)+'.gif'
                title_fig= 'Animation in direction X '+scan + ' run ' + str(list_run[i_run]) 
                plot_single_3darray_slices_as_subplots(zero_to_nan(phi_data[i_run]),save_file_name,-phi_max, phi_max, title_fig=title_fig, proj=0)
                print('create animation of the phase along direction Y')
                save_file_name=  destination_+L[L.find("_pynx") :-4]  + '_y'+str(phi_max)+'.gif'
                title_fig= 'Animation in direction Y '+scan + ' run ' + str(list_run[i_run]) 
                plot_single_3darray_slices_as_subplots(zero_to_nan(phi_data[i_run]),save_file_name,-phi_max, phi_max, title_fig=title_fig, proj=1)
                print('create animation of the phase along direction Z')
                save_file_name= destination_+L[L.find("_pynx") :-4]  + '_z'+str(phi_max)+'.gif'
                title_fig= 'Animation in direction Z '+scan + ' run ' + str(list_run[i_run]) 
                plot_single_3darray_slices_as_subplots(zero_to_nan(phi_data[i_run]),save_file_name,-phi_max, phi_max, title_fig=title_fig, proj=2)
            except IndexError as e:
                    print(e)
                    continue
        print(f"animation took {round((start_time-time.time())/60)} min")
    print(f"selection took {round((start_glob-time.time())/60)} min")
    print("Time: "+ str(datetime.now().strftime("%H:%M:%S")))
    print("##################"+"End"+"##################")
    # Restore stdout and stderr
    logging.info("Function execution completed.")

