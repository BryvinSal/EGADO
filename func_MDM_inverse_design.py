""""""""""""""""""""""""""""""""""""""""""""""""""""
Inverse Design Project implementing multiple methods
Copyright (C)   - FiOWIN Lab @ Fudan University
Change Log      - 2022 v1.0: DBS
                - 2023 v2.0: TO
                - 2023 v3.0: ATD
                - 2023 v4.0: EG_ATD
                
Last Edition    - 2023.12.8
                            by Aolong Sun & Xuyu Deng
"""""""""""""""""""""""""""""""""""""""""""""""""""""

import sys
from splayout import *

sys.path.append('C:\\Downloaded App\\Lumerical 2023 R2.2\\api\\python')
import lumapi
import copy
from lumopt.geometries.topology import TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths
from func_file_operation import *
from func_get_conversion_matrix import *


class MDM_InverseDesign:
    def __init__(self, pitch, shape, rect_size, radius, z_min, z_max, design_region_x, design_region_y, input_wg_width,
                 output_wg_width, num_in_wg, num_out_wg, n_bg, n_wg, wavelength, wavelength_range=100e-9, buff=0,
                 target_fom=1, offset_x=0.0, offset_y=0.0, self_define_materials=True, in_mode_list=None, hide_cad=True,
                 out_mode_list=None, processor='CPU'):
        """
        :param pitch: pixel size
        :param shape: square or circle
        :param rect_size: square size
        :param radius: circle radius
        :param z_min: of waveguide core
        :param z_max: of waveguide core
        :param design_region_x: size of optimized region in x direction
        :param design_region_y: size of optimized region in y direction
        :param input_wg_width: width of input waveguides
        :param output_wg_width: width of output waveguides
        :param num_in_wg: number of input waveguides
        :param num_out_wg: number of output waveguides
        :param n_bg: background index (cladding and BOX)
        :param n_wg: waveguide index
        :param wavelength: center wavelength of simulations
        :param wavelength_range: wavelength range of simulations
        :param buff: used in EG_ATD, determines the kernel size of edge-detection, set to be 0, 2, 4 ...
        :param target_fom: targeted FoM value, only modifies the display without any impact on the optimization
        :param offset_x: offset of optimized region in the x direction
        :param offset_y: offset of optimized region in the y direction
        :param self_define_materials: Use the material index defined by User when set True
        :param in_mode_list: mode list of input sources
        :param out_mode_list: mode list of output monitors
        """

        self.target_fom = target_fom
        self.fdtd = None
        self.pitch = pitch
        self.shape = shape
        self.rect_size_x = rect_size
        self.rect_size_y = rect_size
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max
        self.design_region_x = design_region_x
        self.design_region_y = design_region_y
        self.num_in_wg = num_in_wg
        self.num_out_wg = num_out_wg
        self.output_wg_width = output_wg_width
        self.input_wg_width = input_wg_width
        self.wavelength = wavelength
        self.wavelength_range = wavelength_range
        self.buff = buff
        self.in_mode_list = in_mode_list
        self.out_mode_list = out_mode_list
        self.mode_num = len(self.in_mode_list)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.logger = setlogging()
        self.self_define_materials = self_define_materials
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.processor = processor
        self.hide_cad = hide_cad

        self.pixels_per_row = int(self.design_region_x / self.pitch)
        self.pixels_per_col = int(self.design_region_y / self.pitch)
        self.pixels_number = self.pixels_per_row * self.pixels_per_col

        if int(self.design_region_x / self.rect_size_x) != self.pixels_per_row or int(
                self.design_region_y / self.rect_size_y) != self.pixels_per_col:
            assert False, "Pixel per row or pixel per column is not an integer!"

        if len(self.in_mode_list) != len(self.out_mode_list):
            assert False, "Input mode numbers should equal to output mode numbers!"

        # determine the index values either by material names or by numbers
        if isinstance(n_bg, str) and isinstance(n_wg, str):
            fdtd = lumapi.FDTD()
            c = 299792458
            self.center_wavelength = self.wavelength
            self.n_bg = np.real(fdtd.getindex(n_bg, c / self.center_wavelength))[0, 0]
            self.n_wg = np.real(fdtd.getindex(n_wg, c / self.center_wavelength))[0, 0]
        elif isinstance(n_bg, float) and isinstance(n_wg, float):
            self.n_bg = n_bg
            self.n_wg = n_wg

    #########################################
    ######### Optimization Methods ##########
    #########################################
    ################## TOPOLOGY OPTIMIZATION USING ADJOINT METHOD ######################
    def TO(self, lsf_filename='base_TO.lsf', initial_cond=None, initial_type="uniform", filter_R=100e-9,
           min_feature_size=0, startingBeta=1, start_gradient=0.25, resume=False, epoch=0):
        """
        :param lsf_filename: base script file of .lsf
        :param initial_cond: initial index distribution
        :param initial_type: initial type
        :param filter_R: radius of filter
        :param min_feature_size: minimum feature size
        :param startingBeta: initial beta (determines the discreteness)
        :param start_gradient: enforces a rescaling of the gradient to change the optimization parameters by at least this much;
                                          the default value of zero disables automatic scaling.
        :param resume: set to True when you want to resume this optimization from last interruption
        :param epoch: epoch number
        :return: return current directory name
        """

        def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R,
                   min_feature_size, working_dir, beta, scale_initial_gradient_to, lsf_filename):

            ######## DEFINE A 3D TOPOLOGY OPTIMIZATION REGION ########
            geometry = TopologyOptimization3DLayered(
                params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos,
                filter_R=filter_R, min_feature_size=min_feature_size, beta=beta)

            ######## DEFINE FIGURE OF MERIT FOR EACH OUTPUT WAVEGUIDE ########
            foms = [ModeMatch(monitor_name=f'fom{i + 1}', mode_number=self.out_mode_list[i], direction='Forward',
                              target_T_fwd=lambda wl: 1 * np.ones(wl.size), norm_p=2, target_fom=self.target_fom) for i
                    in
                    range(self.mode_num)]

            ######## DEFINE OPTIMIZATION ALGORITHM ########
            optimizer = ScipyOptimizers(max_iter=400, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4,
                                        scale_initial_gradient_to=scale_initial_gradient_to)

            ######## DEFINE SETUP SCRIPT AND INDIVIDUAL OPTIMIZERS ########
            base_script = self.config_for_lsf(dir_name=self.cur_path, lsf_filename=lsf_filename)

            wavelength = Wavelengths(start=self.wavelength - 0.5 * self.wavelength_range,
                                     stop=self.wavelength + 0.5 * self.wavelength_range, points=11)

            # source name is indexed by mode numbers used in this optimization rather than the actual mode names
            opts = [Optimization(base_script=base_script, wavelengths=wavelength, fom=foms[i],
                                 label=f"TE{self.in_mode_list[i] - 1}", source_name=f"Source_TE_mode{i + 1}_1550",
                                 geometry=geometry, optimizer=optimizer, use_deps=False,
                                 hide_fdtd_cad=self.hide_cad, plot_history=False, store_all_simulations=False,
                                 save_global_index=True, fields_on_cad_only=False) for i in range(self.mode_num)]

            ######## PUT EVERYTHING TOGETHER AND RUN ########
            opt = opts[0]
            for i in range(1, self.mode_num):
                opt += opts[i]
            opt.continuation_max_iter = 40

            ######## RUN THE OPTIMIZER ########
            opt.run(working_dir=working_dir)

        os.makedirs("TO", exist_ok=True)

        size_x = int(self.design_region_x * 1e9)
        size_y = int(self.design_region_y * 1e9)
        size_z = int((self.z_max - self.z_min) * 1e9)

        eps_wg = self.n_wg ** 2
        eps_bg = self.n_bg ** 2

        x_points = int(size_x / 20) + 1
        y_points = int(size_y / 20) + 1
        z_points = int(size_z / 110) + 1

        x_pos = np.linspace(-size_x / 2 * 1e-9, size_x / 2 * 1e-9, x_points)
        y_pos = np.linspace(-size_y / 2 * 1e-9, size_y / 2 * 1e-9, y_points)
        z_pos = np.linspace(-size_z / 2 * 1e-9, size_z / 2 * 1e-9, z_points)

        if not resume:
            working_dir = os.path.join(self.cur_path, "TO",
                                       'x{:04d}_y{:04d}_f{:04d}_m{:04d}'
                                       .format(size_x, size_y, int(filter_R * 1e9), int(min_feature_size * 1e9)))
            assert is_list_with_two_numbers(initial_cond), "Initial_cond must be a list of two numbers (mean, range)!"

            if initial_type == "uniform":
                initial_cond = np.random.rand(x_points, y_points) * initial_cond[0] + initial_cond[1]
            elif initial_type == "blurred":
                initial_cond = generate_blurred_noise(x_points, y_points, initial_cond[0], initial_cond[1],
                                                      os.path.dirname(working_dir), str(epoch))
            elif initial_type == "perlin":
                initial_cond = generate_perlin_noise(x_points, y_points, initial_cond[0], initial_cond[1],
                                                     os.path.dirname(working_dir), str(epoch))
            else:
                assert False, "Initial type must be uniform, blurred or perlin!"

            runSim(initial_cond, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R=filter_R,
                   min_feature_size=min_feature_size, working_dir=working_dir, beta=startingBeta,
                   scale_initial_gradient_to=start_gradient, lsf_filename=lsf_filename)
            working_dir = working_dir + "_{}".format(epoch)
        else:
            assert False, "The resume program has not been developed yet, sorry!"

        return working_dir

    ################## ANALOG TO DIGITAL CONVERSION USING ATDO ######################
    def ATDO(self, TO_working_dir, epoch=0, scan_order="row_by_row", resume=False):
        """
        :param TO_working_dir: directory name of TO files
        :param epoch: epoch number (corresponding to the TO files)
        :param scan_order: pixel scan order
        :param resume: set to True when you want to resume the optimization from last interruption
        :return: this function returns nothing but a well-optimized device
        """

        fdtd_filename = self.config_for_a2d(TO_working_dir=TO_working_dir, lsf_filename="base_A2D.lsf")
        file_path = os.path.dirname(fdtd_filename[0])

        logger = self.logger
        fdtd = [FDTDSimulation(hide=self.hide_cad, load_file=filename) for filename in fdtd_filename]
        if scan_order == "row_by_row":
            order = np.arange(0, self.pixels_number)
        elif scan_order == "col_by_col":
            order = np.arange(0, self.pixels_number).reshape(-1, self.pixels_per_row).flatten(order="F")
        elif scan_order == "random":
            order = np.random.permutation(self.pixels_number)
        else:
            assert False, "scan_order must be row_by_row, col_by_col, or random"
        A2D_pixel_array = []
        self.fdtd = fdtd

        if resume:
            print("---------------------------------------------------------")
            print("Now we are resuming the interrupted project ...\n")
            npz_path = os.path.join(file_path, "A2D_states.npz")
            iterations = np.load(npz_path)["iterations"]
            order = np.load(npz_path)["order"]
            A2D_pixel_array = np.load(npz_path)["A2D_pixel_array"]
            A2D_pixel_array = A2D_pixel_array.tolist()
            print("---------------------------------------------------------")
            print("Now we are reloading the pixels ...\n")
            self.load_pixels(array=A2D_pixel_array, order=order, to_n=iterations, for_resume_a2d=True)
        else:
            iterations = 0
            for fdtd in self.fdtd:
                fdtd.fdtd.run()
            for fdtd in self.fdtd:
                fdtd.fdtd.switchtolayout()

        # START PIXEL FLIPPING OPERATION
        for n in range(iterations, self.pixels_number):
            for fdtd in self.fdtd:
                self.flipPixel(fdtd, order[n], to_state=1)

            etch = self.cost_function(filename=fdtd_filename)
            etch_cost = self.target_fom * len(etch) - sum(etch)

            for fdtd in self.fdtd:
                self.flipPixel(fdtd, order[n], to_state=0)

            deposition = self.cost_function(filename=fdtd_filename)
            deposition_cost = self.target_fom * len(deposition) - sum(deposition)

            etch_info = "; ".join(["TE%d:%.3f" % (self.out_mode_list[i], etch[i]) for i in range(len(etch))])
            deposit_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], deposition[i]) for i in range(len(deposition))])
            progress_info = "%.3f%%" % ((n + 1) / self.pixels_number * 100)
            log_info = f"[Etch]{etch_info}; [Deposit]{deposit_info}. Pixel {order[n]} is deposited; [Progress]{progress_info}"

            if etch_cost > deposition_cost:
                logger.info(log_info)
                A2D_pixel_array.append(0)
            else:
                # Flip the pixel state back to 1 ("deposition")
                for fdtd in self.fdtd:
                    self.flipPixel(fdtd, order[n], to_state=1)
                logger.info(log_info)
                A2D_pixel_array.append(1)

            np.savez(os.path.join(file_path, "A2D_states.npz"), iterations=n + 1, A2D_pixel_array=A2D_pixel_array,
                     order=order)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()

        # rearrange the pixel array in this order: 1, 2, 3, 4 ...
        order_original = np.argsort(order)
        for i in range(self.pixels_number):
            temp = copy.deepcopy(A2D_pixel_array)
            A2D_pixel_array[i] = temp[order_original[i]]

        pixel_array = np.array(A2D_pixel_array).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez(os.path.join(file_path, 'initial_DBS.npz'), initial_DBS=pixel_array)
        np.savez(os.path.join(file_path, 'scan_order.npz'), order=order)

        if os.path.exists(os.path.join(file_path, "A2D_states.npz")):
            os.remove(os.path.join(file_path, "A2D_states.npz"))

        move_files_to(file_path, os.path.join(file_path, "data"))
        new_dirname = "data_epoch_" + str(epoch)
        os.rename(os.path.join(file_path, "data"), os.path.join(file_path, new_dirname))

    ################## ANALOG TO DIGITAL CONVERSION USING EG-ATDO ######################
    def EG_ATDO(self, TO_working_dir, epoch=0, scan_order="row_by_row", resume=False, resume_from_cur_file=False):
        """
        :param TO_working_dir: directory name of TO files
        :param epoch: epoch number (corresponding to the TO files)
        :param scan_order: pixel scan order
        :param resume: set to True when you want to resume the optimization from last interruption
        :return: this function returns nothing but a well-optimized device
        :param resume_from_cur_file: resume without reloading the pixels
        """

        fdtd_filename, conversion_matrix = self.config_for_eg_a2d(TO_working_dir=TO_working_dir,
                                                                  lsf_filename="base_EGA2D.lsf",
                                                                  resume=resume_from_cur_file)
        file_path = os.path.dirname(fdtd_filename[0])


        logger = self.logger
        fdtd = [FDTDSimulation(hide=self.hide_cad, load_file=filename) for filename in fdtd_filename]

        edge = []
        etch_convert = []
        deposit_convert = []

        # 1.Obtain direct_etch_conversion matrix and direct_deposition_conversion matrix through conversion matrix
        # 2.Obtain edge pixel matrix which needs to be further determined through flip-pixel operation
        # Warning: matrix values are sometimes a little bit away from 0, 0.5, and 1
        for row_idx, row in enumerate(conversion_matrix):
            for col_idx, col in enumerate(row):
                if (col > 0.49) and (col < 0.51):
                    edge.append(row_idx * len(row) + col_idx)
                if col > 0.99:
                    etch_convert.append(row_idx * len(row) + col_idx)
                if col < 0.01:
                    deposit_convert.append(row_idx * len(row) + col_idx)

        if scan_order == "row_by_row":
            order = edge
        elif scan_order == "col_by_col":
            position = np.arange(self.pixels_number).reshape((self.pixels_per_col, self.pixels_per_row))
            order = []
            for col in range(self.pixels_per_row):
                for row in range(self.pixels_per_col):
                    if position[row][col] in edge:
                        order.append(position[row][col])
            order = np.array(order)
        else:
            assert False, "Scan_order must be row_by_row, col_by_col! Random_order is under developing!"
        A2D_pixel_array = conversion_matrix.reshape(-1)
        self.fdtd = fdtd

        if resume:
            # In EG-ATD, the saved pixel array is in proper sequence (i.e., row_by_row)
            print("---------------------------------------------------------")
            print("Now we are resuming the interrupted project ...\n")
            npz_path = os.path.join(file_path, "A2D_states.npz")
            iterations = np.load(npz_path)["iterations"]
            order = np.load(npz_path)["order"]
            A2D_pixel_array = np.load(npz_path)["A2D_pixel_array"]
            A2D_pixel_array = A2D_pixel_array.tolist()
            if resume_from_cur_file is False:
                print("---------------------------------------------------------")
                print("Now we are reloading the pixels ...\n")
                self.load_pixels_eg(array=A2D_pixel_array, for_resume_a2d=True)
                print("---------------------------------------------------------")
                print("Pixel loading is done. Now the flip-pixel sweep begins ...\n")
            else:
                print("---------------------------------------------------------")
                print("The optimization is resumed from current files, which don't require reloading the pixels!")
        else:
            iterations = 0
            print("---------------------------------------------------------")
            print("Now we are adding directly-converted pixels ...\n")
            for fdtd in self.fdtd:
                for i in range(len(etch_convert)):
                    self.flipPixel(fdtd, etch_convert[i], to_state=1)
                for i in range(len(deposit_convert)):
                    self.flipPixel(fdtd, deposit_convert[i], to_state=0)
            print("---------------------------------------------------------")
            print("Direct conversion is done. Now the flip-pixel sweep begins ...\n")

        reopen_flag = 0
        for n in range(iterations, len(order)):
            if reopen_flag == 1:
                reopen_flag = 0
                print("---------------------------------------------------------")
                print("To prevent annoying stop of optimization, we close those fdtd files and reopen them now...")
                for i in range(len(fdtd_filename)):
                    self.fdtd[i].fdtd.eval('save("%s");' % fdtd_filename[i])
                    # if we don't save fdtd files, the previous etched-back pixel will not be saved,
                    # and all pixels will be deposited
                for fdtd in self.fdtd:
                    fdtd.fdtd.close()

                fdtd = [FDTDSimulation(hide=self.hide_cad, load_file=filename) for filename in fdtd_filename]
                self.fdtd = fdtd

            reopen_flag += 1

            print(f"Starting ITER.{n} of total {len(order)} iterations...")
            for fdtd in self.fdtd:
                print("Now we etch this pixel...")
                self.flipPixel(fdtd, order[n], to_state=1)

            etch = self.cost_function(filename=fdtd_filename)
            etch_cost = self.target_fom * len(etch) - sum(etch)

            for fdtd in self.fdtd:
                print("Now we deposit this pixel...")
                self.flipPixel(fdtd, order[n], to_state=0)

            deposition = self.cost_function(filename=fdtd_filename)
            deposition_cost = self.target_fom * len(deposition) - sum(deposition)

            etch_info = "; ".join(["TE%d:%.3f" % (self.out_mode_list[i], etch[i]) for i in range(len(etch))])
            deposit_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], deposition[i]) for i in range(len(deposition))])
            progress_info = "%.3f%%" % ((n + 1) / len(order) * 100)

            if etch_cost > deposition_cost:
                print("Deposit is better, the pixel remains deposited!")
                log_info = f"[Etch]{etch_info}; [Deposit]{deposit_info}. Pixel {order[n]} is deposited; [Progress]{progress_info}"
                logger.info(log_info)
                A2D_pixel_array[order[n]] = 0
            else:
                # Flip the pixel state back to 1 ("deposition")
                print("Etch is better, the pixel is etched back!")
                for fdtd in self.fdtd:
                    self.flipPixel(fdtd, order[n], to_state=1)
                log_info = f"[Etch]{etch_info}; [Deposit]{deposit_info}. Pixel {order[n]} is etched; [Progress]{progress_info}"
                logger.info(log_info)
                A2D_pixel_array[order[n]] = 1

            np.savez(os.path.join(file_path, "A2D_states.npz"), iterations=n + 1, A2D_pixel_array=A2D_pixel_array,
                     order=order)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()

        pixel_array = np.array(A2D_pixel_array).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez(os.path.join(file_path, 'initial_DBS.npz'), initial_DBS=pixel_array)
        np.savez(os.path.join(file_path, 'scan_order.npz'), order=order)

        if os.path.exists(os.path.join(file_path, "A2D_states.npz")):
            os.remove(os.path.join(file_path, "A2D_states.npz"))

        move_files_to(file_path, os.path.join(file_path, "data"))
        new_dirname = "data_epoch_" + str(epoch)
        os.rename(os.path.join(file_path, "data"), os.path.join(file_path, new_dirname))

    ################## DBS OPTIMIZATION USING OPEN-ACCESS LIBRARY ######################
    def DBS(self, resume, epoch, scan_order="row_by_row"):
        """
        :param resume: set to True when you want to resume the interrupted optimization
        :param epoch: epoch number
        :param scan_order: pixel scan order
        :return: return nothing but a well-optimized device
        """

        initial_condition = np.load(os.path.join("A2D", "data_{}".format(epoch), "initial_DBS.npz"))['initial_DBS']
        fdtd_filename = self.config_for_dbs(lsf_filename="base_DBS.lsf")
        # definitions for gdsii file creation
        design_region_width = self.design_region_y * 1e6
        design_region_length = self.design_region_x * 1e6
        file_path = os.path.dirname(fdtd_filename[0])
        data_path = os.path.join(file_path, "data")

        self.logger = self.logger(txt=os.path.join(file_path, "log.txt"))

        if self.self_define_materials:
            etch = "n_bg"
        else:
            etch = "SiO2 (Glass) - Palik"
        self.fdtd = []
        pixels = []

        # initialize the simulation frame
        for filename in fdtd_filename:
            if "initial" not in filename:
                self.fdtd.append(FDTDSimulation(hide=self.hide_cad, fdtd_path="C:\\Program Files\\Lumerical\\v202\\api\\python",
                                                load_file=filename.replace("_model", "")))

        ## pixels region definition
        if self.shape == "rect":
            for fdtd in self.fdtd:
                pixels.append(RectanglePixelsRegion(Point(-design_region_length / 2 + self.offset_x * 1e6,
                                                          -design_region_width / 2 + self.offset_y * 1e6),
                                                    Point(design_region_length / 2 + self.offset_x * 1e6,
                                                          design_region_width / 2 + self.offset_y * 1e6),
                                                    pixel_x_length=self.rect_size_x * 1e6,
                                                    pixel_y_length=self.rect_size_y * 1e6, fdtd_engine=fdtd,
                                                    material=etch, z_start=self.z_min * 1e6, z_end=self.z_max * 1e6))
        elif self.shape == "circle":
            for fdtd in self.fdtd:
                pixels.append(CirclePixelsRegion(Point(-design_region_length / 2 + self.offset_x * 1e6,
                                                       -design_region_width / 2 + self.offset_y * 1e6),
                                                 Point(design_region_length / 2 + self.offset_x * 1e6,
                                                       design_region_width / 2 + self.offset_y * 1e6),
                                                 pixel_radius=self.radius * 1e6, fdtd_engine=fdtd, material=etch,
                                                 z_start=self.z_min * 1e6, z_end=self.z_max * 1e6))

        def cost_function(based_matrix):
            for i in range(len(self.fdtd)):
                pixels[i].update(based_matrix.reshape(self.pixels_per_row, self.pixels_per_col))
                self.fdtd[i].fdtd.eval('save("%s");' % fdtd_filename[i].replace("_model", ""))
                self.fdtd[0].fdtd.eval('addjob("%s");' % fdtd_filename[i].replace("_model", ""))

            self.fdtd[0].fdtd.eval('runjobs;')

            for i in range(len(self.fdtd)):
                self.fdtd[i].fdtd.eval('load("%s");' % fdtd_filename[i].replace("_model", ""))

            mode_transmission = []
            for i, fdtd in enumerate(self.fdtd):
                mode_transmission.append(np.mean(fdtd.get_mode_transmission(f"fom{i + 1}_mode_exp")[0, 1, :]))

            fom_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], mode_transmission[i]) for i in range(len(mode_transmission))])

            self.logger.info(fom_info)
            FOM = len(mode_transmission) - sum(mode_transmission)
            return FOM

        def final_file_arrange():
            move_files_to(file_path, data_path)
            new_dirname = "data_epoch_" + str(epoch)
            os.rename(data_path, os.path.join(file_path, new_dirname))

        def call_back():
            pass

        # DEFINE DBS OPTIMIZATION INSTANCE
        initial_array = initial_condition
        DBS = DirectBinarySearchAlgorithm(loS=self.pixels_number, pixels_per_row=self.pixels_per_row,
                                          pixels_per_col=self.pixels_per_col,
                                          cost_function=cost_function, max_iteration=400, threshold=0.5e-2,
                                          scan_order=scan_order,
                                          callback_function=call_back, initial_solution=initial_array,
                                          logger=self.logger,
                                          resume=resume, data_savepath=data_path,
                                          )

        # RUN DBS OPTIMIZATION
        DBS.run()
        # SAVE OPTIMIZATION RESULTS
        npztomatlab(os.path.join(data_path, "final_DBS.mat"), name="final_DBS", array=DBS.best_solution)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()
        final_file_arrange()

    #########################################
    ########## Functional Methods ###########
    #########################################
    class Simulation:
        def __init__(self, fdtdname):
            self.eta = 0.5
            self.dx = 20e-9
            self.dy = 20e-9
            self.dz = 20e-9

            self.eps_max = None
            self.eps_min = None

            self.params = None
            self.sim = None
            self.simname = fdtdname
            self.beta = None

            self.eps = None

            self.x = None
            self.y = None
            self.z = None

        def getnpzsim(self, npzfilename: str):
            if npzfilename[-3:] == 'npz':
                data = np.load(npzfilename)
                self.params = data['params']
                self.eps_max = data['eps_max']
                self.eps_min = data['eps_min']
                self.x = data['x']
                self.y = data['y']
                self.z = data['z']
                self.beta = data['beta']
                self.eps = data['eps']
                self.sim = lumapi.FDTD(filename=self.simname)
                self.sim.switchtolayout()

        def get_eps_from_params(self):
            rho = np.reshape(self.params, (len(self.x), len(self.y)))

            ## Use script function to convert the raw parameters to a permittivity distribution and get the result
            self.sim.putv("topo_rho", rho)
            self.sim.eval(('params = struct;'
                           'params.eps_levels=[{0},{1}];'
                           'params.filter_radius = {2};'
                           'params.beta = {3};'
                           'params.eta = {4};'
                           'params.dx = {5};'
                           'params.dy = {6};'
                           'params.dz = 0.0;'
                           'eps_geo = topoparamstoindex(params,topo_rho);').format(self.eps_min, self.eps_max,
                                                                                   self.filter_R,
                                                                                   self.beta, self.eta, self.dx,
                                                                                   self.dy))
            eps = self.sim.getv("eps_geo")

            return eps

        def set_spatial_interp(self, monitor_name, setting):
            script = 'select("{}");set("spatial interpolation","{}");'.format(monitor_name, setting)
            self.sim.eval(script)

        def add_geo(self, npzfilename: str, filter_R=100e-9, min_feature_size=0.0):
            self.filter_R = filter_R
            self.min_feature_size = min_feature_size
            self.getnpzsim(npzfilename=npzfilename)
            eps = self.get_eps_from_params()

            script = ('addimport;'
                      'set("detail",1);')
            self.sim.eval(script)

            full_eps = np.broadcast_to(eps[:, :, None], (
                len(self.x), len(self.y), len(self.z)))  # < TODO: Move to Lumerical script to reduce transfers

            self.sim.putv('x_geo', self.x)
            self.sim.putv('y_geo', self.y)
            self.sim.putv('z_geo', self.z)
            self.sim.putv('eps_geo', full_eps)

            ## We delete and re-add the import to avoid a warning
            script = ('select("import");'
                      'delete;'
                      'addimport;'
                      'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);')
            self.sim.eval(script)
            self.sim.save()
            self.sim.close()

    def flipPixel(self, fdtd, n, to_state):
        if self.shape == 'rect':
            fdtd.fdtd.eval("switchtolayout;")
            i = n // self.pixels_per_row
            j = n % self.pixels_per_row
            if to_state == 1:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_bg" if self.self_define_materials else "SiO2 (Glass) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            elif to_state == 0:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            else:
                raise ValueError('State Invalid value')
        elif self.shape == "circle":
            fdtd.fdtd.eval("switchtolayout;")
            i = n // self.pixels_per_row
            j = n % self.pixels_per_row
            if to_state == 1:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
                fdtd.fdtd.addcircle(
                    name="pixel" + str(n),
                    material="n_bg" if self.self_define_materials else "SiO2 (Glass) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    radius=self.radius,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            elif to_state == 0:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            else:
                raise ValueError('State Invalid value')

    def load_pixels(self, array, order=None, to_n=None, fdtd=None, for_resume_a2d=False):
        if for_resume_a2d:
            if to_n is None:
                assert False, "for resume A2D, to_n must be specified"
            for fdtd in self.fdtd:
                for i in range(to_n):
                    self.flipPixel(fdtd, order[i], to_state=array[i])
        else:
            for i in range(self.pixels_number):
                if array[i] in (0, 1):
                    # only add pixels whose values equal 0 or 1
                    self.flipPixel(fdtd, i, to_state=array[i])

    def load_pixels_eg(self, array, fdtd=None, for_resume_a2d=False):
        load_order = np.arange(0, self.pixels_number)
        if for_resume_a2d:
            for fdtd in self.fdtd:
                for i in range(self.pixels_number):
                    if array[i] in (0, 1):
                        # only add pixels whose values equal 0 or 1
                        self.flipPixel(fdtd, load_order[i], to_state=array[i])
        else:
            # work for the function-input FDTD rather the self.fdtd
            for i in range(self.pixels_number):
                if array[i] in (0, 1):
                    # only add pixels whose values equal 0 or 1
                    self.flipPixel(fdtd, load_order[i], to_state=array[i])

    def cost_function(self, filename):
        print("Now we run FDTD to get FOM...")
        for i in range(len(filename)):
            self.fdtd[i].fdtd.eval('save("%s");' % filename[i])
            print(f"FDTD file {i} saved...")
            self.fdtd[0].fdtd.eval('addjob("%s");' % filename[i])
            print(f"FDTD job {i} added to the first FDTD GUI...")
        print("Running FDTD...")
        self.fdtd[0].fdtd.eval('runjobs;')

        for i in range(len(filename)):
            # self.fdtd[i].fdtd.eval('save("%s");' % filename[i])
            # print("FDTD completed and files saved...")
            self.fdtd[i].fdtd.eval('load("%s");' % filename[i])
            print(f"FDTD {i} loaded...")

        mode_transmission = []
        for i, fdtd in enumerate(self.fdtd):
            mode_transmission.append(np.mean(fdtd.get_mode_transmission(f"fom{i + 1}_mode_exp")[0, 1, :]))
            print("Monitor data obtained...")

        return mode_transmission

    def get_digital_array_from_text(self, file):
        result = [0] * self.pixels_number  # 初始化结果列表为全零
        i = -1
        with open(file, 'r') as f:
            for line in f:
                i += 1
                last_item = line.strip().split()[-1]
                if last_item == 'deposited':
                    result[i] = 0
                elif last_item == 'etched':
                    result[i] = 1
                else:
                    raise ValueError('Invalid value')
            if i != (self.pixels_number - 1):
                raise ValueError('Invalid length')

        # transform to splayout DBS algorithm matrix order
        pixel_array = np.array(result).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez('initial_DBS.npz', initial_DBS=pixel_array)

        return pixel_array

    #########################################
    ######### Configuration Methods #########
    #########################################
    def config_for_lsf(self, dir_name, lsf_filename):
        base_script = load_from_lsf(os.path.join(dir_name, lsf_filename))
        base_script = base_script.replace("opt_size_x=none", "opt_size_x={}".format(self.design_region_x))
        base_script = base_script.replace("opt_size_y=none", "opt_size_y={}".format(self.design_region_y))
        base_script = base_script.replace("num_in_wg=none", "num_in_wg={}".format(self.num_in_wg))
        base_script = base_script.replace("num_out_wg=none", "num_out_wg={}".format(self.num_out_wg))
        base_script = base_script.replace("lam_c=none", "lam_c={}".format(self.center_wavelength))
        base_script = base_script.replace("out_wg_width=none", "out_wg_width={}".format(self.output_wg_width))
        base_script = base_script.replace("in_wg_width=none", "in_wg_width={}".format(self.input_wg_width))
        base_script = base_script.replace("wavelength_span=none",
                                          "wavelength_span={}".format(self.wavelength_range))
        base_script = base_script.replace("in_mode_list=none", "in_mode_list={}".format(self.in_mode_list))
        base_script = base_script.replace("out_mode_list=none", "out_mode_list={}".format(self.out_mode_list))
        base_script = base_script.replace("mode_num=none", "mode_num={}".format(self.mode_num))

        if self.processor is 'GPU':
            script = ('select("FDTD");'
                      'set("z min bc","PML");'
                      'set("express mode",true);'
                      'setresource("FDTD","GPU",true);'
                      )
        else:
            script = 'setresource("FDTD","CPU",true);'
        base_script = base_script + script
        return base_script

    def config_for_a2d(self, TO_working_dir, lsf_filename=None):
        print(TO_working_dir)
        filter_R = int(TO_working_dir.split("_f")[1][:4]) * 1e-9
        min_feature_size = int(TO_working_dir.split("_m")[1][:4]) * 1e-9
        target_folder = os.path.join(self.cur_path, "A2D")
        copy_specified_filename_to(target_folder=target_folder, filename="_A2D")
        param_path = export_TO_param_for_a2d(working_dir=TO_working_dir, type="MAX")
        file_list = []
        if lsf_filename:
            self.config_for_a2d_fdtd_file(target_folder, lsf_filename)

        for file in os.listdir(target_folder):
            if file.endswith(".fsp"):
                fdtd = self.Simulation(fdtdname=os.path.join(target_folder, file))
                fdtd.add_geo(npzfilename=param_path, filter_R=filter_R, min_feature_size=min_feature_size)
                file_list.append(os.path.join(target_folder, file))

        return file_list

    def config_for_a2d_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(dir_name=target_folder, lsf_filename=lsf_filename)
        if self.self_define_materials:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=true")
        else:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=false")

        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD()
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_A2D.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

    def config_for_eg_a2d(self, TO_working_dir, lsf_filename=None, resume=False):
        print(TO_working_dir)
        filter_R = int(TO_working_dir.split("_f")[1][:4]) * 1e-9
        min_feature_size = int(TO_working_dir.split("_m")[1][:4]) * 1e-9
        target_folder = os.path.join(self.cur_path, "EGA2D")
        copy_specified_filename_to(target_folder=target_folder, filename="_EGA2D")
        param_path = export_TO_param_for_a2d(working_dir=TO_working_dir, type="MAX")
        conversion_matrix = get_conversion_matrix(analog_file=param_path, target_folder=target_folder,
                                                  pixel_size=self.rect_size_x, buff=self.buff)

        file_list = []
        if resume is False:
            if lsf_filename:
                self.config_for_eg_a2d_fdtd_file(target_folder, lsf_filename)
            for i in range(self.mode_num):
                for file in os.listdir(target_folder):
                    if file.endswith(f"{i}_EGA2D.fsp"):
                        fdtd = self.Simulation(fdtdname=os.path.join(target_folder, file))
                        fdtd.add_geo(npzfilename=param_path, filter_R=filter_R, min_feature_size=min_feature_size)
                        file_list.append(os.path.join(target_folder, file))
        else:
            for i in range(self.mode_num):
                for file in os.listdir(target_folder):
                    if file.endswith(f"{i}_EGA2D.fsp"):
                        file_list.append(os.path.join(target_folder, file))
        print(file_list)
        return file_list, conversion_matrix

    def config_for_eg_a2d_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(dir_name=target_folder, lsf_filename=lsf_filename)

        if self.self_define_materials:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=true")
        else:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=false")

        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD()
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_EGA2D.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

    def config_for_dbs(self, lsf_filename=None):
        target_folder = os.path.join(self.cur_path, "DBS")
        copy_specified_filename_to(target_folder=target_folder, filename="_DBS")
        file_list = []
        if lsf_filename:
            self.config_for_dbs_fdtd_file(target_folder, lsf_filename)

        for file in os.listdir(target_folder):
            if file.endswith(".fsp"):
                file_list.append(os.path.join(target_folder, file))
        return file_list

    def config_for_dbs_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(target_folder, lsf_filename)

        if self.self_define_materials:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=true")
        else:
            strings = "self_define_materials=" + base_script.split("self_define_materials=")[1][:4]
            base_script = base_script.replace(strings, "self_define_materials=false")
        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD()
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_DBS.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

# End of this file
