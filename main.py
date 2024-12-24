from func_MDM_inverse_design import *

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

if __name__ == "__main__":
    inverse_design = MDM_InverseDesign(pitch=120e-9,
                                       shape="rect",
                                       rect_size=120e-9,
                                       radius=45e-9,
                                       z_min=-110e-9,
                                       z_max=110e-9,
                                       design_region_x=10e-6,
                                       design_region_y=6e-6,
                                       num_in_wg=5,
                                       num_out_wg=1,
                                       input_wg_width=0.5e-6,
                                       in_mode_list=[1, 1, 1, 1, 1],
                                       output_wg_width=2.5e-6,
                                       out_mode_list=[1, 2, 3, 4, 5],
                                       n_wg="Si (Silicon) - Palik", n_bg="SiO2 (Glass) - Palik",
                                       wavelength=1.55e-6,
                                       wavelength_range=40e-9,
                                       offset_x=0,
                                       offset_y=0,
                                       buff=0,
                                       self_define_materials=True,
                                       processor='CPU',
                                       hide_cad=True,
                                       )

    # PHASE = "TO"
    # PHASE = "ATD"
    PHASE = "EG_ATD"
    # PHASE = "DBS"

    epoch = 0
    if PHASE == "TO":
        # MEAN & RANGE
        TO_initial_pattern = [[0.99, 0.01]]
        inverse_design.TO(initial_cond=TO_initial_pattern[0], initial_type="perlin", epoch=epoch, resume=True, filter_R=240e-9)

    elif PHASE == "ATD":
        TO_working_dir = "D:\\PHDwork\\Future Work\\Multi-dimensional Photonic Convolution\\MDM DEMUX\\EG-ATDO\\TO\\WDM_x6000_y6000_f0100_m0000_0"
        inverse_design.ATDO(TO_working_dir=TO_working_dir, epoch=epoch, resume=False, scan_order="col_by_col")

    elif PHASE == "EG_ATD":
        TO_working_dir = "D:\\User data\\sal\\NEW EGATDO 5MODE DEMUX\\TO\\x10000_y6000_f0239_m0000_2"
        inverse_design.EG_ATDO(TO_working_dir=TO_working_dir, epoch=epoch, resume=True, resume_from_cur_file=True, scan_order="col_by_col")

    elif PHASE == "DBS":
        inverse_design.DBS(epoch=epoch, scan_order="row_by_row", resume=False)

    else:
        assert False, "We only support TO, ATD, EG_ATD & DBS for silicon photonic inverse design!"
