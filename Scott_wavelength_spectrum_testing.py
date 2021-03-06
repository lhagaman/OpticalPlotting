import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fit_parameters_grid, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, chi_squared, get_relative_gaussian_weights
import time

path_run1 = "First Xe Run Measurements\\first measurements with no bubbles in cell 11-01-2\\"
vacuum_path = "First Xe Run Measurements\\follow-up vacuum measurements\\"

vacuum_1_path = vacuum_path + "Sample 1\\"
v1_75 = Run(vacuum_1_path + "2018_11_09__13_41_18.txt")
v1_67 = Run(vacuum_1_path + "2018_11_09__13_47_53.txt")
v1_60 = Run(vacuum_1_path + "2018_11_09__13_55_54.txt")
v1_52 = Run(vacuum_1_path + "2018_11_09__14_02_45.txt")
v1_45 = Run(vacuum_1_path + "2018_11_09__14_08_54.txt")
v1_30 = Run(vacuum_1_path + "2018_11_09__14_14_55.txt")
vacuum_1_runs = [v1_30, v1_45, v1_52, v1_60, v1_67, v1_75]

vacuum_9_path = vacuum_path + "Sample 9\\"
v9_75 = Run(vacuum_9_path + "2018_11_09__14_22_32.txt")
v9_67 = Run(vacuum_9_path + "2018_11_09__14_33_41.txt")
v9_60 = Run(vacuum_9_path + "2018_11_09__14_40_54.txt")
v9_52 = Run(vacuum_9_path + "2018_11_09__14_47_23.txt")
v9_45 = Run(vacuum_9_path + "2018_11_09__14_53_56.txt")
v9_30 = Run(vacuum_9_path + "2018_11_09__15_00_15.txt")
vacuum_9_runs = [v9_30, v9_45, v9_52, v9_60, v9_67, v9_75]

path = "2nd Xenon Run Measurements\\"
#path = "vuv_height_comparison_and_first_data/M18 turn polished/center of sample/"
# m18pol30 = Run(path + "2018_08_30__11_23_42.txt") 
# m18pol45 = Run(path + "2018_08_30__11_29_08.txt") 
# m18pol60 = Run(path + "2018_08_30__11_34_35.txt") 
# m18pol75 = Run(path + "2018_08_30__11_39_45.txt") 
# nxt8530 = Run(path + "2018_08_30__14_13_18.txt") 
# nxt8545 = Run(path + "2018_08_30__14_19_14.txt") 
# nxt8560 = Run(path + "2018_08_30__14_24_51.txt") 
# nxt8575 = Run(path + "2018_08_30__14_30_46.txt")
s5_30 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_24_00.txt")
s5_45 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_29_18.txt")
s5_52 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_38_44.txt") # Two files exist for 52 deg; first probably got redone bc the angle range was too small
s5_60 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_43_44.txt") 
s5_60.change_theta_i(59)
s5_67 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_48_38.txt")
s5_67.change_theta_i(67)
s5_75 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_54_19.txt")
s5_75.change_theta_i(74)
s8_30 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_52_24.txt")
s8_45 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_46_48.txt")
s8_52 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_42_00.txt")
s8_60 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_57_12.txt") 
s8_67 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_01_51.txt")
s8_67.change_theta_i(66.5)
s8_75 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_06_25.txt") 
s8_75.change_theta_i(74.5)
# s9_first_30 = Run(path_run1 + "Sample 9\\2018_11_02__20_38_01.txt")
# s9_first_45 = Run(path_run1 + "Sample 9\\2018_11_02__20_32_34.txt")
# s9_first_52 = Run(path_run1 + "Sample 9\\2018_11_02__20_27_27.txt")
# s9_first_60 = Run(path_run1 + "Sample 9\\2018_11_02__20_22_13.txt")
# s9_first_67 = Run(path_run1 + "Sample 9\\2018_11_02__20_17_11.txt")
# s9_first_75 = Run(path_run1 + "Sample 9\\2018_11_02__20_11_44.txt")
s9_bubbles_30 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_26_05.txt")
s9_bubbles_45 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_21_25.txt")
s9_bubbles_52 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_16_32.txt")
s9_bubbles_60 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_11_46.txt") 
s9_bubbles_67 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_07_06.txt")
s9_bubbles_75 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_02_17.txt") 
s9_nobubbles_30 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_05_41.txt")
s9_nobubbles_45 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_01_01.txt")
s9_nobubbles_52 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_56_06.txt")
s9_nobubbles_60 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_51_00.txt") 
s9_nobubbles_67 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_46_13.txt")
s9_nobubbles_75 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_40_45.txt") 
s9_getter_30 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_13_32.txt")
s9_getter_45 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_08_40.txt")
s9_getter_52 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_03_50.txt")
s9_getter_60 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_58_47.txt") 
s9_getter_67 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_53_50.txt")
s9_getter_75 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_48_50.txt") 
s9_lowp_30 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_52_04.txt")
s9_lowp_30.change_theta_i(29)
s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
s9_lowp_45.change_theta_i(47)
s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
s9_lowp_52.change_theta_i(53)
s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt")
s9_lowp_60.change_theta_i(59) 
s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_67.change_theta_i(67)
s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
s9_lowp_75.change_theta_i(75)
s9_lowp2_30 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_28_41.txt")
s9_lowp2_30.change_theta_i(29)
s9_lowp2_45 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_23_51.txt")
s9_lowp2_45.change_theta_i(47)
s9_lowp2_52 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_18_50.txt")
s9_lowp2_52.change_theta_i(52)
s9_lowp2_60 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_14_04.txt") 
s9_lowp2_60.change_theta_i(59)
s9_lowp2_67 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_09_15.txt")
s9_lowp2_67.change_theta_i(66)
s9_lowp2_75 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_04_19.txt")
s9_lowp2_75.change_theta_i(74)
s9_medp_30 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_56_28.txt")
s9_medp_45 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_51_07.txt")
s9_medp_52 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_44_43.txt")
s9_medp_60 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_39_51.txt") 
s9_medp_67 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_35_13.txt")
s9_medp_75 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_29_31.txt")  
s9_medp2_30 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_42_49.txt")
s9_medp2_45 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_36_41.txt")
s9_medp2_52 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_31_00.txt")
s9_medp2_60 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_25_38.txt") 
s9_medp2_67 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_20_48.txt")
s9_medp2_75 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_14_47.txt")
s9_medp2_75.change_theta_i(75) 
s9_hip_30 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_33_42.txt")
s9_hip_45 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_28_40.txt")
s9_hip_52 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_19_59.txt")
s9_hip_60 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_15_21.txt") 
s9_hip_67 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_10_17.txt")
s9_hip_75 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_03_53.txt") 
# s9_hip2_30 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_59_52.txt")
# s9_hip2_30.change_theta_i(28)
# s9_hip2_45 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_55_11.txt")
# s9_hip2_45.change_theta_i(47)
# s9_hip2_52 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_50_31.txt")
# s9_hip2_52.change_theta_i(54)
# s9_hip2_60 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_45_32.txt") 
# s9_hip2_60.change_theta_i(58)
# s9_hip2_67 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_40_40.txt")
# s9_hip2_67.change_theta_i(65)
# s9_hip2_75 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_35_58.txt") 
# s9_hip2_75.change_theta_i(73)
s9_lowp_above_30 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_37_47.txt")
s9_lowp_above_45 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_32_48.txt")
s9_lowp_above_52 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_27_11.txt")
s9_lowp_above_60 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_22_25.txt") 
s9_lowp_above_67 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_17_42.txt")
s9_lowp_above_75 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_12_10.txt")


# Second run, data at different wavelengths
s9_400nm_30 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_12_04.txt")
s9_400nm_30.change_theta_i(28)
s9_400nm_45 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_45.change_theta_i(45)
s9_400nm_52 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_52.change_theta_i(50)
s9_400nm_60 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_60.change_theta_i(58)
s9_400nm_67 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_67.change_theta_i(65)
s9_400nm_75 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
s9_400nm_75.change_theta_i(73)

s9_300nm_30 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_30.change_theta_i(30)
s9_300nm_45 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_45.change_theta_i(45) 
s9_300nm_52 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_52.change_theta_i(52) 
s9_300nm_60 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt")
s9_300nm_60.change_theta_i(60) 
s9_300nm_67 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_67.change_theta_i(65)
s9_300nm_75 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
s9_300nm_75.change_theta_i(76.001)

s9_220nm_30 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_30.change_theta_i(30)
s9_220nm_45 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_45.change_theta_i(45)
s9_220nm_52 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_52.change_theta_i(52)
s9_220nm_60 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt") 
s9_220nm_60.change_theta_i(60)
s9_220nm_67 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_67.change_theta_i(65)
s9_220nm_75 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
s9_220nm_75.change_theta_i(74)

s9_165nm_30 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_30.change_theta_i(28)
s9_165nm_45 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_45.change_theta_i(44)
s9_165nm_52 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_52.change_theta_i(50)
s9_165nm_60 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_60.change_theta_i(58)
s9_165nm_67 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt")
s9_165nm_67.change_theta_i(66)
s9_165nm_75 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
s9_165nm_75.change_theta_i(74)

path_run3 = "3rd Xenon Run Measurements\\" # Any kind of correction for drifting power? Strongest in 3rd run
s9_r3_lowp_30 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__10_53_38.txt")
s9_r3_lowp_45 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__10_58_38.txt")
s9_r3_lowp_52 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_03_46.txt")
s9_r3_lowp_60 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_09_09.txt")
s9_r3_lowp_67 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_14_26.txt")
s9_r3_lowp_67.change_theta_i(67)
s9_r3_lowp_75 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_19_23.txt")
s9_r3_lowp_75.change_theta_i(75)

s9_r3_hip_30 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_11_07.txt")
s9_r3_hip_45 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_16_35.txt")
s9_r3_hip_52 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_21_41.txt")
s9_r3_hip_60 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_26_59.txt")
s9_r3_hip_67 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_32_16.txt")
s9_r3_hip_67.change_theta_i(67)
s9_r3_hip_75 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_37_45.txt")
s9_r3_hip_75.change_theta_i(75)

s6_r3_lowp_30 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_06_37.txt")
s6_r3_lowp_45 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_17_22.txt")
s6_r3_lowp_52 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_22_29.txt")
s6_r3_lowp_60 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_27_34.txt")
s6_r3_lowp_60.change_theta_i(60)
s6_r3_lowp_67 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_32_34.txt")
s6_r3_lowp_67.change_theta_i(67)
s6_r3_lowp_75 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_37_33.txt")
s6_r3_lowp_75.change_theta_i(75)

s6_r3_hip_30 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_27_14.txt")
s6_r3_hip_45 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_32_41.txt")
s6_r3_hip_52 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_38_14.txt")
s6_r3_hip_60 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_44_40.txt")
s6_r3_hip_67 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_50_57.txt")
s6_r3_hip_67.change_theta_i(67)
s6_r3_hip_75 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_56_59.txt")
s6_r3_hip_75.change_theta_i(75)

s2_r3_lowp_30 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__15_55_53.txt")
s2_r3_lowp_45 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_04_03.txt")
s2_r3_lowp_52 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_10_48.txt")
s2_r3_lowp_60 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_17_59.txt")
s2_r3_lowp_67 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_26_30.txt")
s2_r3_lowp_67.change_theta_i(67)
s2_r3_lowp_75 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_33_07.txt")
s2_r3_lowp_75.change_theta_i(75)

s2_r3_lowp_above_30 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_29_53.txt")
s2_r3_lowp_above_45 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_35_01.txt")
s2_r3_lowp_above_52 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_39_48.txt")
s2_r3_lowp_above_60 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_44_38.txt")
s2_r3_lowp_above_67 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_49_26.txt")
s2_r3_lowp_above_67.change_theta_i(69)
s2_r3_lowp_above_75 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_54_08.txt")
s2_r3_lowp_above_75.change_theta_i(77)

s2_r3_hip_30 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_33_23.txt")
s2_r3_hip_45 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_40_41.txt")
s2_r3_hip_52 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_48_08.txt")
s2_r3_hip_60 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_54_57.txt")
s2_r3_hip_67 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__19_02_21.txt")
s2_r3_hip_67.change_theta_i(67)
s2_r3_hip_75 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__19_12_01.txt")
s2_r3_hip_75.change_theta_i(75)

runs = [s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_165nm_30,s9_165nm_45,s9_165nm_52,s9_165nm_60,s9_165nm_67,s9_165nm_75]#[s5_30,s5_45,s5_52,s5_60,s5_67,s5_75]#[s8_30,s8_45,s8_52,s8_60,s8_67,s8_75]#[s9_220nm_30,s9_220nm_45,s9_220nm_52,s9_220nm_60,s9_220nm_67,s9_220nm_75]#[s2_r3_hip_30,s2_r3_hip_45,s2_r3_hip_52,s2_r3_hip_60,s2_r3_hip_67,s2_r3_hip_75]#[s9_r3_lowp_30,s9_r3_lowp_45,s9_r3_lowp_52,s9_r3_lowp_60,s9_r3_lowp_67,s9_r3_lowp_75]#[s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_bubbles_30,s9_bubbles_45,s9_bubbles_52,s9_bubbles_60,s9_bubbles_67,s9_bubbles_75]#[s9_lowp_30,s9_lowp2_30,s9_medp_30,s9_medp2_30,s9_hip_30,s9_hip2_30]#[s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp2_30,s9_lowp2_45,s9_lowp2_52,s9_lowp2_60,s9_lowp2_67,s9_lowp2_75,s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_lowp_75,s9_medp_75,s9_hip_75]#[s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75,s9_getter_30,s9_getter_45,s9_getter_52,s9_getter_60,s9_getter_67,s9_getter_75]#[s9_first_30,s9_first_45,s9_first_52,s9_first_60,s9_first_67,s9_first_75,s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75]#
labels=[r"$\theta_i=30^{\circ}$","45$^{\circ}$","52$^{\circ}$","60$^{\circ}$", "67$^{\circ}$", "75$^{\circ}$"]#,"30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="LZ Skived"
plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 178 nm", log=True, labels=labels, include_legend=True, errorbars=True, legend_loc=0)
t0=time.time()

# Fit data
# fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[.8, 1.57, .1, 0.1, 1.0],average_angle=4, precision=.25, sigma_theta_i=-1, use_errs=True,use_spike=True, use_nu=False,bounds=([0.1,1.4,0.03,0.01,0.3],[1.6,2.6,0.6,50.,1.7]))
fit_params = [0.7759585422536841, 1.5859038386290387, 0.1152682251245288]#[.281, 1.8, 0.161]#[0.795, 1.586,	0.106]#

print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))

# Plot BRIDF model from fits
n_LXe_178 = 1.69
sigma_theta_i=-1
precision=.25
average_angle=4

# Sellmeier equation for LXe index vs wavelength
def n_Sel(lambda_,a0,aUV,aIR):
	return np.sqrt(a0+aUV*lambda_**2/(lambda_**2-146.9**2)+aIR*lambda_**2/(lambda_**2-827.0**2))
	
params_Sel_162=[1.45541588, 0.44780067, 0.00170906] # From overall fit to Sinnock data plus Grace plot at 178 nm; for ~162 K
params_Sel_178=[1.40705316, 0.449238886, 0.00128087090] # For ~178 K

def n_Sel_PTFE(lambda_,a0,aUV):
    return np.sqrt(a0+aUV*lambda_**2/(lambda_**2-161**2))
	
params_PTFE = [1.6296759,  0.09783294]

min_angle = 0
max_angle = 85
d_theta = 1
n_angles = (max_angle-min_angle)/d_theta+1
theta_r_list = np.linspace(min_angle, max_angle, n_angles)
phi_r = 0
theta_i_list = [29,47,53,59,67,75]#[29, 47, 53, 59, 67, 75]
colors=["r","g","b","m","c","y"]
mu_lambda = 178
# FWHM is ~4 nm/mm slit size; estimated to be ~3 nm for Run 2 LXe data (slit size ~0.75 mm)
sigma_lambda = 2#14/2.355#1.0# in nm; divide by 2.355 to convert from FWHM, assuming Gaussian
sigma_n = sigma_lambda/20
n_samples = 20
lambda_list = np.linspace(-2 * sigma_lambda + mu_lambda, sigma_lambda * 2 + mu_lambda,n_samples)
weights_list = get_relative_gaussian_weights(lambda_list, mu_lambda, sigma_lambda)
for i_th in range(len(theta_i_list)):
	theta_i = theta_i_list[i_th]
	BRIDF_weighted_sum = 0
	for ii in range(len(lambda_list)):
		lambda_i = lambda_list[ii]
		weight_i = weights_list[ii]
		#n_LXe_i = 1.90128173+(lambda_i-mu_lambda)*sigma_n/sigma_lambda # Use a Gaussian in index; scale to a reasonable sigma
		n_LXe_i = 1.69100164+(lambda_i-mu_lambda)*sigma_n/sigma_lambda # Use a Gaussian in index; scale to a reasonable sigma
		#n_LXe_i = n_Sel(lambda_i, *params_Sel_162) # calculate LXe index for this wavelength
		#print(n_LXe_i)
		#n_PTFE_i = fit_params[1]+n_Sel_PTFE(lambda_i, *params_PTFE)-n_Sel_PTFE(mu_lambda, *params_PTFE) # use fit to Claudio's data plus offset to match our own fits
		fit_params_i = fit_params[:]
		#fit_params_i[1] = n_PTFE_i
		BRIDF_weighted_sum += np.array(BRIDF_plotter(theta_r_list, phi_r, theta_i, n_LXe_i, 0.5, fit_params_i, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))*weight_i
	BRIDF_avg = BRIDF_weighted_sum/np.sum(weights_list)
	plt.plot(theta_r_list, BRIDF_avg, color=colors[i_th])
#plt.text(0.3,0.9,r"$\sigma_{{\lambda}}$={0:.2f} nm".format(sigma_lambda),transform=plt.gca().transAxes,fontsize=13)
plt.text(0.3,0.9,r"$\sigma_n$={0:.2f}".format(sigma_n),transform=plt.gca().transAxes,fontsize=13)
fit_text = r"Fit: $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(fit_params[0],fit_params[1],fit_params[2])
plt.text(0.05,0.05, fit_text,transform=plt.gca().transAxes,fontsize=13)
plt.legend()
	
# colors=["r","g","b","m","c","y"]
# phi_r_list=[0,1,2,5,9,15]
# for ii in range(len(phi_r_list)):
    # plot_TSTR_fit(75, n_LXe_178, fit_params, color=colors[ii], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=phi_r_list[ii],label="$\phi_r$={0}".format(phi_r_list[ii]))
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=5)
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=10)
t2=time.time()
print("Plotting time: {0}".format(t2-t1))

# Chi squared calc doesn't work since it calculates the model w/o wavelength averaging
# for ii in range(len(runs)):
	# run_data=get_independent_variables_and_relative_intensities(runs[ii])
	# chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
	# print("Chi squared from fit, "+labels[ii]+": ",chi_sq)
	
# run_data=get_independent_variables_and_relative_intensities(runs)
# chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# print("Chi squared from fit: ",chi_sq)




plt.show()

