import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fit_parameters_grid, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, chi_squared
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
s5_67 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_48_38.txt")
s5_75 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_54_19.txt")
s8_30 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_52_24.txt")
s8_45 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_46_48.txt")
s8_52 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_42_00.txt")
s8_60 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_57_12.txt") 
s8_67 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_01_51.txt")
s8_67.change_theta_i(66.5)
s8_75 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_06_25.txt") 
s8_75.change_theta_i(75)
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
s9_lowp_30.change_theta_i(30)
s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
s9_lowp_52.change_theta_i(53)
s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt")
s9_lowp_60.change_theta_i(60) 
s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_67.change_theta_i(68)
s9_lowp_65 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_65.change_theta_i(66)
s9_lowp_69 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_69.change_theta_i(69)
s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
s9_lowp_73 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
s9_lowp_73.change_theta_i(74)
s9_lowp_77 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
s9_lowp_77.change_theta_i(76.001)
s9_lowp2_30 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_28_41.txt")
s9_lowp2_45 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_23_51.txt")
s9_lowp2_52 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_18_50.txt")
s9_lowp2_60 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_14_04.txt") 
s9_lowp2_67 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_09_15.txt")
s9_lowp2_75 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_04_19.txt")
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
s9_400nm_28 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_12_04.txt")
s9_400nm_28.change_theta_i(28)
s9_400nm_32 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_12_04.txt")
s9_400nm_32.change_theta_i(32)
s9_400nm_45 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_43 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_43.change_theta_i(43)
s9_400nm_47 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_47.change_theta_i(47)
s9_400nm_52 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_50 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_50.change_theta_i(50)
s9_400nm_54 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_54.change_theta_i(54)
s9_400nm_60 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_58 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_58.change_theta_i(58)
s9_400nm_62 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_62.change_theta_i(62)
s9_400nm_67 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_65 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_65.change_theta_i(65)
s9_400nm_69 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_69.change_theta_i(69)
s9_400nm_75 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
s9_400nm_73 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
s9_400nm_73.change_theta_i(73)
s9_400nm_77 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
s9_400nm_77.change_theta_i(77)

s9_300nm_30 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_28 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_28.change_theta_i(28)
s9_300nm_32 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_32.change_theta_i(32)
s9_300nm_45 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_43 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_43.change_theta_i(43)
s9_300nm_47 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_47.change_theta_i(47)
s9_300nm_52 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_50 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_50.change_theta_i(50)
s9_300nm_54 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_54.change_theta_i(54)
s9_300nm_60 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt")
s9_300nm_58 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt")
s9_300nm_58.change_theta_i(58)
s9_300nm_62 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt") 
s9_300nm_62.change_theta_i(62)
s9_300nm_67 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_65 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_65.change_theta_i(65)
s9_300nm_69 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_69.change_theta_i(69)
s9_300nm_75 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
s9_300nm_73 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
s9_300nm_73.change_theta_i(73)
s9_300nm_77 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
s9_300nm_77.change_theta_i(77)

s9_220nm_30 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_28 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_28.change_theta_i(28)
s9_220nm_32 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_32.change_theta_i(32)
s9_220nm_45 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_43 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_43.change_theta_i(43)
s9_220nm_47 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_47.change_theta_i(47)
s9_220nm_52 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_50 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_50.change_theta_i(50)
s9_220nm_54 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_54.change_theta_i(54)
s9_220nm_60 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt") 
s9_220nm_58 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt")
s9_220nm_58.change_theta_i(58)
s9_220nm_62 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt")
s9_220nm_62.change_theta_i(62)
s9_220nm_67 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_65 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_65.change_theta_i(65)
s9_220nm_69 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_69.change_theta_i(69)
s9_220nm_75 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
s9_220nm_73 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
s9_220nm_73.change_theta_i(74)
s9_220nm_77 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
s9_220nm_77.change_theta_i(76.001)

s9_165nm_30 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_28 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_28.change_theta_i(28) 
s9_165nm_32 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_32.change_theta_i(32) 
s9_165nm_45 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_43 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_43.change_theta_i(43) 
s9_165nm_47 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_47.change_theta_i(47) 
s9_165nm_52 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_50 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_50.change_theta_i(51) 
s9_165nm_54 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_54.change_theta_i(53) 
s9_165nm_60 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_58 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_58.change_theta_i(59) 
s9_165nm_62 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_62.change_theta_i(62) 
s9_165nm_67 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt") 
s9_165nm_65 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt")
s9_165nm_65.change_theta_i(66) 
s9_165nm_69 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt")
s9_165nm_69.change_theta_i(69)
s9_165nm_75 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
s9_165nm_73 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
s9_165nm_73.change_theta_i(74)
s9_165nm_77 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
s9_165nm_77.change_theta_i(77)

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

runs =[s9_lowp_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_lowp_67,s9_lowp_75]#[s9_400nm_30,s9_400nm_45,s9_400nm_52,s9_400nm_60,s9_400nm_67,s9_400nm_75]#[s2_r3_hip_30,s2_r3_hip_45,s2_r3_hip_52,s2_r3_hip_60,s2_r3_hip_67,s2_r3_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_r3_lowp_30,s9_r3_lowp_45,s9_r3_lowp_52,s9_r3_lowp_60,s9_r3_lowp_67,s9_r3_lowp_75]#[s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_bubbles_30,s9_bubbles_45,s9_bubbles_52,s9_bubbles_60,s9_bubbles_67,s9_bubbles_75]#[s9_lowp_30,s9_lowp2_30,s9_medp_30,s9_medp2_30,s9_hip_30,s9_hip2_30]#[s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp2_30,s9_lowp2_45,s9_lowp2_52,s9_lowp2_60,s9_lowp2_67,s9_lowp2_75,s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_lowp_75,s9_medp_75,s9_hip_75]#[s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75,s9_getter_30,s9_getter_45,s9_getter_52,s9_getter_60,s9_getter_67,s9_getter_75]#[s9_first_30,s9_first_45,s9_first_52,s9_first_60,s9_first_67,s9_first_75,s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75]#
angles=[30,45,53,60,68,75]#[67,66,69]#[75,74,76.001]#[30,28,32]#[45,43,47]#[52,51,53]#[60,59,62]#
labels=[30,45,52,60,67,75]
fit_params_indiv=[0.711,2.482,0.134]#[[0.841,1.524,0.142], [0.833,1.57,0.382], [0.765,2.23,0.174], [0.711,2.482,0.134], [0.508,4.342,0.078], [0.167,10.108,0.068]]

#labels=["30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]#
colors=["r", "g", "b", "m", "c", "y"]

# Plot BRIDF data
sample_name="LZ skived"
plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 178 nm", log=True, include_legend=True, errorbars=True, legend_loc=0)
y_diffuse=[]
y_specular=[]
y_total=[]
for ii in range(len(runs)):
	#plt.figure()
	run_data = get_independent_variables_and_relative_intensities(runs[ii])
	# plot_runs(runs[ii], title=sample_name+" in 0.2 barg LXe, {0:g} deg, 178 nm".format(angles[ii]), log=True, include_legend=True, errorbars=True, legend_loc=0)
	#fit_params = fit_parameters(run_data,p0=[1.0, 1.5, .04, 0.1, 1.0],average_angle=4, precision=.25, sigma_theta_i=-1, use_errs=True,use_spike=True, use_nu=False,bounds=([0.1,1.1,0.03,0.01,0.3],[1.6,1.7,0.6,50.,1.7]))
	fit_params = [.183,1.630,.075,.38]#[.8, 1.4, 0.078]
	# fit_params = fit_params_indiv[ii]

	print("Fit parameters (rho_L, n, gamma): "+str(fit_params))

	# Plot BRIDF model from fits
	n_LXe_178 = 1.69
	sigma_theta_i=-1
	precision=.25
	average_angle=4
	# if ii==0: fit_params=[1.528,1.536,.466]
	# else: fit_params=[.988,1.651,.348]

	plot_TSTR_fit(angles[ii], n_LXe_178, fit_params, color=colors[ii], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)

	chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
	print("Chi squared from fit for angle={0} deg: {1:.3f}".format(angles[ii],chi_sq))
	plt.text(0.02,0.92-.04*ii,str(labels[ii]) + r"$^\circ$: $\rho_L$={0:.3f}, n={1:.3f}, $\gamma$={2:.3f}, K={3:.2f}, $\chi^2$={4:.2f}".format(fit_params[0],fit_params[1],fit_params[2],fit_params[3],chi_sq),transform=plt.gca().transAxes,fontsize=11)
	plt.xlim(0,85)
	
	# Total reflectance calcs
	# y_diff=reflectance_diffuse(angles[ii], n_LXe_178, 0.5, fit_params)
	# y_diffuse.append(y_diff)
	# y_spec=reflectance_specular(angles[ii], n_LXe_178, 0.5, fit_params)
	# y_specular.append(y_spec)
	# y_total.append(y_diff+y_spec)

print("Diffuse reflectances: ",y_diffuse)
print("Specular reflectances: ",y_specular)
print("Total reflectances: ",y_total)

y_diffuse_indiv= [0.6677449366590479, 0.6410512423149792, 0.6577679871525651, 0.4441767939777289, 0.28033830767442636, 0.0464811352176091]
y_specular_indiv=[0.0037109289928223543, 0.007237724178743148, 0.038061412541961714, 0.28299705555878757, 0.2684291636169114, 0.45048580194369625]
y_total_indiv=[0.67145587, 0.64828897, 0.6958294,  0.72717385, 0.54876747, 0.49696694]
y_diffuse_simple=[0.72, 0.625, 0.588, 0.508, 0.24, 0.043]#[0.787, 0.757, 0.693, 0.580, 0.283, 0.051] # Exactly equals rho in simple case
y_specular_simple=[0.0006, 0.056, 0.085, 0.129, 0.276, 0.379]#[0.0007194280521610818, 0.02133204118469308, 0.06189473569795954, 0.13173197052597485, 0.27391128183098, 0.3794141055089561]
y_total_simple=[d+s for (d,s) in zip(y_diffuse_simple, y_specular_simple)]#[0.7877194280521611, 0.7783320411846931, 0.7548947356979595, 0.7117319705259748, 0.55691128183098, 0.4304141055089561]
x = [0,10,20,30, 45.1, 55, 60, 65, 70, 75, 80, 85]
y_diffuse_LXe_lowp_corr_nrange_0_15 = [0.6849844088090226, 0.6849516137424515, 0.6848980276446702, 0.6847429162132819, 0.6833829882810474, 0.6775133737054704, 0.6648755402871067, 0.46715038281764626, 0.2314865916574596, 0.10280210546075348, 0.055122569722441166, 0.034902985796637635]
y_specular_LXe_lowp_corr_nrange_0_15 = [0.00146013432719507, 0.001478443859610552, 0.0015779403159794014, 0.0019483216615979846, 0.007657043665555211, 0.03925621759438618, 0.10069087274728164, 0.25351695587136364, 0.4636921738221411, 0.6502541779199938, 0.7858725030849131, 1.0104218377702237]
y_total_LXe_lowp_corr_nrange_0_15 = [sum(x) for x in zip(y_specular_LXe_lowp_corr_nrange_0_15,y_diffuse_LXe_lowp_corr_nrange_0_15)]
y_diffuse=y_diffuse_indiv
y_specular=y_specular_indiv
y_total=y_total_indiv
plt.figure()
# plt.plot(angles, y_diffuse, label="diffuse",linestyle='-',color='c')
# plt.plot(angles, y_specular, label="specular",linestyle='-',color='y')
# plt.plot(angles, y_total, label="total",linestyle='-',color='b')
plt.plot(angles, y_diffuse_indiv, label="diffuse indiv",linestyle='-.',color='c')
plt.plot(angles, y_diffuse_simple, label="diffuse simple",linestyle='--',color='c')
plt.plot(x,y_diffuse_LXe_lowp_corr_nrange_0_15, label="diffuse global",linestyle='-',color='c')
plt.plot(angles, y_specular_indiv, label="specular indiv",linestyle='-.',color='y')
plt.plot(angles, y_specular_simple, label="specular simple",linestyle='--',color='y')
plt.plot(x,y_specular_LXe_lowp_corr_nrange_0_15, label="specular global",linestyle='-',color='y')
plt.plot(angles, y_total_indiv, label="total indiv",linestyle='-.',color='b')
plt.plot(angles, y_total_simple, label="total simple",linestyle='--',color='b')
plt.plot(x,y_total_LXe_lowp_corr_nrange_0_15, label="total global",linestyle='-',color='b')
plt.xlabel("incident angle (degrees)")
plt.ylabel("reflectance (fraction)")
plt.xlim(0,85)
plt.ylim(0,1)
plt.legend()
plt.title("Fitted "+sample_name+" Reflectance, 178 nm")

plt.show()

