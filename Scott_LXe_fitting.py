import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fit_parameters_grid, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, chi_squared
import time
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Plot style settings
# 14 pt font seems to work for default figure size of [6.4, 4.8] (half column width figure)
# saving figure as *.eps makes figure less blurry when zoomed in, but lines become quite thick when zoomed out
matplotlib.rcParams['font.size'] = 14
#matplotlib.rcParams['figure.figsize'] = [8.0, 6.0] # Default is [6.4, 4.8]
#print(matplotlib.rcParams['figure.figsize'])
matplotlib.rcParams['legend.fontsize'] = 'small'
matplotlib.rcParams['figure.autolayout'] = True


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


mirror_path_vac = "Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\"
mirror_data_vac = [
[30., mirror_path_vac + "2019_01_10__16_03_43.txt"],
[45., mirror_path_vac + "2019_01_10__16_01_50.txt"],
[52., mirror_path_vac + "2019_01_10__16_00_07.txt"],
[60., mirror_path_vac + "2019_01_10__15_58_07.txt"],
[67., mirror_path_vac + "2019_01_10__15_56_24.txt"],
[75., mirror_path_vac + "2019_01_10__15_53_05.txt"]
]

# First LXe run
mirror_path_lxe = "First Xe Run Measurements\\Mirror alignment checks 10-29\\"

# must be in ascending angle order
mirror_data_lxe = [
[30., mirror_path_lxe + "2018_10_29__09_33_56.txt"],
[45., mirror_path_lxe + "2018_10_29__09_31_00.txt"],
[52., mirror_path_lxe + "2018_10_29__09_31_00.txt"], # data not taken at this angle, must assume same as 45
[60., mirror_path_lxe + "2018_10_29__09_25_03.txt"],
[67., mirror_path_lxe + "2018_10_29__09_25_03.txt"], # data not taken at this angle, must assume same as 60
[75., mirror_path_lxe + "2018_10_29__09_45_18.txt"]]

s9_path = path_run1 + "Sample 9\\"
s9_r1_30 = Run(s9_path + "2018_11_02__20_38_01.txt", mirror_data_lxe)
s9_r1_45 = Run(s9_path + "2018_11_02__20_32_34.txt", mirror_data_lxe)
s9_r1_52 = Run(s9_path + "2018_11_02__20_27_27.txt", mirror_data_lxe)
s9_r1_60 = Run(s9_path + "2018_11_02__20_22_13.txt", mirror_data_lxe)
s9_r1_67 = Run(s9_path + "2018_11_02__20_17_11.txt", mirror_data_lxe)
s9_r1_75 = Run(s9_path + "2018_11_02__20_11_44.txt", mirror_data_lxe)
angle_shifts_s9_r1=[30, 45, 52, 60, 68, 77]

s3_path = path_run1 + "Sample 3\\"
s3_r1_30 = Run(s3_path + "2018_11_02__19_37_08.txt", mirror_data_lxe)#Run(s3_path + "2018_11_01__18_06_10.txt", mirror_data_lxe)
s3_r1_45 = Run(s3_path + "2018_11_02__19_42_42.txt", mirror_data_lxe)
s3_r1_52 = Run(s3_path + "2018_11_02__19_47_37.txt", mirror_data_lxe)
s3_r1_60 = Run(s3_path + "2018_11_02__19_52_30.txt", mirror_data_lxe)
s3_r1_67 = Run(s3_path + "2018_11_02__19_59_24.txt", mirror_data_lxe)
s3_r1_75 = Run(s3_path + "2018_11_02__20_04_42.txt", mirror_data_lxe)#Run(s3_path + "2018_11_01__17_51_10.txt", mirror_data_lxe)#
angle_shifts_s3_r1=[30, 45, 52, 59, 66, 73]

s1_path = path_run1 + "Sample 1\\"
s1_r1_30 = Run(s1_path + "2018_11_02__18_28_43.txt", mirror_data_lxe)
s1_r1_45 = Run(s1_path + "2018_11_02__18_35_18.txt", mirror_data_lxe)
s1_r1_52 = Run(s1_path + "2018_11_02__18_41_20.txt", mirror_data_lxe)
s1_r1_60 = Run(s1_path + "2018_11_02__18_47_03.txt", mirror_data_lxe)
s1_r1_67 = Run(s1_path + "2018_11_02__18_51_55.txt", mirror_data_lxe)
s1_r1_75 = Run(s1_path + "2018_11_02__18_57_35.txt", mirror_data_lxe)
angle_shifts_s1_r1=[30, 45, 52, 59, 67, 73]

# Second LXe run
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
angle_shifts_s5=[30, 45, 52, 59, 67, 74]

s8_30 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_52_24.txt")
s8_45 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_46_48.txt")
s8_52 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_42_00.txt")
s8_60 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_57_12.txt") 
s8_67 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_01_51.txt")
s8_75 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_06_25.txt") 
angle_shifts_s8=[30, 45, 53, 59, 66.5, 74.5]

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
angle_shifts_s9_bubbles=[30, 45, 52, 59, 67, 75]

s9_nobubbles_30 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_05_41.txt")
s9_nobubbles_45 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_01_01.txt")
s9_nobubbles_52 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_56_06.txt")
s9_nobubbles_60 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_51_00.txt") 
s9_nobubbles_67 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_46_13.txt")
s9_nobubbles_75 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_40_45.txt") 
angle_shifts_s9_nobubbles=[30, 45, 52, 59, 68, 75]

s9_getter_30 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_13_32.txt")
s9_getter_45 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_08_40.txt")
s9_getter_52 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_03_50.txt")
s9_getter_60 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_58_47.txt") 
s9_getter_67 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_53_50.txt")
s9_getter_75 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_48_50.txt") 
angle_shifts_s9_getter=[30, 45, 52, 59, 68, 75]

s9_lowp_30 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_52_04.txt")
s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt")
s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
angle_shifts_s9_lowp=[29, 47, 53, 59, 67, 75]

s9_lowp2_30 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_28_41.txt")
s9_lowp2_45 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_23_51.txt")
s9_lowp2_52 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_18_50.txt")
s9_lowp2_60 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_14_04.txt") 
s9_lowp2_67 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_09_15.txt")
s9_lowp2_75 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_04_19.txt")
angle_shifts_s9_lowp2=[29, 47, 52, 59, 67, 75]

s9_medp_30 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_56_28.txt")
s9_medp_45 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_51_07.txt")
s9_medp_52 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_44_43.txt")
s9_medp_60 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_39_51.txt") 
s9_medp_67 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_35_13.txt")
s9_medp_75 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_29_31.txt") 
angle_shifts_s9_medp=[30, 45, 52, 60, 68, 75]

s9_medp2_30 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_42_49.txt")
s9_medp2_45 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_36_41.txt")
s9_medp2_52 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_31_00.txt")
s9_medp2_60 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_25_38.txt")
s9_medp2_67 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_20_48.txt")
s9_medp2_75 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_14_47.txt")
angle_shifts_s9_medp2=[30, 45, 52, 59, 67, 74]

s9_hip_30 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_33_42.txt")
s9_hip_45 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_28_40.txt")
s9_hip_52 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_19_59.txt")
s9_hip_60 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_15_21.txt") 
s9_hip_67 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_10_17.txt")
s9_hip_75 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_03_53.txt") 
angle_shifts_s9_hip=[30, 45, 52, 60, 67, 75]

s9_hip2_30 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_59_52.txt")
s9_hip2_45 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_55_11.txt")
s9_hip2_52 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_50_31.txt")
s9_hip2_60 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_45_32.txt") 
s9_hip2_67 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_40_40.txt")
s9_hip2_75 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_35_58.txt") 
angle_shifts_s9_hip2=[30, 45, 52, 59, 67, 74]

s9_lowp_above_30 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_37_47.txt")
s9_lowp_above_45 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_32_48.txt")
s9_lowp_above_52 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_27_11.txt")
s9_lowp_above_60 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_22_25.txt") 
s9_lowp_above_67 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_17_42.txt")
s9_lowp_above_75 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_12_10.txt")
angle_shifts_s9_lowp_above=[30, 45, 52, 59, 67, 74]

# Second run, data at different wavelengths
s9_400nm_30 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_12_04.txt")
s9_400nm_45 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_52 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_60 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_67 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_75 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
angle_shifts_s9_400nm=[28, 45, 50, 58, 65, 74]

s9_300nm_30 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_45 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_52 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_60 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt")
s9_300nm_67 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_75 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
angle_shifts_s9_300nm=[30, 45, 52, 60, 65, 77]

s9_220nm_30 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_45 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_52 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_60 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt") 
s9_220nm_67 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_75 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
angle_shifts_s9_220nm=[30, 45, 52, 60, 65, 74]

s9_165nm_30 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_45 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_52 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_60 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_67 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt")
s9_165nm_75 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
angle_shifts_s9_165nm=[28, 44, 50, 58, 66, 74]

# Third LXe run
path_run3 = "3rd Xenon Run Measurements\\" # Any kind of correction for drifting power? Strongest in 3rd run
s9_r3_lowp_30 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__10_53_38.txt")
s9_r3_lowp_45 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__10_58_38.txt")
s9_r3_lowp_52 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_03_46.txt")
s9_r3_lowp_60 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_09_09.txt")
s9_r3_lowp_67 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_14_26.txt")
s9_r3_lowp_75 = Run(path_run3 + "Sample 9, 0.2 barg\\2018_12_20__11_19_23.txt")
angle_shifts_s9_r3_lowp=[30, 45, 52, 61, 69, 77]

s9_r3_hip_30 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_11_07.txt")
s9_r3_hip_45 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_16_35.txt")
s9_r3_hip_52 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_21_41.txt")
s9_r3_hip_60 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_26_59.txt")
s9_r3_hip_67 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_32_16.txt")
s9_r3_hip_75 = Run(path_run3 + "Sample 9, 1.4 barg\\2018_12_20__15_37_45.txt")
angle_shifts_s9_r3_hip=[30, 45, 52, 60, 69, 77]

s6_r3_lowp_30 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_06_37.txt")
s6_r3_lowp_45 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_17_22.txt")
s6_r3_lowp_52 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_22_29.txt")
s6_r3_lowp_60 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_27_34.txt")
s6_r3_lowp_67 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_32_34.txt")
s6_r3_lowp_75 = Run(path_run3 + "Sample 6, 0.2 barg\\2018_12_20__10_37_33.txt")
angle_shifts_s6_r3_lowp=[30, 45, 52, 62, 69, 75]

s6_r3_hip_30 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_27_14.txt")
s6_r3_hip_45 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_32_41.txt")
s6_r3_hip_52 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_38_14.txt")
s6_r3_hip_60 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_44_40.txt")
s6_r3_hip_67 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_50_57.txt")
s6_r3_hip_75 = Run(path_run3 + "Sample 6, 1.4 barg\\2018_12_20__14_56_59.txt")
angle_shifts_s6_r3_hip=[30, 45, 52, 61, 69, 75]

s2_r3_lowp_30 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__15_55_53.txt")
s2_r3_lowp_45 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_04_03.txt")
s2_r3_lowp_52 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_10_48.txt")
s2_r3_lowp_60 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_17_59.txt")
s2_r3_lowp_67 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_26_30.txt")
s2_r3_lowp_75 = Run(path_run3 + "Sample 2, 0.2 barg\\2018_12_19__16_33_07.txt")
angle_shifts_s2_r3_lowp=[30, 45, 52, 61, 69, 75]

s2_r3_lowp_above_30 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_29_53.txt")
s2_r3_lowp_above_45 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_35_01.txt")
s2_r3_lowp_above_52 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_39_48.txt")
s2_r3_lowp_above_60 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_44_38.txt")
s2_r3_lowp_above_67 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_49_26.txt")
s2_r3_lowp_above_75 = Run(path_run3 + "Sample 2, 1-8_ above center, 0.2 barg\\2018_12_20__11_54_08.txt")
angle_shifts_s2_r3_lowp_above=[30, 45, 52, 61, 69, 74]

s2_r3_hip_30 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_33_23.txt")
s2_r3_hip_45 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_40_41.txt")
s2_r3_hip_52 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_48_08.txt")
s2_r3_hip_60 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__18_54_57.txt")
s2_r3_hip_67 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__19_02_21.txt")
s2_r3_hip_75 = Run(path_run3 + "Sample 2, 1.4 barg\\2018_12_19__19_12_01.txt")
angle_shifts_s2_r3_hip=[30, 45, 52, 60, 69, 74]

runs = [s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s8_30, s8_45, s8_52, s8_60, s8_67, s8_75]#[s9_r3_hip_30,s9_r3_hip_45,s9_r3_hip_52,s9_r3_hip_60,s9_r3_hip_67,s9_r3_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s6_r3_hip_30,s6_r3_hip_45,s6_r3_hip_52,s6_r3_hip_60,s6_r3_hip_67,s6_r3_hip_75]#[s9_400nm_30,s9_400nm_45,s9_400nm_52,s9_400nm_60,s9_400nm_67,s9_400nm_75]#[s5_30,s5_45,s5_52,s5_60,s5_67,s5_75]#[s8_30,s8_45,s8_52,s8_60,s8_67,s8_75]#[s9_220nm_30,s9_220nm_45,s9_220nm_52,s9_220nm_60,s9_220nm_67,s9_220nm_75]#[s2_r3_hip_30,s2_r3_hip_45,s2_r3_hip_52,s2_r3_hip_60,s2_r3_hip_67,s2_r3_hip_75]#[s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_bubbles_30,s9_bubbles_45,s9_bubbles_52,s9_bubbles_60,s9_bubbles_67,s9_bubbles_75]#[s9_lowp_30,s9_lowp2_30,s9_medp_30,s9_medp2_30,s9_hip_30,s9_hip2_30]#[s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp2_30,s9_lowp2_45,s9_lowp2_52,s9_lowp2_60,s9_lowp2_67,s9_lowp2_75,s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_lowp_75,s9_medp_75,s9_hip_75]#[s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75,s9_getter_30,s9_getter_45,s9_getter_52,s9_getter_60,s9_getter_67,s9_getter_75]#[s9_first_30,s9_first_45,s9_first_52,s9_first_60,s9_first_67,s9_first_75,s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75]#
angle_shifts=angle_shifts_s8
for run, angle in zip(runs, angle_shifts): run.change_theta_i(angle)
labels=[r"$\theta_i=30^{\circ}$","45$^{\circ}$","52$^{\circ}$","60$^{\circ}$", "67$^{\circ}$", "75$^{\circ}$"]#,"30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="807NX turn"
plot_runs(runs, title=sample_name+", Run 1 in 0.82 barg LXe, 178 nm", log=True, labels=labels, include_legend=True, errorbars=True, legend_loc=0)
# plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 178 nm, 75 deg", log=False, labels=False, include_legend=False, errorbars=True, legend_loc=0)
t0=time.time()

# Fit data
# fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[.93, 1.57, .08, 0.8, 4.0],average_angle=4, precision=.25, sigma_theta_i=2, use_errs=True,use_spike=True, use_nu=True,bounds=([0.1,1.1,0.03,0.01,1.0],[1.6,2.6,0.6,10.,50.]))
# fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[1.1,1.55,.2,5.],average_angle=4, precision=.25, sigma_theta_i=2, use_errs=True,use_spike=False,bounds=([0.7,1.4,0.04,3.],[1.7,1.8,0.3,50.]))#[0.5,1.4,0.04,5.0],[1.2,1.8,0.3,50.]#[0.5,1.4,0.1],[1.0,1.8,0.3]#[0.01,1.01,0.05],[2.0,3.0,0.5]
#fit_params_ang = fit_parameters_and_angle(get_independent_variables_and_relative_intensities(runs),average_angle=4.)
#fit_ang = fit_params_ang[0]
#fit_params = fit_params_ang[1:]
#fit_params= [0.68114103391512837, 1.6259581862678401, 0.11282346636450034, 8.5642437795248174]#[0.947, 1.555, 0.091]#[0.784,1.568,0.144]#[0.86,1.50,0.07]#[0.72,1.45,0.2]#[0.800,1.581,0.157]#
# Fitter params for s9 in vacuum after LXe, 2.15 solid angle
fit_params_vacuum=[0.68114103391512837, 1.6259581862678401, 0.11282346636450034, 8.5642437795248174]
# Fitter parameters for s9 in LXe, 2.15 solid angle factor, Lorentz model
# fit_params_lowp=[0.9466040398353657, 1.5542353537306548, 0.09079476300181971]
# fit_params_hip=[0.9871481402574648, 1.5715034469520426, 0.09586940402324681]
# Fitter parameters for s9 in LXe, 2.00 solid angle factor, BC model
fit_params_lowp=[1.1077461844574992, 1.5547594445581758, 0.10054542669454647]
fit_params_lowp_nrange_0_12=[1.1036187692365016, 1.5556531315657196, 0.10075277940099392]
fit_params_hip2=[1.1999650882272368, 1.5436916548263535, 0.20783614159557665]
fit_params_hip=[1.1525410041990918, 1.5691617861938227, 0.11806736015270886, 11.114872654849078]
fit_params_s8=[1.30037846657779, 1.5481537596293447, 0.060365408007835346, 4.900748469692683]
fit_params_s9_400nm=[0.6916903132118614, 1.7685668238585814, 0.052664172718245646]
fit_params_s9_400nm_nrange_0_15=[0.6414663099143563, 2.0285416875437416, 0.2071409460435925]
# fit_params=[0.5884392151835587, 1.568625421661115, 0.13151331842089983, 2.591901372978062]
# fit_params=fit_params_lowp_nrange_0_12
# fit_params=[0.795, 1.586, 0.106]
# Fitter parameters for s9 in LXe, 2.00 soid angle, BC, power correction 0.753
fit_params_s9_lowp_corr_nrange_0_15=[0.8098340929026058, 1.5651721053377226, 0.1117946993003091]
# Fitter parameters for s9 in LXe, 2.00 soid angle, TR, power correction 0.753
fit_params_s9_lowp_corr_nrange_0_22_TR=[0.7954599045509312, 1.5856455123704216, 0.10588126510921678]
fit_params_s9_lowp_corr_F_sigmoid_nrange_0_22 = [0.7863406181428023, 1.575379194134467, 0.1159731109468702, 0.058296968852107584]
fit_params_s9_lowp_corr_F_sigmoid=[0.821156242689131, 1.5764822034015689, 0.11532489637813395, 0.05942653556417246]
# fit_params=fit_params_s9_lowp_corr_F_sigmoid_nrange_0_22
fit_params_s9_lowp2_corr_nrange_0_15=[0.9001033317655373, 1.5675280902284154, 0.11370369845973141]
# Fitter parameters for s8 in LXe, 2.00 soid angle, BC, power correction 0.753
fit_params_s8_lowp_corr_nrange_0_15=[0.9427456100375428, 1.5664201944697242, 0.08352285569818733, 3.9498103310586443]
fit_params_s8_lowp_corr_nrange_0_15_avg_5deg=[0.9402613669832427, 1.5664520623940574, 0.06270405048987444, 4.025893124659275]
# fit_params=fit_params_s8_lowp_corr_nrange_0_15_avg_5deg
# Fitter parameters for s9 in LXe, 2.00 solid angle, TR, power correction 0.753, with wavelength range for n_LXe_178
fit_params_s9_lowp_corr_n_LXe_wavelength=[0.7759585422536841, 1.5859038386290387, 0.1152682251245288, 5.607571663996526]
# fit_params=[0.7759585422536841, 1.5859038386290387, 0.057, 5.607571663996526] # like s9 but w/ a smoother surface; gamma from fit to molded, unpolished PTFE from Claudio's paper
# Fitter parameters for s9 in LXe, 3rd run, 2.00 solid angle factor, BC model
fit_params_r3_s9_lowp= [1.0216459850600514, 1.5467069707623793, 0.1101614104957136]
# fit_params=fit_params_r3_s9_lowp
# Fitter parameters for s2 in LXe, 3rd run, 2.00 solid angle factor, BC model
fit_params_r3_s2_hip=[1.0720924015263558, 1.5425690292388028, 0.20472829546890117] # error on n from fit is 1.542-1.543; correlation is 2% w/ rho, -48% w/ gamma (gamma and rho are -0.8%, i.e. very uncorrelated); not much difference in profile for 20 pts in rho, gamma vs 10
# Fit parameters using power correction, TR dist, 2.00 solid angle factor, Gaussian distribution in n
fit_params_s9_lowp_gauss_n= [0.7617591236933063, 1.581667834504223, 0.10965539347694214, 1.2535811658618297]
fit_params_s9_lowp2_gauss_n= [0.855702300814262, 1.5823928228219537, 0.1174943642524602, 1.375597480899577]
fit_params_s9_hip_gauss_n= [0.7951797306837846, 1.6003118297015986, 0.1096203149254032, 1.219900829304097]
fit_params_s9_hip2_gauss_n= [0.8226683156062747, 1.5974576635618738, 0.11905775599090765, 1.3981072921827302]
fit_params_s9_medp_gauss_n= [0.8090807492966514, 1.601324395321314, 0.11048727271620085, 1.4948435916434195]
fit_params_s9_medp2_gauss_n= [0.8314357200774386, 1.596290134519927, 0.1133753088729935, 1.390219480578202]
fit_params_s9_bubbles_gauss_n= [0.7969293740674858, 1.573971098600423, 0.12385855127165646, 1.4524567768049776]
fit_params_s9_nobubbles_gauss_n= [0.8296345869549955, 1.5755593111871824, 0.12215387161041028, 1.3373494975065916]
fit_params_s9_getter_gauss_n= [0.8318405791997575, 1.5810629481820473, 0.11767447597690964, 1.3769551555713786]
fit_params_s9_lowp_above_gauss_n= [0.8477739823593516, 1.5815242729575472, 0.11821105282357888, 1.3590210508898195]
fit_params_s9_165nm_gauss_n= [0.217183610197179, 1.9181849064913858, 0.1536796098999006, 5.850296346659101]
fit_params_s9_300nm_gauss_n=  [0.9447069486729521, 1.4374429468203447, 0.1070442097507485, 1.5680079162288003]
fit_params_s9_400nm_gauss_n= [0.6021432485069854, 1.5440956423913457, 0.13516753727577357, 2.6357777432157024]
fit_params_s5_lowp_gauss_n= [0.7676075426046298, 1.5751319649534214, 0.11329673174987098, 1.5458560015702658]
fit_params_s8_lowp_gauss_n= [0.9144536241966592, 1.5793399801859325, 0.08088290783956979, 1.2565135586082907, 4.857675748475832] # Uses both sigma_n and specular spike
fit_params_s1_r1_gauss_n= [0.6280995506491662, 1.57787534395258, 0.14910541148290482, 1.5083045922962428]
fit_params_s9_r1_gauss_n= [0.8279, 1.5623, 0.1202, 1.09]
fit_params_s3_r1_gauss_n= [0.765861613638844, 1.572270174034155, 0.16606130754530377, 1.444391660582311]
fit_params_s2_lowp_r3_gauss_n= [0.717, 1.545, 0.209, 1.62]
fit_params_s2_hip_r3_gauss_n= [0.793, 1.577, 0.196, 1.70]
fit_params_s2_lowp_above_r3_gauss_n= [0.771, 1.545, 0.203, 1.54]
fit_params_s6_lowp_r3_gauss_n= [0.744, 1.572, 0.184, 1.89]
fit_params_s6_hip_r3_gauss_n= [0.733, 1.596, 0.192, 2.02]
fit_params=fit_params_s6_hip_r3_gauss_n
fit_params_s9_lowp_r3_gauss_n= [0.742, 1.568, 0.12, 1.34]
fit_params_s9_hip_r3_gauss_n= [0.748, 1.596, 0.12, 1.71]
# fit_params=[0.94, 1.5665, 0.0627, 0.8, 4.03]
# fit_params = fit_params_s9_lowp2_gauss_n
# Keeping rho fixed doesn't affect profile much; fixing gamma narrows it (but still chi^2/n only changes by ~.02 over nominal error range of +/- 0.001 or so; fit seems to think it's even more constrained than raw chi^2/n says) - this was w/o averaging
# Profile is significantly narrower and minimum is deeper when doing full averaging - still estimate ~.02 change in chi^2 over nominal error range of +/- 0.001 (gamma, rho fixed); probably due to difficulty estimating such small changes
# More points in n and gamma but fewer in rho gave slightly deeper minimum
# For fixed gamma, rho, chi^2 changes by ~1 when index changes by ~0.014
# For fixed rho, floating gamma, chi^2 changes by ~0.5 (+/- 1 sigma) when index changes by ~0.013
# Fit w/ absolute_sigma=False gives about twice the error estimate on n, but still much too small
# fit_params=fit_params_r3_s2_hip
# fit_params=[0.2,1.8,.15,-1,1.0]#[.294, 1.641, .193, -1, -1]
# fit_params= [0.7951797306837846, 1.6003118297015986, 0.1096203149254032, 1.219900829304097] #[0.6896695040686606, 1.568090201084451, 0.11961323746926948, 1.3852522239830944]#[0.7888177381717851, 1.5708700308736958, 0.2015175185631231, 1.5301238872420593]#[0.2379,1.9478,0.1524,6.27]#[0.767996990969903, 1.5857868179846335, 0.11544417251106244, 5.588549722237487]#[0.7182406672950148, 1.5809864646743321, 0.11749180473563485, 5.271502543530771]

# # Fitted parameters for s9 2.44 solid angle factor
# fit_params_lowp=[0.784,1.568,0.144]
# fit_params_lowp2=[0.892,1.563,0.176]
# fit_params_hip=[0.800,1.581,0.157]
# fit_params_hip2=[0.868,1.558,0.225]
# fit_params_lowp_215=[1.011,1.563,0.118] # 2.15 solid angle factor
# fit_params=[0.906644312239346, 1.42, 0.18, 1]
print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))
# phi_r_list = np.linspace(-np.pi,np.pi,10)
# BRIDF_test_list = [BRIDF(52., phi_r, 0., 1.69, 0.5, fit_params) for phi_r in phi_r_list]
# print("BRIDF at normal incidence and viewing: ",BRIDF_test_list)

# Plot BRIDF model from fits
n_LXe_178 = 1.69
n_LXe_220 = 1.5044552
n_LXe_300 = 1.42975267
n_LXe_400 = 1.404459446
n_LXe = n_LXe_178
sigma_theta_i=2
precision=.25
average_angle=4
# plot_TSTR_fit(20., n_LXe_178, fit_params, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(30., n_LXe_178, fit_params, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(45., n_LXe_178, fit_params, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(60, n_LXe_178, fit_params, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(67, n_LXe_178, fit_params, color="black", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(75, n_LXe_178, fit_params, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plt.ylim(1e-3,1e2)
# plt.yscale("log")
# plot_TSTR_fit(angle_shifts[0], n_LXe, fit_params, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(angle_shifts[1], n_LXe, fit_params, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(angle_shifts[2], n_LXe, fit_params, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(angle_shifts[3], n_LXe, fit_params, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(angle_shifts[4], n_LXe, fit_params, color="c", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(angle_shifts[5], n_LXe, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# colors=["r","g","b","m","c","y"]
# phi_r_list=[0,1,2,5,9,15]
# for ii in range(len(phi_r_list)):
    # plot_TSTR_fit(75, n_LXe_178, fit_params, color=colors[ii], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=phi_r_list[ii],label="$\phi_r$={0}".format(phi_r_list[ii]))
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=5)
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=10)

#plt.tight_layout()

t2=time.time()
print("Plotting time: {0}".format(t2-t1))

# Try averaging over multiple indices
# plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 165 nm", log=True, labels=labels, include_legend=True, errorbars=True, legend_loc=0, figure=True)

# for ii in range(len(runs)):
	# run_data=get_independent_variables_and_relative_intensities(runs[ii])
	# chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
	# print("Chi squared from fit, "+labels[ii]+": ",chi_sq)
	
# run_data=get_independent_variables_and_relative_intensities(runs)
# chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# print("Chi squared from fit: ",chi_sq)
#plt.text(0.3,0.9,r"F=1/(1+exp(-x)), $\chi^2$={0:.2f}".format(chi_sq),transform=plt.gca().transAxes,fontsize=13)

# Profiling in 3D is expensive, unnecessary when variables are weakly correlated
# Separate out into blocks correlated by ~>10%; e.g. do combined 2D profile for n, gamma; then separate 1D profile for rho
# To stay near minimum, set center of range to best fit, use an odd number of points
# Start w/ a wide range in profile, few points, then adjust and refine
# grid_results = fit_parameters_grid(get_independent_variables_and_relative_intensities(runs),rho_start=1.072, rho_end=1.072, rho_num=1, n_start=1.542, n_end=1.5432, n_num=15, gamma_start=0.202, gamma_end=0.208, gamma_num=15, plot=True, show=True, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, do_profiles=[False, True, True])

# Now calculate hemispherical reflectance
plt.figure()

# Different incident angles to calculate for
x = [0,10,20,30, 45.1, 55, 60, 65, 70, 75, 80, 85]#[0,30,45.1,60,70,75,80,85]#[5,60,70,80]#[5,30, 45, 60, 70, 75, 80, 85]#[0,10,20,30, 45, 55, 60, 65, 70, 75, 80, 85]#

y_diffuse = [reflectance_diffuse(theta, n_LXe, 0.5, fit_params) for theta in x]
y_specular = [reflectance_specular(theta, n_LXe, 0.5, fit_params) for theta in x]
y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

print("Diffuse reflectances: ",y_diffuse)
print("Specular reflectances: ",y_specular)
print("Total reflectances: ",y_total)

# x_full=x[:]
# x_full.append(90)
# sin_th = np.array([np.sin(xx*np.pi/180.) for xx in x_full])
# d_th = np.array([10,10,10,15,10,5,5,5,5,5,5,5])*np.pi/180.
# y_diffuse_vacuum_after=[0.5578751171383803, 0.5578645143005274, 0.5576939575569828, 0.5568501861704306, 0.551086564170089, 0.538862412662774, 0.5268697521170906, 0.5079125730733413, 0.4778737783402822, 0.4298921339077521, 0.3520995149402873, 0.22288077539943185]
# y_specular_vacuum_after= [0.055008024656618325, 0.05518586897714156, 0.055937258940607706, 0.05802618505411558, 0.06795857669468221, 0.08518472501034707, 0.10042656606592987, 0.1230381756764488, 0.15759182167624392, 0.21483421777180697, 0.32793108506231394, 0.6332518225200205]
# y_specular_vacuum_after=[min(x[0],1-x[1]) for x in zip(y_specular_vacuum_after,y_diffuse_vacuum_after)]
# y_total_vacuum_after = [sum(x) for x in zip(y_specular_vacuum_after,y_diffuse_vacuum_after)]
# y_total_vacuum_after_full = y_total_vacuum_after[:]
# y_total_vacuum_after_full.append(y_total_vacuum_after[-1])
# y_total_vacuum_after_full=np.array(y_total_vacuum_after_full)
# y_sky_vacuum_after_right = np.sum(sin_th[1:]*d_th*y_total_vacuum_after_full[1:])
# y_sky_vacuum_after_left = np.sum(sin_th[:-1]*d_th*y_total_vacuum_after_full[:-1])
# y_sky_vacuum_after_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_vacuum_after_full[:-1]+y_total_vacuum_after_full[1:])/2)
# print("Uniform reflectance, vacuum after, low est: ",y_sky_vacuum_after_left)
# print("Uniform reflectance, vacuum after, high est: ",y_sky_vacuum_after_right)
# print("Uniform reflectance, vacuum after, middle est: ",y_sky_vacuum_after_mid)
# # # y_diffuse_vacuum_before=[0.45766348181118743, 0.4580440040036757, 0.45840578569295093, 0.458373743496906, 0.45519279945081914, 0.4467776090150314, 0.4382352346643636, 0.42459365404619115, 0.4030226445070669, 0.3691595655515466, 0.31672576341572695, 0.23997195829098059]
# # # y_specular_vacuum_before=[0.06255417549996851, 0.06265684345501554, 0.06321515540210665, 0.06512830173742697, 0.07570232176333869, 0.09608785701329016, 0.11458965758798859, 0.14207103191906068, 0.1823613281476766, 0.240653361837574, 0.3259507169947907, 0.4865190393937654]
# # # y_total_vacuum_before = [sum(x) for x in zip(y_specular_vacuum_before,y_diffuse_vacuum_before)]
# # # y_total_vacuum_before_full = y_total_vacuum_before[:]
# # # y_total_vacuum_before_full.append(y_total_vacuum_before[-1])
# # # y_sky_vacuum_before_right = np.sum(sin_th[1:]*d_th*y_total_vacuum_before_full[1:])
# # # y_sky_vacuum_before_left = np.sum(sin_th[:-1]*d_th*y_total_vacuum_before_full[:-1])
# # # print("Uniform reflectance, vacuum before, low est: ",y_sky_vacuum_before_left)
# # # print("Uniform reflectance, vacuum before, high est: ",y_sky_vacuum_before_right)
# # # y_diffuse_LXe_lowp_old=[0.6570412916757591, 0.6579357341415137, 0.6591016792259574, 0.6606138318516783, 0.6633688576680011, 0.6626957378115929, 0.6549616061782121, 0.6062281312066095, 0.0, 0.0, 0.0, 0.0]
# # # y_specular_LXe_lowp_old= [0.0014054544422652119, 0.0014218233642250282, 0.0015141490462703899, 0.0018695965399099326, 0.0057283880366878585, 0.03827531730547219, 0.09984977500055535, 0.25165997130973317, 0.5000189208911733, 0.6960823791225147, 0.8249335365771578, 1.0]#1.0750904931429557]
# # y_diffuse_LXe_lowp_215=[0.7982997601321108, 0.7982673604741001, 0.7982065233759797, 0.7980038464847359, 0.7960858935743564, 0.7874156102421991, 0.7676760751881307, 0.6532634974917723, 0.23652460596145553, 0.13341800718215685, 0.08127525337984297, 0.05402807309618492]
# # y_specular_LXe_lowp_215=[0.0017852149063135958, 0.001818308226244964, 0.0019795301316915003, 0.002543034835696333, 0.009548179713833347, 0.053554449148170015, 0.11833288201197621, 0.2823038620542749, 0.5406700804638772, 0.6790627367772443, 0.798665136503454, 1.0700312085436152]
# y_diffuse_LXe_lowp=[0.9266064823674807, 0.9265704632837276, 0.9265018071134418, 0.9262701814312928, 0.9240684435664849, 0.9141206564704799, 0.8915297546482749, 0.761874690361145, 0.2629238588222386, 0.11379313619522459, 0.06190015347709465, 0.03963860351008269]
# y_specular_LXe_lowp=[0.0017275000631306692, 0.0017476375724292414, 0.0018610656328319315, 0.0022948430948827032, 0.007651243849065006, 0.04394224210952713, 0.11074467133765065, 0.3068205025510641, 0.6153235998043434, 0.7629784298122186, 0.8530040361670844, 1.0528088681336794]
# y_diffuse_LXe_lowp_nrange_0_12 = [0.9241829143644459, 0.9241468889960962, 0.9240787598471459, 0.9238504970970207, 0.9216890931525608, 0.9119557742592572, 0.8899549445268882, 0.5903236321522066, 0.26568393272061674, 0.11483593779759056, 0.06235568461855687, 0.03979878583818896]
# y_specular_LXe_lowp_nrange_0_12 =  [0.0017036874347853338, 0.0017235345857333317, 0.0018352543890812374, 0.002261999394083116, 0.008492343244381738, 0.04342943475762406, 0.1124622878735221, 0.29591521525792086, 0.5473931732412308, 0.7276610603448042, 0.8337474320040098, 1.0344270398761855]
# y_diffuse_LXe_lowp_corr_nrange_0_15 = [0.6849844088090226, 0.6849516137424515, 0.6848980276446702, 0.6847429162132819, 0.6833829882810474, 0.6775133737054704, 0.6648755402871067, 0.46715038281764626, 0.2314865916574596, 0.10280210546075348, 0.055122569722441166, 0.034902985796637635]
# y_specular_LXe_lowp_corr_nrange_0_15 = [0.00146013432719507, 0.001478443859610552, 0.0015779403159794014, 0.0019483216615979846, 0.007657043665555211, 0.03925621759438618, 0.10069087274728164, 0.25351695587136364, 0.4636921738221411, 0.6502541779199938, 0.7858725030849131, 1.0104218377702237]
# y_diffuse_LXe_lowp2_corr_nrange_0_15 = [0.763294497364252, 0.7632567449693723, 0.7631966240341286, 0.7630278394234797, 0.7615748539054874, 0.7553677956586213, 0.7421539873425429, 0.5331105917037516, 0.26671978257084816, 0.1192176914479078, 0.06382869151298011, 0.04017584560718217]
# y_specular_LXe_lowp2_corr_nrange_0_15 = [0.0014030695227905936, 0.0014208328438786982, 0.0015168243547133834, 0.00187241722987367, 0.007290886116810948, 0.03765159072851764, 0.09632609156169926, 0.24320400471201084, 0.44969889213403996, 0.6368825017639688, 0.7766238762554981, 1.0047942687380587]
# y_diffuse_LXe_lowp_corr_F_sigmoid = [0.6939534382770102, 0.6939238542022005, 0.6938770795368621, 0.6937472005284198, 0.6926464622916416, 0.6750525221890046, 0.5398433679810332, 0.423639281548519, 0.2777275475419868, 0.10848915684277789, 0.04495432741868956, 0.021750295553242656]
# y_specular_LXe_lowp_corr_F_sigmoid = [3.724532255638682e-06, 7.962592344563237e-06, 3.854288968971456e-05, 0.0002481320778125987, 0.005359747403441883, 0.040600996135019794, 0.10490145613874569, 0.23739829859639835, 0.43395190559116736, 0.6272414556185343, 0.772839863489491, 0.9787985398543227] 
# y_diffuse_LXe_lowp_corr_F_sigmoid_nrange_0_22 = [0.6708449058354563, 0.6708161979803668, 0.6707705418022764, 0.6706428315090178, 0.6695549644449806, 0.665009113239715, 0.6556506604924496, 0.45666359908482695, 0.26529992916858347, 0.10363290699731809, 0.043241365205486354, 0.021024477224664117]
# y_specular_LXe_lowp_corr_F_sigmoid_nrange_0_22 = [3.299783903254981e-06, 7.176268021824865e-06, 3.562786697695568e-05, 0.00023487077499503855, 0.005239484493297938, 0.040371072575241605, 0.10512079033078614, 0.23924735712969872, 0.43794599363548137, 0.6316843313910302, 0.7764658222206714, 0.9831121033384215]
# y_diffuse_LXe_lowp_corr_F_sigmoid_nrange_0_22_nLXe_1_67 = [0.6866005855775796, 0.6865719537575551, 0.6865311065795917, 0.6864330018318727, 0.6856857892488337, 0.6827696301495876, 0.677201428664633, 0.5267726261950291, 0.35645100783672284, 0.14697864145993247, 0.05800715299364864, 0.02817602566846454]
# y_specular_LXe_lowp_corr_F_sigmoid_nrange_0_22_nLXe_1_67 = [1.9003613529567307e-06, 4.1340936717377195e-06, 2.0555280292206632e-05, 0.00013632168665773326, 0.00317101212825511, 0.026031127272492932, 0.0711227103147388, 0.17402926971950358, 0.3505118117460451, 0.5530134224438349, 0.7199635624509056, 0.935191675171452]
# y_diffuse_LXe_lowp_corr_nrange_0_22_TR = [0.6888046535744143, 0.6887810822030959, 0.6887434427210154, 0.6886379514565755, 0.6877461500892859, 0.6841154973521073, 0.6769332280202829, 0.5024879159050202, 0.3153096259018822, 0.11602918281252782, 0.04438435618733414, 0.02172072577518708]
# y_specular_LXe_lowp_corr_nrange_0_22_TR = [0.0010163140786094772, 0.0010233133613646629, 0.0010714612901757627, 0.0012700767440732093, 0.004740608536223065, 0.028127969121605208, 0.08525664163580114, 0.22473833903720294, 0.40226501948572335, 0.5625467075147237, 0.6969938597899717, 0.8868847523298867]
# y_diffuse_LXe_lowp_corr_nrange_0_22_TR_nLXe_1_67 = [0.7043527637657724, 0.7043295036319436, 0.7042964672828244, 0.7042178439638453, 0.7036271863594424, 0.7013858638992404, 0.6972513554518018, 0.574597223946129, 0.3998753310021367, 0.1813278111394726, 0.06572385974915745, 0.028319841234986508]
# y_specular_LXe_lowp_corr_nrange_0_22_TR_nLXe_1_67 =  [0.0006719511944173549, 0.0006763533974343866, 0.0007068175232318622, 0.0008311971136526351, 0.0028821786707109685, 0.018049416119748193, 0.0560793218882344, 0.16542639451625527, 0.327520528510148, 0.4812689236581855, 0.6114193784905372, 0.7894422181831445]
# y_diffuse_LXe_lowp_corr_n_LXe_wavelength_range=[0.67007, 0.67004, 0.66999, 0.66984, 0.66843, 0.66106, 0.62666, 0.50303, 0.30878, 0.12308, 0.04947, 0.02462]
# y_specular_LXe_lowp_corr_n_LXe_wavelength_range=[0.0014188411412140378, 0.001431907642593201, 0.0015159581374150227, 0.0018842387090620644, 0.007265352611230925, 0.03963603057177141, 0.10358179399254425, 0.2400332573988373, 0.4429704818620232, 0.6440315962968354, 0.7935845190765715, 0.9996177076083168]
# y_diffuse_LXe_lowp_corr_n_LXe_wavelength=[0.613641223404187, 0.6133760975826529, 0.6130072353223912, 0.6120905222102035, 0.5930822105735014, 0.4798123971363298, 0.3984079667203065, 0.3439863866365737, 0.28251501193396583, 0.1123977797913826, 0.04498752209911323, 0.022105545897536955]
# y_specular_LXe_lowp_corr_n_LXe_wavelength=[0.006538644846720643, 0.006694815353937048, 0.007599845005755459, 0.011437422485756712, 0.061801746057328355, 0.20095292860703112, 0.29259198211564524, 0.3815491648950294, 0.4605211514277208, 0.5297966384836602, 0.6040879430392835, 0.7825422605261033]
# y_diffuse_LXe_lowp_corr_n_LXe_wavelength_gamma_057=[0.6178564448169105, 0.617774888854572, 0.6176007243576107, 0.6169014328824324, 0.5981784147199931, 0.48428646260784003, 0.40233200603920455, 0.3476197886402647, 0.27739471720902237, 0.048007736555920055, 0.013753076625775759, 0.006135032483746843]
# y_specular_LXe_lowp_corr_n_LXe_wavelength_gamma_057=[0.006483353492754421, 0.006525102014159482, 0.006892874895349075, 0.0087460628885176, 0.048547658277852845, 0.21282473167883556, 0.32074377688205963, 0.42170482718133634, 0.5077687208330124, 0.5836028556718805, 0.6410270260476396, 0.7226977011508462]
# y_diffuse=y_diffuse_LXe_lowp_corr_n_LXe_wavelength
# y_specular=y_specular_LXe_lowp_corr_n_LXe_wavelength
# y_total=[d+s for (d,s) in zip(y_diffuse,y_specular)]
# y_diffuse_LXe_lowp_corr_n_LXe_wavelength_range_high_p_Sellmeier=[0.6715134167065824, 0.6714853609714969, 0.6714421935374175, 0.6713127787644213, 0.6701439586550816, 0.6643050005459191, 0.6409912810309316, 0.5423901276175388, 0.3093725442548185, 0.12331615489025698, 0.04956517979944786, 0.024666187267823915]
# y_specular_LXe_lowp_corr_n_LXe_wavelength_range_high_p_Sellmeier=[0.0011803678022674808, 0.0011910321514857327, 0.0012596389942219881, 0.0015557156196008767, 0.005811796424766054, 0.031535120778654335, 0.08341341643011518, 0.1984864758988867, 0.38032487887665684, 0.5776807997683754, 0.7396554874419252, 0.9542941981495459]
# y_specular_LXe_lowp=[min(x[0],1-x[1]) for x in zip(y_specular_LXe_lowp,y_diffuse_LXe_lowp)]
# y_total_LXe_lowp = [sum(x) for x in zip(y_specular_LXe_lowp,y_diffuse_LXe_lowp)]
# y_total_LXe_lowp_full = y_total_LXe_lowp[:]
# y_total_LXe_lowp_full.append(y_total_LXe_lowp[-1])
# y_total_LXe_lowp_full=np.array(y_total_LXe_lowp_full)
# y_sky_LXe_lowp_right = np.sum(sin_th[1:]*d_th*y_total_LXe_lowp_full[1:])
# y_sky_LXe_lowp_left = np.sum(sin_th[:-1]*d_th*y_total_LXe_lowp_full[:-1])
# y_sky_LXe_lowp_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_LXe_lowp_full[:-1]+y_total_LXe_lowp_full[1:])/2)
# print("Uniform reflectance, LXe 0.2 barg, low est: ",y_sky_LXe_lowp_left)
# print("Uniform reflectance, LXe 0.2 barg, high est: ",y_sky_LXe_lowp_right)
# print("Uniform reflectance, LXe 0.2 barg, middle est: ",y_sky_LXe_lowp_mid)
# # # y_diffuse_LXe_lowp2= [0.7402398286291142, 0.7417007202832876, 0.7436238363017107, 0.7461638687993092, 0.7512130151093066, 0.752163989934962, 0.7433876636862394, 0.6765365007371161, 0.0, 0.0, 0.0, 0.0]
# # # y_specular_LXe_lowp2= [0.001527762931549758, 0.0015516293860655497, 0.0016747374647144962, 0.0021276141278918324, 0.007227369888786746, 0.04961757416725003, 0.11917259018207207, 0.2626128326744476, 0.4667668545228056, 0.6430628471896185, 0.7907470379047791, 1.0]#1.0997835008723356]
# # # y_total_LXe_lowp2 = [sum(x) for x in zip(y_specular_LXe_lowp2,y_diffuse_LXe_lowp2)]
# # # y_diffuse_LXe_hip_old= [0.6829307222870662, 0.6840336258528498, 0.6854793518118913, 0.6873917251376706, 0.6913214705135405, 0.6930010462636261, 0.6892947926267271, 0.6618038165708974, 0.0, 0.0, 0.0, 0.0]
# # # y_specular_LXe_hip_old= [0.001112165427732242, 0.0011263277428219362, 0.00120284458242256, 0.0014878128453774913, 0.00431818948159568, 0.028577408166662696, 0.0767039434227785, 0.1938216373997157, 0.40704522413508704, 0.6172302526293288, 0.7723284466198603, 1.0]#1.047912122232299]
# y_diffuse_LXe_hip= [0.8491875929406879, 0.8491509762266888, 0.8490914075177584, 0.8489201262096208, 0.8474340080642626, 0.8411375534056131, 0.8279556472564547, 0.7687016503469466, 0.3323357823672721, 0.16558390232400014, 0.10476839339443421, 0.06534115759627902]
# y_specular_LXe_hip= [0.0013400522898790678, 0.0013641072780542977, 0.001480245806814926, 0.0018772752465937084, 0.005731975904048718, 0.036692820567370533, 0.0848219866352421, 0.2022403891367498, 0.4541913398256783, 0.6259906120320465, 0.7543231293523422, 1.0211097366859878]
# y_specular_LXe_hip=[min(x[0],1-x[1]) for x in zip(y_specular_LXe_hip,y_diffuse_LXe_hip)]
# y_total_LXe_hip = [sum(x) for x in zip(y_specular_LXe_hip,y_diffuse_LXe_hip)]
# y_total_LXe_hip_full = y_total_LXe_hip[:]
# y_total_LXe_hip_full.append(y_total_LXe_hip[-1])
# y_total_LXe_hip_full=np.array(y_total_LXe_hip_full)
# y_sky_LXe_hip_right = np.sum(sin_th[1:]*d_th*y_total_LXe_hip_full[1:])
# y_sky_LXe_hip_left = np.sum(sin_th[:-1]*d_th*y_total_LXe_hip_full[:-1])
# y_sky_LXe_hip_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_LXe_hip_full[:-1]+y_total_LXe_hip_full[1:])/2)
# print("Uniform reflectance, LXe 1.45 barg, low est: ",y_sky_LXe_hip_left)
# print("Uniform reflectance, LXe 1.45 barg, high est: ",y_sky_LXe_hip_right)
# print("Uniform reflectance, LXe 1.45 barg, middle est: ",y_sky_LXe_hip_mid)
# # # y_diffuse_LXe_hip2= [0.7116527697234036, 0.7138323922133201, 0.7167431409327432, 0.7206668394297263, 0.729091743241813, 0.7335885883463384, 0.7262823707645063, 0.6474078491699706, 0.0, 0.0, 0.0, 0.0]
# # # y_specular_LXe_hip2=  [0.0016528720786998794, 0.0016884735251564945, 0.001857732976822431, 0.0024483848517261844, 0.009600045662265523, 0.06251613539473333, 0.13418304588129548, 0.25679333990263276, 0.4159533052261603, 0.5728408076108408, 0.7414240888009019, 1.0]#1.1194372218351034]
# # # y_total_LXe_hip2 = [sum(x) for x in zip(y_specular_LXe_hip2,y_diffuse_LXe_hip2)]
# y_diffuse_LXe_s8_lowp_corr_nrange_0_15_avg_5deg = [0.8017796267007459, 0.8017684625339829, 0.8017352112065645, 0.8015897500538604, 0.8001009901884555, 0.7934880233354094, 0.7792406093909442, 0.5542014870444953, 0.2257014237606343, 0.07370467156411019, 0.03699734732934129, 0.022905381919653186]
# y_specular_LXe_s8_lowp_corr_nrange_0_15_avg_5deg = [0.00142773600440886, 0.0014368159853060235, 0.001501487718275121, 0.0017686947070900182, 0.005310897952563967, 0.02417679572170201, 0.06737109246784116, 0.26669457614261605, 0.5622565793863318, 0.8284511674732585, 0.9687327632248928, 1.1205957885096496]
# y_diffuse_LXe_s9_lowp_gauss_n = [0.6555220900082763, 0.6552533863910289, 0.6549497225806359, 0.6545160708355516, 0.6526151469567986, 0.6455809912310351, 0.6253474225536518, 0.4828032742121666, 0.27398385756675303, 0.10555036651013987, 0.04161843185672369, 0.019423169473866696]
# y_specular_LXe_s9_lowp_gauss_n = [0.0013792301879329701, 0.0013907277886691955, 0.0014670816824138941, 0.0017944406223630114, 0.006597003741463414, 0.036212891671076934, 0.09897912422754776, 0.24429904591916568, 0.4497687829214006, 0.633264713105453, 0.7660458024612156, 0.9554088934644756]
# uniform_diffuse_LXe_s9_lowp_gauss_n = 0.425
# uniform_specular_LXe_s9_lowp_gauss_n = 0.313
# y_diffuse_LXe_s9_lowp2_gauss_n = [0.7358121411941425, 0.7354717000189173, 0.7350892288906885, 0.7345510068201263, 0.7322553646452498, 0.7236224620153533, 0.6926155639821712, 0.535340266710626, 0.32481103564292974, 0.1284128365044831, 0.05252421857299336, 0.024782814789651997]
# y_specular_LXe_s9_lowp2_gauss_n = [0.0014180134289509612, 0.0014314470658129146, 0.0015166934309719376, 0.0018806710550431024, 0.007408328522106007, 0.040449495530537415, 0.10623147609514205, 0.2456214682102545, 0.4334522033382075, 0.6035483434064692, 0.7356425276348293, 0.9374893634748291]
# uniform_diffuse_LXe_s9_lowp2_gauss_n = 0.478
# uniform_specular_LXe_s9_lowp2_gauss_n = 0.305
# y_diffuse_LXe_s9_hip_gauss_n = [0.6997660324974849, 0.699468492582338, 0.6991385934106876, 0.6986962550924479, 0.6970473871690566, 0.6918691585634782, 0.679931560838754, 0.585859887898757, 0.36726620668126864, 0.1678183259387764, 0.06268718805733493, 0.027468700215127245]
# y_specular_LXe_s9_hip_gauss_n = [0.0010104976372926186, 0.0010185433932873192, 0.0010721098923197083, 0.0012975402697268724, 0.004343132659857975, 0.02370073533346418, 0.06594756436865275, 0.1761636249875636, 0.3597823212436191, 0.5487368333198612, 0.696267635071545, 0.8893647485821567]
# uniform_diffuse_LXe_s9_hip_gauss_n = 0.473
# uniform_specular_LXe_s9_hip_gauss_n = 0.274
# y_diffuse_LXe_s9_hip2_gauss_n = [0.7192584088452663, 0.7189080582201859, 0.7185196117868319, 0.7179983757520753, 0.7160336381199532, 0.7095588888655432, 0.6921125839963785, 0.5718433631734015, 0.36292000604668195, 0.17269547861169557, 0.06558144320138619, 0.031142242204475163]
# y_specular_LXe_s9_hip2_gauss_n = [0.001139041409169669, 0.0011497395923636816, 0.0012170884689858882, 0.0014992947904140163, 0.0056199040726042475, 0.030680521285571003, 0.08146253853020188, 0.19632676330729687, 0.36580359121885264, 0.5337975644634694, 0.6726149187503581, 0.8762276535074885]
# uniform_diffuse_LXe_s9_hip2_gauss_n = 0.483
# uniform_specular_LXe_s9_hip2_gauss_n = 0.273
# y_diffuse_LXe_s9_medp_gauss_n = [0.7111455055614035, 0.7109239774678702, 0.7104974417043892, 0.710116308761732, 0.7081960419039595, 0.701876941050101, 0.6835848072366963, 0.5674177983302076, 0.37632867949797566, 0.17503355357972794, 0.0655659670675892, 0.028728295396723908]
# y_specular_LXe_s9_medp_gauss_n = [0.0011194484849399544, 0.0011288358004272973, 0.0011908998853848666, 0.0014565898559486992, 0.005328450007349814, 0.02902369857698539, 0.07895150673064644, 0.1936315526016375, 0.3618291998482997, 0.5273956250612164, 0.6613583327474009, 0.8466805544230485]
# uniform_diffuse_LXe_s9_medp_gauss_n = 0.479
# uniform_specular_LXe_s9_medp_gauss_n = 0.267
# y_diffuse_LXe_s9_medp2_gauss_n = [0.727077260506764, 0.7267523935322792, 0.7263906477622871, 0.7258976202033405, 0.7239634010695039, 0.7173837912533303, 0.6993929946515262, 0.5758076031944919, 0.35480574111059776, 0.16515741466654474, 0.060754628653960634, 0.028624585539639705]
# y_specular_LXe_s9_medp2_gauss_n = [0.0011558658724239324, 0.0011659241903302653, 0.0012312692110772662, 0.001508321708507372, 0.005528864756189555, 0.030208015838346196, 0.08171201230731272, 0.20103622748012784, 0.37750706868848, 0.5497116194366362, 0.6874543341448208, 0.8811169490444379]
# uniform_diffuse_LXe_s9_medp2_gauss_n = 0.485
# uniform_specular_LXe_s9_medp2_gauss_n = 0.278
# y_diffuse_LXe_s9_bubbles_gauss_n = [0.6766248176951102, 0.6762877506405857, 0.6759073068911247, 0.6753628195291412, 0.6729283524613883, 0.6631603595858275, 0.6198744490449015, 0.45415880574588535, 0.2638068471614982, 0.10861893537589697, 0.04672679359806711, 0.022770836942781213]
# y_specular_LXe_s9_bubbles_gauss_n = [0.0016375113759560975, 0.0016548889424338727, 0.0017618330824682234, 0.002225939177427519, 0.009510874010122091, 0.05123909181313545, 0.12854072547844822, 0.27605885853497253, 0.4592853282200319, 0.6193275886226043, 0.7465095771680348, 0.9589620274010656]
# uniform_diffuse_LXe_s9_bubbles_gauss_n = 0.432
# uniform_specular_LXe_s9_bubbles_gauss_n = 0.318
# y_diffuse_LXe_s9_nobubbles_gauss_n = [0.7069026766517852, 0.7065573602335936, 0.7061686178051533, 0.7056175595779327, 0.7032221331832219, 0.694089759526603, 0.6604044885564836, 0.48138970330161285, 0.28029674557590334, 0.11463307902497671, 0.04882115444134056, 0.02362416177404641]
# y_specular_LXe_s9_nobubbles_gauss_n = [0.001548263691114051, 0.0015639985630463563, 0.0016615234242418379, 0.0020768700789602227, 0.008498992051394531, 0.0461905755451267, 0.11866007764388116, 0.2660221504753117, 0.4574996619390656, 0.6262062891620022, 0.7568856638119414, 0.9694159083538482]
# uniform_diffuse_LXe_s9_nobubbles_gauss_n = 0.453
# uniform_specular_LXe_s9_nobubbles_gauss_n = 0.317
# y_diffuse_LXe_s9_getter_gauss_n = [0.7138333806264641, 0.7135032674984393, 0.7131319655170606, 0.7126073033483711, 0.7103466262168892, 0.7018152952755305, 0.6707035994690923, 0.5141474075025859, 0.29801507874801275, 0.12197137747336179, 0.050109647372608944, 0.023688452253717907]
# y_specular_LXe_s9_getter_gauss_n = [0.0014465174541999927, 0.001460297733980103, 0.0015476605276251317, 0.0019213610980000435, 0.007618870704881095, 0.041578335575947215, 0.1089219439007581, 0.25033734568976856, 0.43906659777709334, 0.6086900041148855, 0.7399404771243379, 0.9417325535440622]
# uniform_diffuse_LXe_s9_getter_gauss_n = 0.462
# uniform_specular_LXe_s9_getter_gauss_n = 0.308
# y_diffuse_LXe_s9_lowp_above_gauss_n = [0.7279869914120184, 0.7276472975357132, 0.7272656927256014, 0.7267288321581082, 0.7244382291152476, 0.7159304051596127, 0.6852402394907756, 0.5280394190726628, 0.30534989098604237, 0.12599890318602708, 0.051747334966352594, 0.02448842302901786]
# y_specular_LXe_s9_lowp_above_gauss_n = [0.0014286372489337761, 0.001442292951034309, 0.0015286189828606603, 0.0018964467522580913, 0.007485922243414949, 0.04086753738264991, 0.10713680956588581, 0.247472451888785, 0.4364786855369349, 0.6072432578873403, 0.7395082885064886, 0.9429141711577821]
# uniform_diffuse_LXe_s9_lowp_above_gauss_n = 0.472
# uniform_specular_LXe_s9_lowp_above_gauss_n = 0.307
# y_diffuse_LXe_s5_lowp_gauss_n = [0.6540522751694569, 0.6537722148241993, 0.6534527764580352, 0.6529791704573941, 0.6506935756565445, 0.640897584048703, 0.5958583040682076, 0.43864936853077857, 0.2554823497948873, 0.0972302607251807, 0.039935228797638805, 0.01912639531153884]
# y_specular_LXe_s5_lowp_gauss_n = [0.0016573177548896492, 0.00167279145147488, 0.0017733137462495009, 0.002223475666193368, 0.009207893728737525, 0.05031016449621828, 0.13001319213000145, 0.2823791837094827, 0.46806249359519253, 0.6267500871124593, 0.7478036075306166, 0.9398916521009449]
# uniform_diffuse_LXe_s5_lowp_gauss_n = 0.415
# uniform_specular_LXe_s5_lowp_gauss_n = 0.317
# y_diffuse_LXe_s8_lowp_gauss_n = [0.7884466683816694, 0.788256353418893, 0.7880321186735665, 0.7876662132784379, 0.7856335665490475, 0.7770829665973333, 0.7507743533210781, 0.5745027731544261, 0.3115078692200302, 0.08448520812234325, 0.02924765179180736, 0.013162071720879362]
# y_specular_LXe_s8_lowp_gauss_n = [0.0014238945753465631, 0.001430458451793799, 0.0014888550616363077, 0.001759214904748927, 0.005393303821050943, 0.027585126391612244, 0.08267691197508208, 0.23013986297620168, 0.5896515389981184, 0.7880207287155164, 0.9326905430583597, 1.1146922646663053]
# y_specular_lobe_only_LXe_s8_lowp_gauss_n = [0.0014149943798950461, 0.001420865957344394, 0.001476681561927991, 0.0017398773265264382, 0.005305921143541413, 0.02704503980857446, 0.08081099020737664, 0.2207991308429237, 0.39978292584086694, 0.5035881396097269, 0.5024997833416238, 0.45985880641307364]
# uniform_diffuse_LXe_s8_lowp_gauss_n = 0.502
# uniform_specular_LXe_s8_lowp_gauss_n = 0.369
# uniform_specular_lobe_only_LXe_s8_lowp_gauss_n = 0.206
# y_diffuse_LXe_s1_r1_lowp_gauss_n = [0.5328103438327415, 0.5324417068220226, 0.5320329167651725, 0.5314832260378756, 0.5293640447155185, 0.5216726873917915, 0.48830482122497315, 0.3619982978664425, 0.22025890364168432, 0.10708063776456868, 0.050342737521486165, 0.02528326366325062]
# y_specular_LXe_s1_r1_lowp_gauss_n = [0.0015798923359595745, 0.0016020251407323995, 0.001726123116792811, 0.0022479575421026654, 0.010590541004287253, 0.05440378215151557, 0.1254965789540431, 0.25132034047071644, 0.40842817511233986, 0.5565288305860067, 0.6922828776961385, 0.9454972155647411]
# uniform_diffuse_LXe_s1_r1_lowp_gauss_n = 0.345
# uniform_specular_LXe_s1_r1_lowp_gauss_n = 0.300
# y_diffuse_LXe_s9_r1_lowp_gauss_n = [0.6939711305393514, 0.6936496124680243, 0.6932837255956826, 0.6927461328632194, 0.6902360220011244, 0.6804124529546723, 0.6495777243849579, 0.44502390279759557, 0.22290194705693034, 0.08617897329446969, 0.03762872509697619, 0.019791038641742774]
# y_specular_LXe_s9_r1_lowp_gauss_n = [0.0017640220643999603, 0.0017813895029294982, 0.0018900001732070518, 0.00234978604005238, 0.009439270675937292, 0.05177711811019578, 0.13503918297823533, 0.3101610663107208, 0.5320642979294079, 0.7083592535076338, 0.8306061207583993, 1.0387754744298883]
# uniform_diffuse_LXe_s9_r1_lowp_gauss_n = 0.435
# uniform_specular_LXe_s9_r1_lowp_gauss_n = 0.352
# y_diffuse_LXe_s3_r1_lowp_gauss_n = [0.6424853198092165, 0.6419553498902326, 0.641370268351101, 0.6405964055850856, 0.6377431003231158, 0.6277572830747802, 0.5853231150326986, 0.4255604071760017, 0.25662195257316167, 0.1318573446791808, 0.06561434846148195, 0.03405380848105668]
# y_specular_LXe_s3_r1_lowp_gauss_n = [0.0016746618787116545, 0.0017020427651980722, 0.0018489080769098437, 0.002455412392582395, 0.012262590656869946, 0.06081926513775412, 0.13375954378354243, 0.25699356922733113, 0.4087716267506628, 0.5543622655042861, 0.6961321093840429, 0.9776886585996237]
# uniform_diffuse_LXe_s3_r1_lowp_gauss_n = 0.415
# uniform_specular_LXe_s3_r1_lowp_gauss_n = 0.307
# y_diffuse_LXe_s2_r3_lowp_gauss_n = [0.5747522618505175, 0.5740777583543583, 0.5733308820968189, 0.5723289983657194, 0.5683900467424476, 0.5510832842263745, 0.4579725258040034, 0.2899280387561567, 0.17983498206954104, 0.10432207218488321, 0.05816476034451382, 0.03256220953891131]
# y_specular_LXe_s2_r3_lowp_gauss_n = [0.002478744689072998, 0.0025391113838572573, 0.00284677594403214, 0.0042961430056456034, 0.02545084157295724, 0.10293412580791013, 0.19078030527284803, 0.3113892084588939, 0.4439915192293557, 0.573626721355484, 0.7235457530082064, 1.072002029189436]
# uniform_diffuse_LXe_s2_r3_lowp_gauss_n = 0.353
# uniform_specular_LXe_s2_r3_lowp_gauss_n = 0.341
# y_diffuse_LXe_s2_r3_hip_gauss_n = [0.6618001579111336, 0.6610675025942065, 0.660265450039994, 0.6592377473193664, 0.6557651312612194, 0.6439143015580011, 0.5874355058686932, 0.4379201535304581, 0.2770344169939162, 0.1595690084196591, 0.08643772299742744, 0.046378706936399325]
# y_specular_LXe_s2_r3_hip_gauss_n = [0.0017039989989366747, 0.0017398277132652997, 0.0019227018083998347, 0.002704128178634338, 0.014945498680997116, 0.06663983744021144, 0.13315134496757355, 0.23495901074337683, 0.35789569952954586, 0.48404245936427237, 0.6259291748423287, 0.9292131694095614]
# uniform_diffuse_LXe_s2_r3_hip_gauss_n = 0.432
# uniform_specular_LXe_s2_r3_hip_gauss_n = 0.283
# y_diffuse_LXe_s2_lowp_above_r3_gauss_n = [0.6192351567084332, 0.6185444221482798, 0.6177787181043282, 0.6167477857969319, 0.6126688872102739, 0.5952796152812168, 0.5071174146203029, 0.3123489654244317, 0.19193891787689168, 0.10958198485070754, 0.06041878074685584, 0.03365441205188768]
# y_specular_LXe_s2_lowp_above_r3_gauss_n = [0.0024363254289879533, 0.0024927325278871495, 0.002781087628643878, 0.004105286138365415, 0.02409365939833029, 0.10047165848168658, 0.19010973621651794, 0.3155209898687198, 0.45361238450633456, 0.5863366929767548, 0.7356668558607419, 1.0803030289592912]
# uniform_diffuse_LXe_s2_lowp_above_r3_gauss_n = 0.380
# uniform_specular_LXe_s2_lowp_above_r3_gauss_n = 0.345
# y_diffuse_LXe_s6_lowp_r3_gauss_n = [0.6183400728981118, 0.6177331323906701, 0.617063095515402, 0.6161754578445222, 0.6128785209344368, 0.5989176706177282, 0.5278504850714418, 0.38071747191881944, 0.24945464569652384, 0.1295592547383651, 0.07157134910257285, 0.03804795338613029]
# y_specular_LXe_s6_lowp_r3_gauss_n = [0.0019314962073781262, 0.001970602667470927, 0.0021765599600805426, 0.003137936895668712, 0.017441635588805952, 0.07669558865029248, 0.15136042899240945, 0.2606728814982911, 0.3860480958632102, 0.5092560799959311, 0.643386925401161, 0.9335492865418483]
# uniform_diffuse_LXe_s6_lowp_r3_gauss_n = 0.395
# uniform_specular_LXe_s6_lowp_r3_gauss_n = 0.295
# y_diffuse=y_diffuse_LXe_s8_lowp_gauss_n
# y_specular=y_specular_LXe_s8_lowp_gauss_n
# y_total=[d+s for (d,s) in zip(y_diffuse,y_specular)]
# y_diffuse_LXe_s9_220nm_gauss_n = [0.6763307923931233, 0.6762449434516675, 0.6761179939759093, 0.6759732667689763, 0.6753831412425655, 0.673620218027684, 0.6705089231574517, 0.6595348271755659, 0.5410351430121954, 0.2383361549302746, 0.06530737915836278, 0.024805767926769112]
# y_specular_LXe_s9_220nm_gauss_n =  [0.0004943753861428848, 0.000496536420344527, 0.0005151252505833433, 0.0005960907106902293, 0.0013978862432142394, 0.005933705323438009, 0.01737188902535051, 0.06031010936585616, 0.2184359385945738, 0.5149422678017304, 0.7485215542823045, 0.9131231783901029]
# y_diffuse_LXe_s9_165nm_gauss_n = [0.18206, 0.18206, 0.18205, 0.18197, 0.18098, 0.16867, 0.15384, 0.13448, 0.11557, 0.10150, 0.08278, 0.06219]
# y_specular_LXe_s9_165nm_gauss_n = [0.00493, 0.00499, 0.00535, 0.00702, 0.02446, 0.08344, 0.13736, 0.20097, 0.26847, 0.33816, 0.42157, 0.60471]
# y_diffuse_LXe_s9_300nm_gauss_n = [0.8782708022130876, 0.8782702930432829, 0.8782615910440826, 0.8782125329077218, 0.8777346654362405, 0.8759512156943553, 0.8726861311440243, 0.8523284130281035, 0.7749585827895182, 0.6528412096722851, 0.535399748515495, 0.3906307039334239]
# y_specular_LXe_s9_300nm_gauss_n = [0.0006217396835632567, 0.0006249432583414607, 0.0006478931292022323, 0.0007391151219186492, 0.0015995109518875, 0.005970190930095282, 0.015058264824990838, 0.04194875746810962, 0.1017582157428121, 0.18969929984141778, 0.2915351286065927, 0.43173298081513317]
# y_diffuse_LXe_s9_400nm_gauss_n = [0.5309231665435903, 0.5305400146393436, 0.5301096087041035, 0.5295196681221188, 0.5275868022833434, 0.5238995280095862, 0.5198843508809361, 0.5122494398796086, 0.48959972283957354, 0.4478319403920365, 0.38332953099647304, 0.25468316773928806]
# y_specular_LXe_s9_400nm_gauss_n = [0.004165797736000184, 0.004182517617872596, 0.004291566382412554, 0.004667317220860221, 0.007058934876843024, 0.013029641816818979, 0.02032056588648655, 0.03476831533741435, 0.06259677066645455, 0.10749236843594039, 0.17575867773952733, 0.30697067886350216]

# Make linear interpolation functions for diffuse, specular reflectivity vs theta_i
th_list=np.array(x)*np.pi/180 # x is in degrees; want in radians
f_diff = interp1d(th_list, y_diffuse)
f_spec = interp1d(th_list, y_specular)

# Make a version that will extrapolate beyond the measured range (0-85 deg)
def f_ext(x_list, interp_func):
    
	def pointwise(x_):
		if x_<0: return interp_func(0)
		if x_>85*np.pi/180: return interp_func(85*np.pi/180)
		else: return interp_func(x_)
        
	if np.size(x_list)<2: return pointwise(x_list)
    
	return np.array([pointwise(x) for x in x_list])

# Function to integrate over; assumes uniform illumination, i.e. sin(theta) solid angle factor (no phi dependence)
def integral_unif_func(x_, interp_func):
	return np.sin(x_)*f_ext(x_, interp_func)
	
# White sky reflectance; gives an extra weighting factor of cos(theta) (then must multiply by 2 for normalization)
# This is uniform in "projected solid angle"
def integral_white_sky_func(x_, interp_func):
	return 2*np.cos(x_)*np.sin(x_)*f_ext(x_, interp_func)
	
uniform_diff = quad(integral_unif_func, 0, np.pi/2, f_diff)[0]
uniform_spec = quad(integral_unif_func, 0, np.pi/2, f_spec)[0]
print("Uniform diffuse reflectance: {0:.3f}, specular: {1:.3f}, total: {2:.3f}".format(uniform_diff, uniform_spec, uniform_diff+uniform_spec))
	
white_sky_diff = quad(integral_white_sky_func, 0, np.pi/2, f_diff)[0]
white_sky_spec = quad(integral_white_sky_func, 0, np.pi/2, f_spec)[0]
print("White sky diffuse reflectance: {0:.3f}, specular: {1:.3f}, total: {2:.3f}".format(white_sky_diff, white_sky_spec, white_sky_diff+white_sky_spec)) 

plt.plot(x, y_diffuse, label="diffuse")
plt.plot(x, y_specular, label="specular")
plt.plot(x, y_total, label="total")
# plt.plot(x, y_specular_vacuum_after, label="specular, vacuum",linestyle='-',color='y')
# plt.plot(x, y_specular_LXe_lowp, label="specular, 0.2 barg LXe",linestyle='-.',color='y')
# # #plt.plot(x, y_specular_LXe_lowp2, label="spec_lowp2")
# # plt.plot(x, y_specular_LXe_hip, label="specular, 1.45 barg LXe",linestyle='--',color='y')
# # #plt.plot(x, y_specular_LXe_hip2, label="spec_hip2")
# plt.plot(x, y_diffuse_vacuum_after, label="diffuse, vacuum",linestyle='-',color='c')
# plt.plot(x, y_diffuse_LXe_lowp, label="diffuse, 0.2 barg LXe",linestyle='-.',color='c')
# # #plt.plot(x, y_diffuse_LXe_lowp2, label="diff_lowp2")
# # # plt.plot(x, y_diffuse_LXe_hip, label="diffuse, 1.45 barg LXe",linestyle='--',color='c')
# # #plt.plot(x, y_diffuse_LXe_hip2, label="diff_hip2")
# plt.plot(x, y_total_vacuum_after, label="total, vacuum",linestyle='-',color='b')
# plt.plot(x, y_total_LXe_lowp, label="total, 0.2 barg LXe",linestyle='-.',color='b')
# # #plt.plot(x, y_total_LXe_lowp2, label="total_lowp2")
# # # plt.plot(x, y_total_LXe_hip, label="total, 1.45 barg LXe",linestyle='--',color='b')
# # #plt.plot(x, y_total_LXe_hip2, label="total_hip2")
# # # # Line styles: '-', '--', '-.', ':'

plt.xlabel("incident angle (degrees)")
plt.ylabel("reflectance (fraction)")
plt.legend()
plt.ylim(0,1.1)

plt.title("Fitted "+sample_name+" Reflectance, 178 nm")
t3=time.time()
print("Reflectance calc time: {0}".format(t3-t2))
plt.show()

