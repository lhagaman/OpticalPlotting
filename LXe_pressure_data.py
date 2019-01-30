import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF
import time

path_run1 = "First Xe Run Measurements\\first measurements with no bubbles in cell 11-01-2\\"
path = "2nd Xenon Run Measurements\\"


inc_020 = Run(path + "Sample 9 changing pressures\\2018_12_10__13_19_42.txt")
inc_022 = Run(path + "Sample 9 changing pressures\\2018_12_10__13_28_46.txt")
inc_029 = Run(path + "Sample 9 changing pressures\\2018_12_10__13_39_49.txt")
inc_038 = Run(path + "Sample 9 changing pressures\\2018_12_10__13_56_30.txt")
inc_045 = Run(path + "Sample 9 changing pressures\\2018_12_10__14_07_05.txt")
inc_053 = Run(path + "Sample 9 changing pressures\\2018_12_10__14_19_57.txt")
inc_060 = Run(path + "Sample 9 changing pressures\\2018_12_10__14_29_45.txt")
inc_070 = Run(path + "Sample 9 changing pressures\\2018_12_10__14_41_45.txt")
inc_078 = Run(path + "Sample 9 changing pressures\\2018_12_10__14_48_58.txt")
inc_092 = Run(path + "Sample 9 changing pressures\\2018_12_10__15_19_57.txt")
inc_098 = Run(path + "Sample 9 changing pressures\\2018_12_10__15_30_33.txt")
inc_124 = Run(path + "Sample 9 changing pressures\\2018_12_10__16_19_42.txt")
inc_134 = Run(path + "Sample 9 changing pressures\\2018_12_10__16_28_49.txt")
dec_126 = Run(path + "Sample 9 changing pressures\\2018_12_10__16_40_24.txt")
dec_107 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_03_45.txt")
dec_099 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_12_31.txt")
dec_092 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_26_42.txt")
dec_079 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_41_03.txt")
dec_071 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_48_47.txt")
dec_061 = Run(path + "Sample 9 changing pressures\\2018_12_10__17_58_52.txt")
dec_053 = Run(path + "Sample 9 changing pressures\\2018_12_10__18_09_02.txt")
dec_046 = Run(path + "Sample 9 changing pressures\\2018_12_10__18_18_34.txt")
dec_039 = Run(path + "Sample 9 changing pressures\\2018_12_10__18_29_12.txt")
dec_031 = Run(path + "Sample 9 changing pressures\\2018_12_10__18_42_33.txt")

runs = [dec_031,dec_053,dec_071,dec_092,dec_107,dec_126]#[inc_020,inc_038,inc_060,inc_078,inc_098,inc_124,inc_134]#[inc_020,inc_022,inc_029,inc_038,inc_045,inc_053,inc_060,inc_070,inc_078,inc_092,inc_098,inc_124,inc_134]#[dec_126,dec_107,dec_099,dec_092,dec_079,dec_071,dec_061,dec_053,dec_046,dec_039,dec_031]##[s9_lowp_30,s9_lowp2_30,s9_medp_30,s9_medp2_30,s9_hip_30,s9_hip2_30]#[s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp2_30,s9_lowp2_45,s9_lowp2_52,s9_lowp2_60,s9_lowp2_67,s9_lowp2_75,s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_lowp_75,s9_medp_75,s9_hip_75]#[s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s5_30,s5_45,s5_52,s5_60,s5_67,s5_75]#[s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75,s9_getter_30,s9_getter_45,s9_getter_52,s9_getter_60,s9_getter_67,s9_getter_75]#[s9_first_30,s9_first_45,s9_first_52,s9_first_60,s9_first_67,s9_first_75,s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75]#

labels=["0.31 barg","0.53 barg","0.71 barg","0.92 barg","1.07 barg","1.26 barg"]#["0.2 barg","0.38 barg","0.60 barg","0.78 barg","0.98 barg","1.24 barg","1.34 barg"]#["0.2 barg","0.22 barg","0.29 barg","0.38 barg","0.45 barg","0.53 barg","0.60 barg","0.70 barg","0.78 barg","0.92 barg","0.98 barg","1.24 barg","1.34 barg"]#["1.26 barg","1.07 barg","0.99 barg","0.92 barg","0.79 barg","0.71 barg","0.61 barg","0.53 barg","0.46 barg","0.39 barg","0.31 barg"]#["1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["0.2 barg, 30 deg","0.2 barg, 45 deg","0.2 barg, 52 deg","0.2 barg, 60 deg","0.2 barg, 67 deg","0.2 barg, 75 deg","1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["center, 30 deg","center, 45 deg","center, 52 deg","center, 60 deg","center, 67 deg","center, 75 deg","1/8\" above, 30 deg","1/8\" above, 45 deg","1/8\" above, 52 deg","1/8\" above, 60 deg","1/8\" above, 67 deg","1/8\" above, 75 deg"]#["0.2 barg","1.0 barg","1.5 barg"]#["1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["0.2 barg, 30 deg","0.2 barg, 45 deg","0.2 barg, 52 deg","0.2 barg, 60 deg","0.2 barg, 67 deg","0.2 barg, 75 deg","1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["30 deg","45 deg","52 deg","60 deg","67 deg","75 deg"]#["before getter, 30 deg", "before getter, 45 deg", "before getter, 52 deg", "before getter, 60 deg", "before getter, 67 deg", "before getter, 75 deg", "after getter, 30 deg", "after getter, 45 deg", "after getter, 52 deg", "after getter, 60 deg", "after getter, 67 deg", "after getter, 75 deg"]#["75 degrees"]#["run 1, 30 deg", "run 1, 45 deg", "run 1, 52 deg", "run 1, 60 deg", "run 1, 67 deg", "run 1, 75 deg", "run 2, 30 deg", "run 2, 45 deg", "run 2, 52 deg", "run 2, 60 deg", "run 2, 67 deg", "run 2, 75 deg"]#
# Plot BRIDF data
sample_name="LZ Skived, 60 deg"
plot_runs(runs, title=sample_name+", 178nm", log=False, labels=labels, errorbars=True, legend_loc=0, colormap=True)
t0=time.time()

# Fit data
#fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[0.24,1.2,.1,5.0],average_angle=4., precision=0.25,use_errs=True,use_spike=True)
#fit_params_ang = fit_parameters_and_angle(get_independent_variables_and_relative_intensities(runs),average_angle=4.)
#fit_ang = fit_params_ang[0]
#fit_params = fit_params_ang[1:]
#print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))

# Plot BRIDF model from fits
#plot_TSTR_fit(30., 1., fit_params, label="fitted", color="k", average_angle=4., precision=0.25)
#plot_TSTR_fit(45., 1., fit_params, color="k", average_angle=4., precision=0.25)
#plot_TSTR_fit(60., 1., fit_params, color="k", average_angle=4., precision=0.25)
#plot_TSTR_fit(75., 1., fit_params, color="k", average_angle=4., precision=0.25)
#plt.text(0.1,0.1,r"Fit: $\rho_L$={0:.3f}, n={1:.2f}, $\gamma$={2:.3f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
#plt.text(0.1,0.2,r"Fit: $\rho_L$={0:.3f}, n={1:.2f}, $\gamma$={2:.3f}, K={3:.3f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
t2=time.time()
print("Plotting time: {0}".format(t2-t1))

# Now calculate hemispherical reflectance
# plt.figure()

# # Different incident angles to calculate for
# x = [30, 45, 60, 75]#[0,10,20,30, 45, 55, 60, 65, 70, 75, 80, 85]#

# y_diffuse = [reflectance_diffuse(theta, 1., 0.5, fit_params) for theta in x]
# y_specular = [reflectance_specular(theta, 1., 0.5, fit_params) for theta in x]
# y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

# plt.plot(x, y_diffuse, label="diffuse")
# plt.plot(x, y_specular, label="specular")
# plt.plot(x, y_total, label="total")

# plt.xlabel("incident angle (degrees)")
# plt.ylabel("reflectance (fraction)")
# plt.legend()

# plt.title("Fitted "+sample_name+" Reflectance, 175 nm")
# t3=time.time()
# print("Reflectance calc time: {0}".format(t3-t2))
plt.show()
