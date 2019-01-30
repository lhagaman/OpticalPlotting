import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF
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
# s5_30 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_24_00.txt")
# s5_45 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_29_18.txt")
# s5_52 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_38_44.txt") # Two files exist for 52 deg; first probably got redone bc the angle range was too small
# s5_60 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_43_44.txt") 
# s5_67 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_48_38.txt")
# s5_75 = Run(path + "Sample 5 with bubbles\\2018_11_30__14_54_19.txt")
# s8_30 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_52_24.txt")
# s8_45 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_46_48.txt")
# s8_52 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_42_00.txt")
# s8_60 = Run(path + "Sample 8 no bubbles\\2018_12_03__14_57_12.txt") 
# s8_67 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_01_51.txt")
# s8_75 = Run(path + "Sample 8 no bubbles\\2018_12_03__15_06_25.txt") 
# s9_first_30 = Run(path_run1 + "Sample 9\\2018_11_02__20_38_01.txt")
# s9_first_45 = Run(path_run1 + "Sample 9\\2018_11_02__20_32_34.txt")
# s9_first_52 = Run(path_run1 + "Sample 9\\2018_11_02__20_27_27.txt")
# s9_first_60 = Run(path_run1 + "Sample 9\\2018_11_02__20_22_13.txt")
# s9_first_67 = Run(path_run1 + "Sample 9\\2018_11_02__20_17_11.txt")
# s9_first_75 = Run(path_run1 + "Sample 9\\2018_11_02__20_11_44.txt")
# s9_bubbles_30 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_26_05.txt")
# s9_bubbles_45 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_21_25.txt")
# s9_bubbles_52 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_16_32.txt")
# s9_bubbles_60 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_11_46.txt") 
# s9_bubbles_67 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_07_06.txt")
# s9_bubbles_75 = Run(path + "Sample 9 with bubbles\\2018_11_30__15_02_17.txt") 
# s9_nobubbles_30 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_05_41.txt")
# s9_nobubbles_45 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__17_01_01.txt")
# s9_nobubbles_52 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_56_06.txt")
# s9_nobubbles_60 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_51_00.txt") 
# s9_nobubbles_67 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_46_13.txt")
# s9_nobubbles_75 = Run(path + "Sample 9 no bubbles\\Before getter\\2018_11_30__16_40_45.txt") 
# s9_getter_30 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_13_32.txt")
# s9_getter_45 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_08_40.txt")
# s9_getter_52 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__11_03_50.txt")
# s9_getter_60 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_58_47.txt") 
# s9_getter_67 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_53_50.txt")
# s9_getter_75 = Run(path + "Sample 9 no bubbles\\After getter\\2018_12_03__10_48_50.txt") 
s9_lowp_30 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_52_04.txt")
s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt")
s9_lowp_60.change_theta_i(60) 
s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_67.change_theta_i(67)
s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
# s9_lowp2_30 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_28_41.txt")
# s9_lowp2_45 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_23_51.txt")
# s9_lowp2_52 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_18_50.txt")
# s9_lowp2_60 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_14_04.txt") 
# s9_lowp2_67 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_09_15.txt")
# s9_lowp2_75 = Run(path + "Sample 9 lower pressure 2\\2018_12_07__13_04_19.txt")
# s9_medp_30 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_56_28.txt")
# s9_medp_45 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_51_07.txt")
# s9_medp_52 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_44_43.txt")
# s9_medp_60 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_39_51.txt") 
# s9_medp_67 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_35_13.txt")
# s9_medp_75 = Run(path + "Sample 9 medium pressure\\2018_12_05__15_29_31.txt")  
# s9_medp2_30 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_42_49.txt")
# s9_medp2_45 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_36_41.txt")
# s9_medp2_52 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_31_00.txt")
# s9_medp2_60 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_25_38.txt") 
# s9_medp2_67 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_20_48.txt")
# s9_medp2_75 = Run(path + "Sample 9 medium pressure 2\\2018_12_07__14_14_47.txt") 
s9_hip_30 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_33_42.txt")
s9_hip_45 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_28_40.txt")
s9_hip_52 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_19_59.txt")
s9_hip_60 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_15_21.txt") 
s9_hip_67 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_10_17.txt")
s9_hip_75 = Run(path + "Sample 9 higher pressure\\2018_12_05__17_03_53.txt") 
# s9_hip2_30 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_59_52.txt")
# s9_hip2_45 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_55_11.txt")
# s9_hip2_52 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_50_31.txt")
# s9_hip2_60 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_45_32.txt") 
# s9_hip2_67 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_40_40.txt")
# s9_hip2_75 = Run(path + "Sample 9 higher pressure 2\\2018_12_07__16_35_58.txt") 
# s9_lowp_above_30 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_37_47.txt")
# s9_lowp_above_45 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_32_48.txt")
# s9_lowp_above_52 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_27_11.txt")
# s9_lowp_above_60 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_22_25.txt") 
# s9_lowp_above_67 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_17_42.txt")
# s9_lowp_above_75 = Run(path + "Sample 9 1-8_ above center\\2018_12_07__11_12_10.txt")


runs = [s9_lowp_30, s9_lowp_45, s9_lowp_52, s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_lowp_30,s9_lowp2_30,s9_medp_30,s9_medp2_30,s9_hip_30,s9_hip2_30]#[s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp2_30,s9_lowp2_45,s9_lowp2_52,s9_lowp2_60,s9_lowp2_67,s9_lowp2_75,s9_medp2_30,s9_medp2_45,s9_medp2_52,s9_medp2_60,s9_medp2_67,s9_medp2_75,s9_hip2_30,s9_hip2_45,s9_hip2_52,s9_hip2_60,s9_hip2_67,s9_hip2_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_lowp_above_30,s9_lowp_above_45,s9_lowp_above_52,s9_lowp_above_60,s9_lowp_above_67,s9_lowp_above_75]#[s9_lowp_75,s9_medp_75,s9_hip_75]#[s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75,s9_medp_30,s9_medp_45,s9_medp_52,s9_medp_60,s9_medp_67,s9_medp_75,s9_hip_30,s9_hip_45,s9_hip_52,s9_hip_60,s9_hip_67,s9_hip_75]#[s5_30,s5_45,s5_52,s5_60,s5_67,s5_75]#[s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75,s9_getter_30,s9_getter_45,s9_getter_52,s9_getter_60,s9_getter_67,s9_getter_75]#[s9_first_30,s9_first_45,s9_first_52,s9_first_60,s9_first_67,s9_first_75,s9_nobubbles_30,s9_nobubbles_45,s9_nobubbles_52,s9_nobubbles_60,s9_nobubbles_67,s9_nobubbles_75]#
labels=["30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]#["0.2 barg","0.2 barg fixed","1.0 barg","1.0 barg fixed","1.5 barg","1.5 barg fixed"]#["1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["0.2 barg, 30 deg","0.2 barg, 45 deg","0.2 barg, 52 deg","0.2 barg, 60 deg","0.2 barg, 67 deg","0.2 barg, 75 deg","1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["center, 30 deg","center, 45 deg","center, 52 deg","center, 60 deg","center, 67 deg","center, 75 deg","1/8\" above, 30 deg","1/8\" above, 45 deg","1/8\" above, 52 deg","1/8\" above, 60 deg","1/8\" above, 67 deg","1/8\" above, 75 deg"]#["0.2 barg","1.0 barg","1.5 barg"]#["1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["0.2 barg, 30 deg","0.2 barg, 45 deg","0.2 barg, 52 deg","0.2 barg, 60 deg","0.2 barg, 67 deg","0.2 barg, 75 deg","1.0 barg, 30 deg","1.0 barg, 45 deg","1.0 barg, 52 deg","1.0 barg, 60 deg","1.0 barg, 67 deg","1.0 barg, 75 deg","1.5 barg, 30 deg","1.5 barg, 45 deg","1.5 barg, 52 deg","1.5 barg, 60 deg","1.5 barg, 67 deg","1.5 barg, 75 deg"]#["30 deg","45 deg","52 deg","60 deg","67 deg","75 deg"]#["before getter, 30 deg", "before getter, 45 deg", "before getter, 52 deg", "before getter, 60 deg", "before getter, 67 deg", "before getter, 75 deg", "after getter, 30 deg", "after getter, 45 deg", "after getter, 52 deg", "after getter, 60 deg", "after getter, 67 deg", "after getter, 75 deg"]#["75 degrees"]#["run 1, 30 deg", "run 1, 45 deg", "run 1, 52 deg", "run 1, 60 deg", "run 1, 67 deg", "run 1, 75 deg", "run 2, 30 deg", "run 2, 45 deg", "run 2, 52 deg", "run 2, 60 deg", "run 2, 67 deg", "run 2, 75 deg"]#

# Plot BRIDF data
sample_name="LZ Skived"
plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 178 nm", log=True, labels=labels, include_legend=True, errorbars=True, legend_loc=0)
# plt.text(0.88,0.88,r"$75^{\circ}$",transform=plt.gca().transAxes,fontsize=13) # Positions for s9 low_p
# plt.text(0.85,0.78,r"$67^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.62,r"$60^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.48,r"$52^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.38,r"$45^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.7,0.25,r"$30^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.88,0.84,r"$75^{\circ}$",transform=plt.gca().transAxes,fontsize=13) # Positions for low_p2 and hi_p2
# plt.text(0.85,0.72,r"$67^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.53,r"$60^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.38,r"$52^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.8,0.26,r"$45^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.7,0.16,r"$30^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.88,0.84,r"$75^{\circ}$",transform=plt.gca().transAxes,fontsize=13) # Positions for hi_p
# plt.text(0.85,0.72,r"$67^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.78,0.52,r"$60^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.78,0.3,r"$52^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.67,0.16,r"$45^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
# plt.text(0.1,0.4,r"$30^{\circ}$",transform=plt.gca().transAxes,fontsize=13)
t0=time.time()

# Fit data
# tmp = np.array([[1,2],[3,4]])
# tmp2 = list(tmp)
# print(tmp,tmp2)
# tmp1=np.arange(10)
# tmp2=np.arange(4,14)
# print(tmp1,tmp2,np.linspace(tmp1,tmp2,1))
fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[0.99,1.54,.15],average_angle=4, precision=.25, sigma_theta_i=2., use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,1.8,0.3]))#[0.5,1.4,0.04,5.0],[1.2,1.8,0.3,50.]#[0.5,1.4,0.1],[1.0,1.8,0.3]#[0.01,1.01,0.05],[2.0,3.0,0.5]
#fit_params_ang = fit_parameters_and_angle(get_independent_variables_and_relative_intensities(runs),average_angle=4.)
#fit_ang = fit_params_ang[0]
#fit_params = fit_params_ang[1:]
#fit_params= [0.68114103391512837, 1.6259581862678401, 0.11282346636450034, 8.5642437795248174]#[0.947, 1.555, 0.091]#[0.784,1.568,0.144]#[0.86,1.50,0.07]#[0.72,1.45,0.2]#[0.800,1.581,0.157]#
# Fitter params for s9 in vacuum after LXe, 2.15 solid angle
fit_params_vacuum=[0.68114103391512837, 1.6259581862678401, 0.11282346636450034, 8.5642437795248174]
# Fitter parameters for s9 in LXe, 2.15 solid angle factor, Lorentz model
fit_params_lowp=[0.9466040398353657, 1.5542353537306548, 0.09079476300181971]
fit_params_hip=[0.9871481402574648, 1.5715034469520426, 0.09586940402324681]
#fit_params=fit_params_lowp
# # Fitted parameters for s9 2.44 solid angle factor
# fit_params_lowp=[0.784,1.568,0.144]
# fit_params_lowp2=[0.892,1.563,0.176]
# fit_params_hip=[0.800,1.581,0.157]
# fit_params_hip2=[0.868,1.558,0.225]
# fit_params_lowp_215=[1.011,1.563,0.118] # 2.15 solid angle factor
print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))
# phi_r_list = np.linspace(-np.pi,np.pi,10)
# BRIDF_test_list = [BRIDF(52., phi_r, 0., 1.69, 0.5, fit_params) for phi_r in phi_r_list]
# print("BRIDF at normal incidence and viewing: ",BRIDF_test_list)

# Plot BRIDF model from fits
n_LXe_178 = 1.69
sigma_theta_i=2.
precision=0.25
average_angle=4
#plot_TSTR_fit(20., n_LXe_178, fit_params, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
plot_TSTR_fit(30., n_LXe_178, fit_params, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
plot_TSTR_fit(45., n_LXe_178, fit_params, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
plot_TSTR_fit(52., n_LXe_178, fit_params, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
plot_TSTR_fit(60., n_LXe_178, fit_params, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
#independent_variables_array = get_independent_variables_and_relative_intensities(runs)[0]
#independent_variables_array = [list+[sigma_theta_i,precision,average_angle] for list in independent_variables_array]
#print([list[2] for list in independent_variables_array])
#theta_r_list = [list[0] for list in independent_variables_array]
#plt.plot(theta_r_list,fitter(independent_variables_array, np.log(fit_params[0]),np.log(fit_params[1]-1),np.log(fit_params[2])),color="y")
plot_TSTR_fit(67., n_LXe_178, fit_params, color="c", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=5)
# plot_TSTR_fit(75., n_LXe_178, fit_params, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i,phi_r=10)
#plt.text(0.1,0.1,r"Fit: $\rho_L$={0:.3f}, n={1:.2f}, $\gamma$={2:.3f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
#plt.text(0.1,0.2,r"Fit: $\rho_L$={0:.3f}, n={1:.2f}, $\gamma$={2:.3f}, K={3:.3f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
t2=time.time()
print("Plotting time: {0}".format(t2-t1))

# Now calculate hemispherical reflectance
# plt.figure()

# Different incident angles to calculate for
x = [0,10,20,30, 45.1, 55, 60, 65, 70, 75, 80, 85]#[0,30,45.1,60,70,75,80,85]#[5,60,70,80]#[5,30, 45, 60, 70, 75, 80, 85]#[0,10,20,30, 45, 55, 60, 65, 70, 75, 80, 85]#

# y_diffuse = [reflectance_diffuse(theta, n_LXe_178, 0.5, fit_params) for theta in x]
# y_specular = [reflectance_specular(theta, n_LXe_178, 0.5, fit_params) for theta in x]
# y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

# print("Diffuse reflectances: ",y_diffuse)
# print("Specular reflectances: ",y_specular)

x_full=x[:]
x_full.append(90)
sin_th = np.array([np.sin(xx*np.pi/180.) for xx in x_full])
d_th = np.array([10,10,10,15,10,5,5,5,5,5,5,5])*np.pi/180.
y_diffuse_vacuum_after=[0.5578751171383803, 0.5578645143005274, 0.5576939575569828, 0.5568501861704306, 0.551086564170089, 0.538862412662774, 0.5268697521170906, 0.5079125730733413, 0.4778737783402822, 0.4298921339077521, 0.3520995149402873, 0.22288077539943185]
y_specular_vacuum_after= [0.055008024656618325, 0.05518586897714156, 0.055937258940607706, 0.05802618505411558, 0.06795857669468221, 0.08518472501034707, 0.10042656606592987, 0.1230381756764488, 0.15759182167624392, 0.21483421777180697, 0.32793108506231394, 0.6332518225200205]
y_specular_vacuum_after=[min(x[0],1-x[1]) for x in zip(y_specular_vacuum_after,y_diffuse_vacuum_after)]
y_total_vacuum_after = [sum(x) for x in zip(y_specular_vacuum_after,y_diffuse_vacuum_after)]
y_total_vacuum_after_full = y_total_vacuum_after[:]
y_total_vacuum_after_full.append(y_total_vacuum_after[-1])
y_total_vacuum_after_full=np.array(y_total_vacuum_after_full)
y_sky_vacuum_after_right = np.sum(sin_th[1:]*d_th*y_total_vacuum_after_full[1:])
y_sky_vacuum_after_left = np.sum(sin_th[:-1]*d_th*y_total_vacuum_after_full[:-1])
y_sky_vacuum_after_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_vacuum_after_full[:-1]+y_total_vacuum_after_full[1:])/2)
print("White sky reflectance, vacuum after, low est: ",y_sky_vacuum_after_left)
print("White sky reflectance, vacuum after, high est: ",y_sky_vacuum_after_right)
print("White sky reflectance, vacuum after, middle est: ",y_sky_vacuum_after_mid)
# # y_diffuse_vacuum_before=[0.45766348181118743, 0.4580440040036757, 0.45840578569295093, 0.458373743496906, 0.45519279945081914, 0.4467776090150314, 0.4382352346643636, 0.42459365404619115, 0.4030226445070669, 0.3691595655515466, 0.31672576341572695, 0.23997195829098059]
# # y_specular_vacuum_before=[0.06255417549996851, 0.06265684345501554, 0.06321515540210665, 0.06512830173742697, 0.07570232176333869, 0.09608785701329016, 0.11458965758798859, 0.14207103191906068, 0.1823613281476766, 0.240653361837574, 0.3259507169947907, 0.4865190393937654]
# # y_total_vacuum_before = [sum(x) for x in zip(y_specular_vacuum_before,y_diffuse_vacuum_before)]
# # y_total_vacuum_before_full = y_total_vacuum_before[:]
# # y_total_vacuum_before_full.append(y_total_vacuum_before[-1])
# # y_sky_vacuum_before_right = np.sum(sin_th[1:]*d_th*y_total_vacuum_before_full[1:])
# # y_sky_vacuum_before_left = np.sum(sin_th[:-1]*d_th*y_total_vacuum_before_full[:-1])
# # print("White sky reflectance, vacuum before, low est: ",y_sky_vacuum_before_left)
# # print("White sky reflectance, vacuum before, high est: ",y_sky_vacuum_before_right)
# # y_diffuse_LXe_lowp_old=[0.6570412916757591, 0.6579357341415137, 0.6591016792259574, 0.6606138318516783, 0.6633688576680011, 0.6626957378115929, 0.6549616061782121, 0.6062281312066095, 0.0, 0.0, 0.0, 0.0]
# # y_specular_LXe_lowp_old= [0.0014054544422652119, 0.0014218233642250282, 0.0015141490462703899, 0.0018695965399099326, 0.0057283880366878585, 0.03827531730547219, 0.09984977500055535, 0.25165997130973317, 0.5000189208911733, 0.6960823791225147, 0.8249335365771578, 1.0]#1.0750904931429557]
y_diffuse_LXe_lowp=[0.7982997601321108, 0.7982673604741001, 0.7982065233759797, 0.7980038464847359, 0.7960858935743564, 0.7874156102421991, 0.7676760751881307, 0.6532634974917723, 0.23652460596145553, 0.13341800718215685, 0.08127525337984297, 0.05402807309618492]
y_specular_LXe_lowp=[0.0017852149063135958, 0.001818308226244964, 0.0019795301316915003, 0.002543034835696333, 0.009548179713833347, 0.053554449148170015, 0.11833288201197621, 0.2823038620542749, 0.5406700804638772, 0.6790627367772443, 0.798665136503454, 1.0700312085436152]
y_specular_LXe_lowp=[min(x[0],1-x[1]) for x in zip(y_specular_LXe_lowp,y_diffuse_LXe_lowp)]
y_total_LXe_lowp = [sum(x) for x in zip(y_specular_LXe_lowp,y_diffuse_LXe_lowp)]
y_total_LXe_lowp_full = y_total_LXe_lowp[:]
y_total_LXe_lowp_full.append(y_total_LXe_lowp[-1])
y_total_LXe_lowp_full=np.array(y_total_LXe_lowp_full)
y_sky_LXe_lowp_right = np.sum(sin_th[1:]*d_th*y_total_LXe_lowp_full[1:])
y_sky_LXe_lowp_left = np.sum(sin_th[:-1]*d_th*y_total_LXe_lowp_full[:-1])
y_sky_LXe_lowp_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_LXe_lowp_full[:-1]+y_total_LXe_lowp_full[1:])/2)
print("White sky reflectance, LXe 0.2 barg, low est: ",y_sky_LXe_lowp_left)
print("White sky reflectance, LXe 0.2 barg, high est: ",y_sky_LXe_lowp_right)
print("White sky reflectance, LXe 0.2 barg, middle est: ",y_sky_LXe_lowp_mid)
# # y_diffuse_LXe_lowp2= [0.7402398286291142, 0.7417007202832876, 0.7436238363017107, 0.7461638687993092, 0.7512130151093066, 0.752163989934962, 0.7433876636862394, 0.6765365007371161, 0.0, 0.0, 0.0, 0.0]
# # y_specular_LXe_lowp2= [0.001527762931549758, 0.0015516293860655497, 0.0016747374647144962, 0.0021276141278918324, 0.007227369888786746, 0.04961757416725003, 0.11917259018207207, 0.2626128326744476, 0.4667668545228056, 0.6430628471896185, 0.7907470379047791, 1.0]#1.0997835008723356]
# # y_total_LXe_lowp2 = [sum(x) for x in zip(y_specular_LXe_lowp2,y_diffuse_LXe_lowp2)]
# # y_diffuse_LXe_hip_old= [0.6829307222870662, 0.6840336258528498, 0.6854793518118913, 0.6873917251376706, 0.6913214705135405, 0.6930010462636261, 0.6892947926267271, 0.6618038165708974, 0.0, 0.0, 0.0, 0.0]
# # y_specular_LXe_hip_old= [0.001112165427732242, 0.0011263277428219362, 0.00120284458242256, 0.0014878128453774913, 0.00431818948159568, 0.028577408166662696, 0.0767039434227785, 0.1938216373997157, 0.40704522413508704, 0.6172302526293288, 0.7723284466198603, 1.0]#1.047912122232299]
y_diffuse_LXe_hip= [0.8491875929406879, 0.8491509762266888, 0.8490914075177584, 0.8489201262096208, 0.8474340080642626, 0.8411375534056131, 0.8279556472564547, 0.7687016503469466, 0.3323357823672721, 0.16558390232400014, 0.10476839339443421, 0.06534115759627902]
y_specular_LXe_hip= [0.0013400522898790678, 0.0013641072780542977, 0.001480245806814926, 0.0018772752465937084, 0.005731975904048718, 0.036692820567370533, 0.0848219866352421, 0.2022403891367498, 0.4541913398256783, 0.6259906120320465, 0.7543231293523422, 1.0211097366859878]
y_specular_LXe_hip=[min(x[0],1-x[1]) for x in zip(y_specular_LXe_hip,y_diffuse_LXe_hip)]
y_total_LXe_hip = [sum(x) for x in zip(y_specular_LXe_hip,y_diffuse_LXe_hip)]
y_total_LXe_hip_full = y_total_LXe_hip[:]
y_total_LXe_hip_full.append(y_total_LXe_hip[-1])
y_total_LXe_hip_full=np.array(y_total_LXe_hip_full)
y_sky_LXe_hip_right = np.sum(sin_th[1:]*d_th*y_total_LXe_hip_full[1:])
y_sky_LXe_hip_left = np.sum(sin_th[:-1]*d_th*y_total_LXe_hip_full[:-1])
y_sky_LXe_hip_mid = np.sum((sin_th[:-1]+sin_th[1:])/2*d_th*(y_total_LXe_hip_full[:-1]+y_total_LXe_hip_full[1:])/2)
print("White sky reflectance, LXe 1.45 barg, low est: ",y_sky_LXe_hip_left)
print("White sky reflectance, LXe 1.45 barg, high est: ",y_sky_LXe_hip_right)
print("White sky reflectance, LXe 1.45 barg, middle est: ",y_sky_LXe_hip_mid)
# # y_diffuse_LXe_hip2= [0.7116527697234036, 0.7138323922133201, 0.7167431409327432, 0.7206668394297263, 0.729091743241813, 0.7335885883463384, 0.7262823707645063, 0.6474078491699706, 0.0, 0.0, 0.0, 0.0]
# # y_specular_LXe_hip2=  [0.0016528720786998794, 0.0016884735251564945, 0.001857732976822431, 0.0024483848517261844, 0.009600045662265523, 0.06251613539473333, 0.13418304588129548, 0.25679333990263276, 0.4159533052261603, 0.5728408076108408, 0.7414240888009019, 1.0]#1.1194372218351034]
# # y_total_LXe_hip2 = [sum(x) for x in zip(y_specular_LXe_hip2,y_diffuse_LXe_hip2)]

# # plt.plot(x, y_diffuse, label="diffuse")
# # plt.plot(x, y_specular, label="specular")
# # plt.plot(x, y_total, label="total")
# plt.plot(x, y_specular_vacuum_after, label="specular, vacuum",linestyle='-',color='y')
# plt.plot(x, y_specular_LXe_lowp, label="specular, 0.2 barg LXe",linestyle='-.',color='y')
# # #plt.plot(x, y_specular_LXe_lowp2, label="spec_lowp2")
# plt.plot(x, y_specular_LXe_hip, label="specular, 1.45 barg LXe",linestyle='--',color='y')
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
# # # Line styles: '-', '--', '-.', ':'

# plt.xlabel("incident angle (degrees)")
# plt.ylabel("reflectance (fraction)")
# plt.legend()

# plt.title("Fitted "+sample_name+" Reflectance, 178 nm")
# t3=time.time()
# print("Reflectance calc time: {0}".format(t3-t2))
plt.show()

