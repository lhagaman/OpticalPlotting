import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, chi_squared
import time

mirror_path = "First Xe Run Measurements/Mirror alignment checks 10-29/"

# must be in ascending angle order
mirror_filenames_and_angles = [
[30., mirror_path + "2018_10_29__09_33_56.txt"],
[45., mirror_path + "2018_10_29__09_31_00.txt"],
[52., mirror_path + "2018_10_29__09_31_00.txt"], # data not taken at this angle, must assume same as 45
[60., mirror_path + "2018_10_29__09_25_03.txt"],
[67., mirror_path + "2018_10_29__09_25_03.txt"], # data not taken at this angle, must assume same as 60
[75., mirror_path + "2018_10_29__09_45_18.txt"]]

path = "First Xe Run Measurements/first measurements with no bubbles in cell 11-01-2/"

s9_path = path + "Sample 9/"

s9_30 = Run(s9_path + "2018_11_02__20_38_01.txt")#, mirror_filenames_and_angles)
s9_45 = Run(s9_path + "2018_11_02__20_32_34.txt")#, mirror_filenames_and_angles)
s9_52 = Run(s9_path + "2018_11_02__20_27_27.txt")#, mirror_filenames_and_angles)
s9_60 = Run(s9_path + "2018_11_02__20_22_13.txt")#, mirror_filenames_and_angles)
s9_67 = Run(s9_path + "2018_11_02__20_17_11.txt")#, mirror_filenames_and_angles)
s9_75 = Run(s9_path + "2018_11_02__20_11_44.txt")#, mirror_filenames_and_angles)


s3_path = path + "Sample 3/"

s3_30 = Run(s3_path + "2018_11_01__18_06_10.txt", mirror_filenames_and_angles)
s3_45 = Run(s3_path + "2018_11_02__19_42_42.txt", mirror_filenames_and_angles)
s3_52 = Run(s3_path + "2018_11_02__19_47_37.txt", mirror_filenames_and_angles)
s3_60 = Run(s3_path + "2018_11_02__19_52_30.txt", mirror_filenames_and_angles)
s3_67 = Run(s3_path + "2018_11_02__19_59_24.txt", mirror_filenames_and_angles)
s3_75 = Run(s3_path + "2018_11_01__17_51_10.txt", mirror_filenames_and_angles)

s1_path = path + "Sample 1/"

s1_30 = Run(s1_path + "2018_11_02__18_28_43.txt", mirror_filenames_and_angles)
s1_45 = Run(s1_path + "2018_11_02__18_35_18.txt", mirror_filenames_and_angles)
s1_52 = Run(s1_path + "2018_11_02__18_41_20.txt", mirror_filenames_and_angles)
s1_60 = Run(s1_path + "2018_11_02__18_47_03.txt", mirror_filenames_and_angles)
s1_67 = Run(s1_path + "2018_11_02__18_51_55.txt", mirror_filenames_and_angles)
s1_75 = Run(s1_path + "2018_11_02__18_57_35.txt", mirror_filenames_and_angles)

"""
s9_30 = Run(s9_path + "2018_11_02__20_38_01.txt", mirror_filenames_and_angles)
s9_30.change_theta_i(30)
s9_45 = Run(s9_path + "2018_11_02__20_32_34.txt", mirror_filenames_and_angles)
s9_45.change_theta_i(45)
s9_52 = Run(s9_path + "2018_11_02__20_27_27.txt", mirror_filenames_and_angles)
s9_52.change_theta_i(52)
s9_60 = Run(s9_path + "2018_11_02__20_22_13.txt", mirror_filenames_and_angles)
s9_60.change_theta_i(60)
s9_67 = Run(s9_path + "2018_11_02__20_17_11.txt", mirror_filenames_and_angles)
s9_67.change_theta_i(67)
s9_75 = Run(s9_path + "2018_11_02__20_11_44.txt", mirror_filenames_and_angles)
s9_75.change_theta_i(75)

s9_30_low = Run(s9_path + "2018_11_02__20_38_01.txt")
s9_30_low.change_theta_i(30 - 2)
s9_45_low = Run(s9_path + "2018_11_02__20_32_34.txt")
s9_45_low.change_theta_i(45 - 2)
s9_52_low = Run(s9_path + "2018_11_02__20_27_27.txt")
s9_52_low.change_theta_i(52 - 2)
s9_60_low = Run(s9_path + "2018_11_02__20_22_13.txt")
s9_60_low.change_theta_i(60 - 2)
s9_67_low = Run(s9_path + "2018_11_02__20_17_11.txt")
s9_67_low.change_theta_i(67 - 2)
s9_75_low = Run(s9_path + "2018_11_02__20_11_44.txt")
s9_75_low.change_theta_i(75 - 2)

s9_30_high = Run(s9_path + "2018_11_02__20_38_01.txt")
s9_30_high.change_theta_i(30 + 2)
s9_45_high = Run(s9_path + "2018_11_02__20_32_34.txt")
s9_45_high.change_theta_i(45 + 2)
s9_52_high = Run(s9_path + "2018_11_02__20_27_27.txt")
s9_52_high.change_theta_i(52 + 2)
s9_60_high = Run(s9_path + "2018_11_02__20_22_13.txt")
s9_60_high.change_theta_i(60 + 2)
s9_67_high = Run(s9_path + "2018_11_02__20_17_11.txt")
s9_67_high.change_theta_i(67 + 2)
s9_75_high = Run(s9_path + "2018_11_02__20_11_44.txt")
s9_75_high.change_theta_i(75 + 2)
"""
runs = [s9_30, s9_45, s9_52, s9_60,s9_67,s9_75]
labels=["30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="LZ Skived, delta n = 0, no mirror correction"
plot_runs(runs, title=sample_name+" in LXe, Run 1, 178 nm", log=True, labels=labels, include_legend=False, errorbars=True, legend_loc=0)

t0=time.time()

# Fit data

n_LXe_178 = 1.69
sigma_theta_i=2
precision=0.25
average_angle=4

chi_30_list = []
chi_45_list = []
chi_52_list = []
chi_60_list = []
chi_67_list = []
chi_75_list = []

independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(runs)
fit_params_all = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))

"""
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_30)
fit_params_30 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_30_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_30, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 30...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_45)
fit_params_45 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_45_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_45, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 45...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_52)
fit_params_52 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_52_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_52, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 52...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_60)
fit_params_60 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_60_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_60, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 60...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_67)
fit_params_67 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_67_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_67, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 67...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_75)
fit_params_75 = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[10.,2.0,0.5]))
chi_75_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_75, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 75...")
"""
"""
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_30_low)
fit_params_30_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_30_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_30_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 30 low...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_45_low)
fit_params_45_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_45_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_45_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 45 low...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_52_low)
fit_params_52_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_52_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_52_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 52 low...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_60_low)
fit_params_60_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_60_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_60_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 60 low...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_67_low)
fit_params_67_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_67_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_67_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 67 low...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_75_low)
fit_params_75_low = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_75_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_75_low, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 75 low...")
"""
"""
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_30_high)
fit_params_30_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_30_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_30_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 30 high...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_45_high)
fit_params_45_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_45_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_45_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 45 high...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_52_high)
fit_params_52_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_52_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_52_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 52 high...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_60_high)
fit_params_60_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_60_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_60_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 60 high...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_67_high)
fit_params_67_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_67_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_67_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 67 high...")
independent_variables_array_intensity_array = get_independent_variables_and_relative_intensities(s9_lowp_75_high)
fit_params_75_high = fit_parameters(independent_variables_array_intensity_array,p0=[0.99,1.54,.15],
	average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=False,bounds=([0.5,1.4,0.04],[1.2,2.0,0.3]))
chi_75_list.append(chi_squared(independent_variables_array_intensity_array[0], independent_variables_array_intensity_array[1], 
	independent_variables_array_intensity_array[2], fit_params_75_high, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
print("done fitting 75 high...")
"""


"""
fit_params_30 = [0.99,1.54,.15]
fit_params_45 = [0.99,1.54,.15]
fit_params_52 = [0.99,1.54,.15]
fit_params_60 = [0.99,1.54,.15]
fit_params_67 = [0.99,1.54,.15]
fit_params_75 = [0.99,1.54,.15]

chi_30_list = [3.1415]
chi_45_list = [3.1415]
chi_52_list = [3.1415]
chi_60_list = [3.1415]
chi_67_list = [3.1415]
chi_75_list = [3.1415]
"""
"""
print("chi squared values: ")
print(chi_30_list)
print(chi_45_list)
print(chi_52_list)
print(chi_60_list)
print(chi_67_list)
print(chi_75_list)

if chi_30_list.index(min(chi_30_list)) == 0:
	angle_30 = 30
elif chi_30_list.index(min(chi_30_list)) == 1:
	fit_params_30 = fit_params_30_low
	angle_30 = 28
elif chi_30_list.index(min(chi_30_list)) == 2:
	fit_params_30 = fit_params_30_high
	angle_30 = 32

if chi_45_list.index(min(chi_45_list)) == 0:
	angle_45 = 45
elif chi_45_list.index(min(chi_45_list)) == 1:
	fit_params_45 = fit_params_45_low
	angle_45 = 43
elif chi_45_list.index(min(chi_45_list)) == 2:
	fit_params_45 = fit_params_45_high
	angle_45 = 47

if chi_52_list.index(min(chi_52_list)) == 0:
	angle_52 = 52
elif chi_52_list.index(min(chi_52_list)) == 1:
	fit_params_52 = fit_params_52_low
	angle_52 = 50
elif chi_52_list.index(min(chi_52_list)) == 2:
	fit_params_52 = fit_params_52_high
	angle_52 = 54

if chi_60_list.index(min(chi_60_list)) == 0:
	angle_60 = 60
elif chi_60_list.index(min(chi_60_list)) == 1:
	fit_params_60 = fit_params_60_low
	angle_60 = 58
elif chi_60_list.index(min(chi_60_list)) == 2:
	fit_params_60 = fit_params_60_high
	angle_60 = 62

if chi_67_list.index(min(chi_67_list)) == 0:
	angle_67 = 67
elif chi_67_list.index(min(chi_67_list)) == 1:
	fit_params_67 = fit_params_67_low
	angle_67 = 65
elif chi_67_list.index(min(chi_67_list)) == 2:
	fit_params_67 = fit_params_67_high
	angle_67 = 69

if chi_75_list.index(min(chi_75_list)) == 0:
	angle_75 = 75
elif chi_75_list.index(min(chi_75_list)) == 1:
	fit_params_75 = fit_params_75_low
	angle_75 = 73
elif chi_75_list.index(min(chi_75_list)) == 2:
	fit_params_75 = fit_params_75_high
	angle_75 = 77
"""
"""
fit_params_30 = [0.99,1.54,.15]
fit_params_45 = [0.99,1.54,.15]
fit_params_52 = [0.99,1.54,.15]
fit_params_60 = [0.99,1.54,.15]
fit_params_67 = [0.99,1.54,.15]
fit_params_75 = [0.99,1.54,.15]
"""

t1=time.time()
print("Fitting time: {0}".format(t1-t0))

plot_TSTR_fit(30, n_LXe_178, fit_params_all, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=True)
plot_TSTR_fit(45, n_LXe_178, fit_params_all, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(52, n_LXe_178, fit_params_all, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(60, n_LXe_178, fit_params_all, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(67, n_LXe_178, fit_params_all, color="c", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(75, n_LXe_178, fit_params_all, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)

"""
plot_TSTR_fit(angle_30, n_LXe_178, fit_params_30, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(angle_45, n_LXe_178, fit_params_45, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(angle_52, n_LXe_178, fit_params_52, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(angle_60, n_LXe_178, fit_params_60, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(angle_67, n_LXe_178, fit_params_67, color="c", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(angle_75, n_LXe_178, fit_params_75, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)

plt.text(0.02,0.22,r"Fit " + str(angle_30) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_30),transform=plt.gca().transAxes,fontsize=11)
plt.text(0.02,0.18,r"Fit " + str(angle_45) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_45),transform=plt.gca().transAxes,fontsize=11)
plt.text(0.02,0.14,r"Fit " + str(angle_52) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_52),transform=plt.gca().transAxes,fontsize=11)
plt.text(0.02,0.10,r"Fit " + str(angle_60) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_60),transform=plt.gca().transAxes,fontsize=11)
plt.text(0.02,0.06,r"Fit " + str(angle_67) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_67),transform=plt.gca().transAxes,fontsize=11)
plt.text(0.02,0.02,r"Fit " + str(angle_75) + r": $\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params_75),transform=plt.gca().transAxes,fontsize=11)
"""

t2=time.time()
print("Plotting time: {0}".format(t2-t1))

plt.show()

