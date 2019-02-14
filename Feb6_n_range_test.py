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
s9_30 = Run(s9_path + "2018_11_02__20_38_01.txt", mirror_filenames_and_angles)
s9_45 = Run(s9_path + "2018_11_02__20_32_34.txt", mirror_filenames_and_angles)
s9_52 = Run(s9_path + "2018_11_02__20_27_27.txt", mirror_filenames_and_angles)
s9_60 = Run(s9_path + "2018_11_02__20_22_13.txt", mirror_filenames_and_angles)
s9_67 = Run(s9_path + "2018_11_02__20_17_11.txt", mirror_filenames_and_angles)
s9_75 = Run(s9_path + "2018_11_02__20_11_44.txt", mirror_filenames_and_angles)

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

runs = [s9_30, s9_45, s9_52, s9_60,s9_67,s9_75]
labels=["30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="LZ Skived"
plot_runs(runs, title=sample_name+" in LXe, Run 1, 178 nm", log=True, labels=labels, include_legend=False, errorbars=True, legend_loc=0)

n_LXe_178 = 1.69
sigma_theta_i=2
precision=0.25
average_angle=4

fit_params_all = [0.977, 1.55, 0.105]

plot_TSTR_fit(30, n_LXe_178, fit_params_all, color="r", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=True)
plot_TSTR_fit(45, n_LXe_178, fit_params_all, color="g", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(52, n_LXe_178, fit_params_all, color="b", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(60, n_LXe_178, fit_params_all, color="m", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(67, n_LXe_178, fit_params_all, color="c", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
plot_TSTR_fit(75, n_LXe_178, fit_params_all, color="y", average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)

plt.show()

