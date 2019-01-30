import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit, plot_double_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF
import time

#path = "First Xe Run Measurements/first measurements with no bubbles in cell 11-01-2/Sample 1/"
#path = "2nd Xenon Run Measurements\\"
path_run1 = "First Xe Run Measurements\\first measurements with no bubbles in cell 11-01-2\\"

# s1_30 = Run(path + "2018_11_02__18_28_43.txt")
# s1_45 = Run(path + "2018_11_02__18_35_18.txt")
# s1_52 = Run(path + "2018_11_02__18_41_20.txt")
# s1_60 = Run(path + "2018_11_02__18_47_03.txt")
# s1_67 = Run(path + "2018_11_02__18_51_55.txt")
# s1_75 = Run(path + "2018_11_02__18_57_35.txt")
s9_first_30 = Run(path_run1 + "Sample 9\\2018_11_02__20_38_01.txt")
s9_first_45 = Run(path_run1 + "Sample 9\\2018_11_02__20_32_34.txt")
s9_first_52 = Run(path_run1 + "Sample 9\\2018_11_02__20_27_27.txt")
s9_first_60 = Run(path_run1 + "Sample 9\\2018_11_02__20_22_13.txt")
s9_first_67 = Run(path_run1 + "Sample 9\\2018_11_02__20_17_11.txt")
s9_first_75 = Run(path_run1 + "Sample 9\\2018_11_02__20_11_44.txt")
# s9_lowp_30 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_52_04.txt")
# s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
# s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
# s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt") 
# s9_lowp_60.change_theta_i(60) 
# s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
# s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")

#runs = [s1_30, s1_45, s1_52, s1_60, s1_67, s1_75]
runs = [s9_first_30, s9_first_45, s9_first_52, s9_first_60, s9_first_67, s9_first_75]#[s9_lowp_60,s9_lowp_67,s9_lowp_75]#[s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]

labels=["30 degrees", "45 degrees", "52 degrees", "60 degrees", "67 degrees", "75 degrees"]#["30 degrees", "45 degrees", "52 degrees", "60 degrees", "67 degrees", "75 degrees"]
#labels=["30 degrees", "45 degrees", "60 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="LZ Skived"
#plot_runs(runs, title=sample_name+", 178nm", log=True, labels=labels, errorbars=True)
#plt.show()

# In a loop: ask for input params, plot model overlaid on data
get_input=True
params=[0.9,1.55,0.15,-1,
        0.1,0.5]
param_names=["rho_L_1","n_1","gamma_1","K_1",
             "gamma_2","frac_1"]
fig_num=0
while get_input:
    fig_num += 1
    for ii,name in enumerate(param_names):
        while True:
            param = input('Input '+name+' (or enter to use '+str(params[ii])+'): ')
            try:
                params[ii] = float(param)
                break
            except ValueError: 
                if param == '': break
                print("Error: input a number")
    
    n_LXe_178 = 1.69
    n=n_LXe_178
    sigma_theta_i=2.0
    precision=0.25
    average_angle=4. # 6 for 60, 6.9 for 67, 8 for 75
	
    #plt.figure(fig_num)
    plot_runs(runs, title=sample_name+", 178 nm", log=True, labels=labels, errorbars=True)
	# Trying setting incident angles by hand, for better fits to specular spike
    plot_double_TSTR_fit(30., n, params, labels=["fitted total", "fitted 1 (LXe-PTFE)", "fitted 2 (LXe-gas)"], colors=["k","r","b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 30...")
    plot_double_TSTR_fit(45., n, params, colors=["k", "r", "b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 45...")
    plot_double_TSTR_fit(52., n, params, colors=["k", "r", "b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 52...")
    plot_double_TSTR_fit(60., n, params, colors=["k", "r", "b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 60...")
    plot_double_TSTR_fit(67., n, params, colors=["k","r","b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 67...")
    plot_double_TSTR_fit(75., n, params, colors=["k", "r", "b"], average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
    #print("done with 75...")

    #plt.text(0.05,0.1,r"Fit: $\rho_L$={0:.3f}, n={1:.2f}, $\gamma$={2:.3f}, K={3:.3f}, $\sigma (\theta_i)$={4:.3f}".format(*params),transform=plt.gca().transAxes,fontsize=11)
    plt.show()
    get_input=input("New parameters? (y/n) ").lower()=='y'
