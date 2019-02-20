import matplotlib.pyplot as plt
import numpy as np
import time

# this is a model for the fresnel factor which uses a uniform range of n values, from n - n_range / 2 to n + n_range / 2
# the range is split into two parts, the part which does not totally internally reflect, and the part which does
# the part that does totally internally reflect is simple to calculate perfectly
# the part that does not is approximated by being represented by the center n value for the range

n_range = 0.14

# assumes unpolarized light
# takes in a masked array only including where total internal reflection doesn't happen
# uses algebra to avoid taking the sin and cosines of an arcsin
def F_single(theta_i, n_0, n):
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    n_ratio = n_0 / n
    s_reflection = np.ma.power((n_0 * c - n * np.real(np.emath.sqrt(1 - n_ratio * n_ratio * s * s))) / 
                               (n_0 * c + n * np.real(np.emath.sqrt(1 - n_ratio * n_ratio * s * s))), 2)
    p_reflection = np.ma.power((n_0 * np.real(np.emath.sqrt(1 - n_ratio * n_ratio * s * s)) - n * c) / 
                               (n_0 * np.real(np.emath.sqrt(1 - n_ratio * n_ratio * s * s)) + n * c), 2)
    
    return 0.5 * (s_reflection + p_reflection)

def F_single_no_internal(theta_i, n_0, n):
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    n_ratio = n_0 / n
    s_reflection = np.ma.power((n_0 * c - n * np.sqrt(1 - n_ratio * n_ratio * s * s)) / 
                               (n_0 * c + n * np.sqrt(1 - n_ratio * n_ratio * s * s)), 2)
    p_reflection = np.ma.power((n_0 * np.sqrt(1 - n_ratio * n_ratio * s * s) - n * c) / 
                               (n_0 * np.sqrt(1 - n_ratio * n_ratio * s * s) + n * c), 2)
    
    return 0.5 * (s_reflection + p_reflection)

# takes in an array of theta_i values, doesn't actually use polarization values, assumes unpolarized
def F(theta_i, n_0, n, polarization=0.5):
    n_crit = n_0 * np.sin(theta_i)
    # if n is less than n_crit, then there is TIR

    n_min_tir = n - n_range / 2.
    n_max_non_tir = n + n_range / 2.
    n_boundary_tir = np.maximum(n_min_tir, np.minimum(n_crit, n_max_non_tir)) # ideally n_crit, but forced to be bounded by the n range
    n_center_no_tir = (n_max_non_tir + n_boundary_tir) / 2.
    
    tir_frac = (n_boundary_tir - n_min_tir) / n_range # gives fraction of n values which cause TIR

    mask = np.abs(n_0 / n_center_no_tir * np.sin(theta_i)) > 1 # masks the theta_i values that will totally internally reflect with n=n_center
    theta_i_mask = np.ma.masked_array(theta_i, mask) # list of theta_i values that don't totally internally reflect at n_center

    non_tir_frac = 1 - tir_frac
    non_tir_array_masked = np.multiply(F_single_no_internal(theta_i, n_0, n_center_no_tir), non_tir_frac) # multiplies fresnel factor by fraction of n values represented by n_center
    non_tir_array = np.ma.filled(non_tir_array_masked, fill_value=0.) # the non-TIR contribution to each theta_i
    
    return np.add(tir_frac, non_tir_array)

# takes in an array of theta_i values, doesn't actually use polarization values, assumes unpolarized
def F_not_masked(theta_i, n_0, n, polarization=0.5):
    n_crit = n_0 * np.sin(theta_i)
    # if n is less than n_crit, then there is TIR

    n_min_tir = n - n_range / 2.
    n_max_non_tir = n + n_range / 2.
    n_boundary_tir = np.maximum(n_min_tir, np.minimum(n_crit, n_max_non_tir)) # ideally n_crit, but forced to be bounded by the n range
    n_center_no_tir = (n_max_non_tir + n_boundary_tir) / 2.
    
    tir_frac = (n_boundary_tir - n_min_tir) / n_range # gives fraction of n values which cause TIR

    tir_doesnt_happen = np.abs(n_0 / n_center_no_tir * np.sin(theta_i)) < 1 # true where TIR doesn't happen

    non_tir_frac = 1. - tir_frac
    non_tir_array = np.ma.filled(np.multiply(np.multiply(F_single_no_internal(theta_i, n_0, n_center_no_tir), non_tir_frac), tir_doesnt_happen), fill_value=0) # multiplies fresnel factor by fraction of n values represented by n_center
    return np.add(tir_frac, non_tir_array)

theta_i_array_deg = np.arange(85.)
theta_i_array = [x * np.pi / 180. for x in theta_i_array_deg]

t0=time.time()
f_array = F(theta_i_array, 1.69, 1.5)
t1=time.time()
f_array_old = F_single(theta_i_array, 1.69, 1.5)
t2=time.time()
f_array_not_masked = F_not_masked(theta_i_array, 1.69, 1.5)
t3=time.time()

print("Old calculation time: {0}".format(t2-t1))
print("New calculation time: {0}".format(t1-t0))
print("New non masked  time: {0}".format(t3-t2))

plt.plot(theta_i_array, f_array, label="range of n values")
plt.plot(theta_i_array, f_array_old, label="one n value")
plt.plot(theta_i_array, f_array_not_masked, label="range, not masked")

plt.legend()
plt.show()

