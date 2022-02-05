from computing_and_visualisation import boundary_in_fixed_frame, region_in_fixed_frame, boundary_in_shrinking_frame, energy_levels, print_latex_table
from new_functions import cal_h

''' Please create a folder named "plots" inside the working directory! '''

#%%
''' Hamiltonian Parameters '''
m=1/2
w=2
g=1

#%%
''' Boundary Search ''' 

''' Double Well - Energy Level 0 '''
double_well = True
N = 0 # here only relevant for plotting
K_arr = [8]
E_start = 1.82
E_stop = 1.94
exp_x2_start = 0.39
exp_x2_stop = 0.48

''' Anharmonic Oscillator - Energy Level 0 '''
# double_well = False
# N = 0
# K_arr = [7]
# E_start = 1.34
# E_stop = 1.44
# exp_x2_start = 0.293
# exp_x2_stop = 0.312

# spacings in E and exp x2 direction
h1 = cal_h(E_start, E_stop, scale = 1.5e3)
h2 = cal_h(exp_x2_start, exp_x2_stop, scale = 1.5e3)

''' Choose one of three methods '''
# region_in_fixed_frame(N, K_arr=K_arr, E_start=E_start, E_stop=E_stop, exp_x2_start = exp_x2_start, exp_x2_stop = exp_x2_stop, h1=h1, h2=h2, double_well=double_well, tol=1e-8)
boundary_in_fixed_frame(N, K_arr=K_arr, E_start=E_start, E_stop=E_stop, exp_x2_start=exp_x2_start, exp_x2_stop=exp_x2_stop, h1=h1, h2=h2, double_well=double_well, tol=1e-8, max_iterations=5e3, f=1)
# boundary_in_shrinking_frame(N, K_arr=K_arr, E_start=E_start, E_stop=E_stop, exp_x2_start=exp_x2_start, exp_x2_stop=exp_x2_stop, double_well=double_well, f=1, max_iterations=5e3, tol=1e-8)

#%%
''' Energy Level Approximation '''
double_well = True
N = 3 # number of energy levels to find
if double_well:
    E_diff = 6
    x_diff = 3
elif not double_well:
    E_diff = 6
    x_diff = 1
E_tuple, x_tuple, K_list, N_list = energy_levels(N, E_stop_i=E_diff, exp_x2_stop_i=x_diff, m=m, w=w, g=g, double_well=double_well)

#%%
''' Full Methode '''
double_well = True
N = 3 # number of energy levels to find
K_max = 17
if double_well:
    E_diff = 6
    x_diff = 3
elif not double_well:
    E_diff = 6
    x_diff = 1
E_tuple, x_tuple, K_list, N_list = energy_levels(N, E_stop_i=E_diff, exp_x2_stop_i=x_diff, m=m, w=w, g=g, double_well=double_well)
pre_result = []
result = []
for i in range(N):
    E_start, E_stop = E_tuple[i]
    exp_x2_start, exp_x2_stop = x_tuple[i]
    K = int(K_list[i])
    pre_result.append([E_start, E_stop, exp_x2_start, exp_x2_stop, K])
    n = int(N_list[i])
    if len(range(K, K_max)) > 0:
        result_temp = boundary_in_shrinking_frame(n, K_arr=range(K, K_max), E_start=E_start, E_stop=E_stop, exp_x2_start=exp_x2_start, exp_x2_stop=exp_x2_stop, check=False, m=m, w=w, g=g, double_well=double_well, tol=None, scale=3e3, max_iterations=1e4, f=1)
        result.append(result_temp)
print('Table for Energy Level Function: \n')
print_latex_table(pre_result)
print('\n')
print('Table for full Calculation: \n')
print_latex_table(result)
