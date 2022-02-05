import numpy as np
import matplotlib.pyplot as plt
from BTinPython import is_pos_sdef, calc_constrM
from new_functions import *
import time
from tqdm import tqdm

# Boundary Search for all K in K_arr, but leaving the frame fixed
def boundary_in_fixed_frame(N, K_arr, E_start, E_stop, exp_x2_start, exp_x2_stop, h1, h2, f = 3, max_iterations = 5*10**3, tol = 1e-8, m=1/2, w=2, g=1, double_well=False):
    '''
    Parameters
    ----------
    N : int
        Energy level, only used for in plotting and file name.
    K_arr : list of int
        Boundary search starts at K_arr[0] and goes to K_arr[-1].
    E_start : float
        Search frame E axis start value.
    E_stop : float
        Search frame E axis stop value.
    exp_x2_start : float
        Search frame exp x2 axis start value.
    exp_x2_stop : float
        Search frame exp x2 axis stop value.
    h1 : float
        Step size for E axis.
    h2 : float
        Step size for exp x2 axis.
    f : int
        Parameter determining how coarse the starting point search is. The default is 3.
    max_iterations : float
        Maximum number of central points the algorithms goes through for a given K. The default is 5*10**3.
    tol : float
        Tolerance of the bootstrapping algorithm. The default is 1e-8.
    m : float
        Parameter of the Hamiltonian. The default is 1/2.
    w : float
        Parameter of the Hamiltonian. The default is 2.
    g : float
        Parameter of the Hamiltonian. The default is 1.
    double_well : boolean
        Chooses double well or anharmonic oscillator. The default is False.

    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots()
    E_arr = np.arange(E_start, E_stop, h1)
    exp_x2_arr = np.arange(exp_x2_start, exp_x2_stop, h2)
    print(f'Number of grid points: {len(E_arr)*len(exp_x2_arr)}')
    for K in K_arr:
        print(f'K = {K}')
        start1 = time.time()
        start_point, success = find_start_point(K, f, E_arr, exp_x2_arr, tol, m, w, g, double_well)
        if not success:
            break
        region_boundary, success = find_region_boundary(h1, h2, K, E_arr, exp_x2_arr, start_point, tol, max_iterations, m, w, g, double_well)
        end1 = time.time()
        t = round(end1-start1)
        c = range(len(region_boundary[:,0]))
        ax.scatter(region_boundary[:,0], region_boundary[:,1], c=c, marker='.', label=f'n={N}, {K=}, {t=}s')
        print(f'Number of boundary points: {len(region_boundary)}')
        print(f'Bootstrapping along region boundary for K = {K} took {t} seconds.')
    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$\langle \hat{x}^2 \rangle$')
    ax.set_xlim(E_start, E_stop)
    ax.set_ylim(exp_x2_start, exp_x2_stop)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'plots/boundary_n={N}_{K_arr=}.png', dpi=200)
    plt.show()

# Full Grid Search (normal Bootstrapping) in fixed frame
def region_in_fixed_frame(N, K_arr, E_start, E_stop, exp_x2_start, exp_x2_stop, h1, h2, tol = 1e-8, m=1/2, w=2, g=1, double_well=False):
    '''
    Parameters
    ----------
    N : int
        Energy level, only used for in plotting and file name.
    K_arr : list of int
        Boundary search starts at K_arr[0] and goes to K_arr[-1].
    E_start : float
        Search frame E axis start value.
        Search frame E axis stop value.
    exp_x2_start : float
        Search frame exp x2 axis start value.
    exp_x2_stop : float
        Search frame exp x2 axis stop value.
    h1 : float
        Step size for E axis.
    h2 : float
        Step size for exp x2 axis.
    tol : float
        Tolerance of the bootstrapping algorithm. The default is 1e-8.
    m : float
        Parameter of the Hamiltonian. The default is 1/2.
    w : float
        Parameter of the Hamiltonian. The default is 2.
    g : float
        Parameter of the Hamiltonian. The default is 1.
    double_well : boolean
        Chooses double well or anharmonic oscillator. The default is False.

    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots()
    E_arr = np.arange(E_start, E_stop, h1)
    exp_x2_arr = np.arange(exp_x2_start, exp_x2_stop, h2)
    print(f'Number of grid points: {len(E_arr)*len(exp_x2_arr)}')
    for K in K_arr:
        start = time.time()
        print(f'K = {K}')
        region = np.empty((0,2))
        for E in tqdm(E_arr):
            for exp_x2 in exp_x2_arr:
                point = [E, exp_x2]
                if is_pos_sdef(calc_constrM(point, K, m, w, g, double_well), tol):
                    region = np.vstack((region, point))
        end = time.time()
        t = round(end-start)
        print(f'Bootstrapping full region for K = {K} took {t} seconds.')
        ax.scatter(region[:,0], region[:,1], marker='.', label=f'n={N}, {K=}, {t=}s')
    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$\langle \hat{x}^2 \rangle$')
    ax.set_xlim(E_start, E_stop)
    ax.set_ylim(exp_x2_start, exp_x2_stop)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'plots/region_n={N}_{K_arr=}.png', dpi=200)
    plt.show()

# Boundary Search in a shrinking frame, which is able to work on bigger ranges of K
def boundary_in_shrinking_frame(N, K_arr, E_start, E_stop, exp_x2_start, exp_x2_stop, f=3, max_iterations=5e3, check=False, tol=1e-8, s=10, scale=1.5e3, m=1/2, w=2, g=1, double_well=False):
    '''
    Parameters
    ----------
    N : int
        Energy level, only used for in plotting and file name.
    K_arr : list of int
        Boundary search starts at K_arr[0] and goes to K_arr[-1].
    E_start : float
        Search frame E axis start value.
    E_stop : float
        Search frame E axis stop value.
    exp_x2_start : float
        Search frame exp x2 axis start value.
    exp_x2_stop : float
        Search frame exp x2 axis stop value.
    f : int
        Parameter determining how coarse the starting point search is. The default is 3.
    max_iterations : float
        Maximum number of central points the algorithms goes through for a given K. The default is 5*10**3.
    check : boolean
        Checks if region at given K was found successfully. Not doing this check is useful, because the boundary search does not have to complete, to give a good restriction of the search frame. The default is False.
    tol : float
        Tolerance of the bootstrapping algorithm. Setting tol=None leads to automatic tolerance adjustment and is needed for the Full Method. The default is 1e-8.
    s : int
        Increasing the search frame minimally in the end so that everything is inside the search frame. The default is 10.
    scale : float
        Scaling of steps h1 and h2. The default is 1.5e3.
    m : float
        Parameter of the Hamiltonian. The default is 1/2.
    w : float
        Parameter of the Hamiltonian. The default is 2.
    g : float
        Parameter of the Hamiltonian. The default is 1.
    double_well : boolean
        Chooses double well or anharmonic oscillator. The default is False.

    Returns
    -------
    list
        Search frame and K at highest K.
    '''
    success = True
    counter = 0
    region_boundary = np.empty((0,2))
    h1 = cal_h(E_start, E_stop, scale)
    h2 = cal_h(exp_x2_start, exp_x2_stop, scale)
    print(f'Maximum number of iterations: {max_iterations}')
    while success and counter < len(K_arr):
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$E$')
        ax.set_ylabel(r'$\langle \hat{x}^2 \rangle$')
        K = int(K_arr[counter])
        # Automatic adjustment of tol, needed when using values from energy_levels()
        if not tol:
            tol = 1e-2*10**(-counter)
            if tol < 1e-8:
                tol = 1e-8
        start = time.time()
        print(f'K = {K}')
        E_arr = np.arange(E_start, E_stop, h1)
        exp_x2_arr = np.arange(exp_x2_start, exp_x2_stop, h2)
        print(f'Number of grid points: {len(E_arr)*len(exp_x2_arr)}')
        start_point, success = find_start_point(K, f, E_arr, exp_x2_arr, tol, m, w, g, double_well)
        if not success:
            print('no start_point')
            break
        region_boundary, success = find_region_boundary(h1, h2, K, E_arr, exp_x2_arr, start_point, tol, max_iterations, m, w, g, double_well)
        if not check:
            success = True
        print(f'Number of boundary points: {len(region_boundary)}')
        if success:
            E_start = np.min(region_boundary[:,0]) - s*h1
            E_stop = np.max(region_boundary[:,0]) + s*h1
            exp_x2_start = np.min(region_boundary[:,1]) - s*h2
            exp_x2_stop = np.max(region_boundary[:,1]) + s*h2
            h1 = cal_h(E_start, E_stop, scale)
            h2 = cal_h(exp_x2_start, exp_x2_stop, scale)
        end = time.time()
        if success:
            t = round(end-start)
            print(f'Bootstrapping along region boundary for K = {K} took {t} seconds.')
        else:
            t = 9999
        print('')
        c = range(len(region_boundary[:,0]))
        ax.scatter(region_boundary[:,0], region_boundary[:,1], c=c, marker='.', label=f'n={N}, {K=}, {t=}s')
        ax.set_xlim(E_start - 100*h1, E_stop + 100*h1)
        ax.set_ylim(exp_x2_start - 100*h2, exp_x2_stop + 100*h2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'plots/boundary_n={N}_{K=}.png', dpi=200)
        counter += 1
    print(f'Search ended at K = {K}.')
    E_start = np.min(region_boundary[:,0]) - s*h1
    E_stop = np.max(region_boundary[:,0]) + s*h1
    exp_x2_start = np.min(region_boundary[:,1]) - s*h2
    exp_x2_stop = np.max(region_boundary[:,1]) + s*h2
    return [E_start, E_stop, exp_x2_start, exp_x2_stop, K]

# Energy Level Approximation
def energy_levels(N, E_start_i=0, E_stop_i=6, exp_x2_start_i=0, exp_x2_stop_i=3, tol=1e-2, scale=1.5e3, m=1/2, w=2, g=1, double_well=False):
    '''
    Parameters
    ----------
    N : int
        Number of energy levels to find.
    E_start_i : float
        Initial start value of search frame E axis. The default is 0.
    E_stop_i : float
        Initial stop value of search frame E axis. Should be bigger than the maximum E distance between two energy levels. The default is 6.
    exp_x2_start_i : float
        Initial start value of search frame exp x2 axis. The default is 0.
    exp_x2_stop_i : float
        Initial stop value of search frame exp x2 axis. Should be bigger than the maximum exp x2 distance between two energy levels. The default is 3.
    tol : float
        Tolerance of the bootstrapping algorithm. "Big" value of tol=1e-2 is necessary to make regions more smooth. The default is 1e-2.
    scale : float
        Scaling of steps h1 and h2. The default is 1.5e3.
    m : float
        Parameter of the Hamiltonian. The default is 1/2.
    w : float
        Parameter of the Hamiltonian. The default is 2.
    g : float
        Parameter of the Hamiltonian. The default is 1.
    double_well : boolean
        Chooses double well or anharmonic oscillator. The default is False.

    Returns
    -------
    E_tuple : list of float
        Min and Max of E for given K and n.
    x_tuple : list of float
        Min and Max of exp x2 for given K and n.
    K_list : list of int
        K.
    N_list : list of int
        n (energy level).
    '''
    start = time.time()
    print(f'Searching energy levels 0 to {N-1}')
    K = 2
    E_tuple = np.empty((0,2))
    x_tuple = np.empty((0,2))
    K_list = np.empty((0))
    N_list = np.empty((0))
    for level in range(N):
        print(f'\nEnergy level {level}')
        con = True
        break_b = False
        if level == 0:
            E_start, E_stop = [E_start_i, E_stop_i]
            exp_x2_start, exp_x2_stop = [exp_x2_start_i, exp_x2_stop_i]
        print("Searching E interval")
        while con:
            lower = 0
            upper = 0
            E_old = 0
            col_false = 0
            h1 = cal_h(E_start, E_stop, scale)*10
            h2 = cal_h(exp_x2_start, exp_x2_stop, scale)
            E_arr = np.arange(E_start, E_stop, h1)
            exp_x2_arr = np.arange(exp_x2_start, exp_x2_stop, h2)
            print(f'\n{K=}')
            for E in tqdm(E_arr, position=0):
                if break_b == True:
                    break
                count_false = 0
                for exp_x2 in exp_x2_arr:
                    point = [E, exp_x2]
                    switch = is_pos_sdef(calc_constrM(point, K, m, w, g, double_well), tol)
                    if switch == True:
                        col_false = 0
                    if switch == True and lower == 0:
                        lower = E_old
                        break
                    if switch == False and lower != 0:
                        count_false += 1
                    if count_false == len(exp_x2_arr) and E > lower:
                        col_false += 1
                    if col_false == 40 and E > lower:
                        upper = E - 39*h1
                        break_b = True
                        break
                E_old = E
            print(f'\nE_start = {lower}')
            print(f'E_stop = {upper}')
            if upper == 0:
                K += 1
                if lower > 0:
                    E_start = lower
                elif lower == 0:
                    print('\nSearch did not complete!')
                    break
            elif lower != 0 and upper != 0:
                E_tuple = np.vstack((E_tuple, [lower, upper]))
                K_list = np.append(K_list, K)
                N_list =  np.append(N_list, level)
                con = False
        E_start, E_stop = [lower, upper]
        E_arr = np.arange(E_start, E_stop, h1)
        lower = 0
        upper = 0
        x_old = 0
        break_b = False
        col_false = 0
        print("\nSearching exp_x2 interval")
        for exp_x2 in tqdm(exp_x2_arr):
            count_false = 0
            if break_b == True:
                break
            for E in E_arr:
                point = [E, exp_x2]
                switch = is_pos_sdef(calc_constrM(point, K, m, w, g, double_well), tol)
                if switch == True:
                    col_false = 0
                if switch == True and lower == 0:
                    lower = x_old
                    break
                if switch == False and lower != 0:
                    count_false += 1
                if count_false == len(E_arr) and exp_x2 > lower:
                    col_false += 1
                if col_false == 100 and exp_x2 > lower:
                    upper = exp_x2 - 99*h2
                    break_b = True
                    break
            x_old = exp_x2
        K += 1
        print(f'\nexp_x2_start = {lower}')
        print(f'exp_x2_stop = {upper}')
        if lower == 0 or upper == 0:
            print('\nSearch did not complete!')
            break
        x_tuple = np.vstack((x_tuple, [lower, upper]))
        exp_x2_start, exp_x2_stop = [upper, exp_x2_stop_i + upper]
        lower = E_tuple[-1][0]
        upper = E_tuple[-1][1]
        E_start, E_stop = [upper, E_stop_i + upper]
        print("\nNew search frame:")
        print(f'{E_start=}, {E_stop=}')
        print(f'{exp_x2_start=}, {exp_x2_stop=}')
    end = time.time()
    t = round(end-start)
    print(f'\nSearching energy levels 0 to {N-1} took {t} seconds.')
    print('\nResults:')
    for i in range(N):
        print(f'{E_tuple[i]}, {x_tuple[i]}, {K_list[i]}, {N_list[i]}')
    return E_tuple, x_tuple, K_list, N_list

# print function to give output as latex table
def print_latex_table(result):
    print('''\\begin{table}[!htbp]
    \\centering
    \\begin{tabular}{cccccc}
    \\toprule
    Energy Level & $E_{\\text{min}}$ & $E_{\\text{max}}$ & $\expv{\hat{x}^2}_{\\text{min}}$ & $\expv{\hat{x}^2}_{\\text{max}}$ & $K$ \\\\
    \\midrule''')

    for i, temp in enumerate(result):
        diff_E = (temp[0] - temp[1])
        com_E = str(format(diff_E*0.1, 'e'))[-1]
        diff_x = (temp[2] - temp[3])
        com_x = str(format(diff_x*0.1, 'e'))[-1]
        print(f'{i} & {temp[0]:.{com_E}f} & {temp[1]:.{com_E}f} & {temp[2]:.{com_x}f} & {temp[3]:.{com_x}f} & {temp[4]} \\\\')

    print('''\\bottomrule
    \\end{tabular}
\end{table}''')
