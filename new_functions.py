import numpy as np
from BTinPython import is_pos_sdef, calc_constrM

# finds start point of region, f parameter determines how coarse used grid is
def find_start_point(K, f, E_arr, exp_x2_arr, tol, m=1/2, w=2, g=1, double_well=False):
    i = 0
    k = 0
    switch = False
    while not switch and i < K/f:
        exp_x2 = exp_x2_arr[i*int(f*len(exp_x2_arr) / K)]
        for E in E_arr:
            switch = is_pos_sdef(calc_constrM([E, exp_x2], K, m, w, g, double_well), tol)
            if switch == True:
                start_point = np.array([E, exp_x2])
                break
            k += 1
        i += 1
    print(f"In search of a starting point {k} points were checked.")
    if not switch:
        print("No starting point found")
        success = False
        start_point = np.array([])
    elif switch:
        success = True
    return start_point, success

# calculates 8 neighbouring points, where r determines the distance to the central point
def find_neighbours(h1, h2, E_arr, exp_x2_arr, point, r):
    h1 = r*h1
    h2 = r*h2
    neighbours = np.array([[point[0], point[1] + h2],
                           [point[0] - h1, point[1] + h2],
                           [point[0] - h1, point[1]],
                           [point[0] - h1, point[1] - h2],
                           [point[0], point[1] - h2],
                           [point[0] + h1, point[1] - h2],
                           [point[0] + h1, point[1]],
                           [point[0] + h1, point[1] + h2]])
    return neighbours

# check if point is in array and delete it
def delete_point_from_array(arr, point):
    for i, e in enumerate(arr):
        if np.array_equal(e, point):
            arr = np.delete(arr, i, axis=0)
    return arr

# given a point and its neighbours determine the next point to use as central point
def next_point(middle_point, last_middle_point, other_points, ToF):
    dist_now = (((other_points - middle_point)**2) / 2).sum(1)
    dist_last = (((other_points - last_middle_point)**2) / 2).sum(1)
    calc = dist_now + 2*dist_last
    arg = np.argmax(calc)
    dist = np.max(calc)
    return other_points[arg], dist

# check if neighbours are True/False
def check_neighbours(h1, h2, K, E_arr, exp_x2_arr, start_point, tol, ToF, r, m=1/2, w=2, g=1, double_well=False):
    neighbours = np.empty((0, 2))
    for i in range(1, r+1):
        neighbours = np.vstack((neighbours, find_neighbours(h1, h2, E_arr, exp_x2_arr, start_point, i)))
    neighbours_ToF = np.empty((0, 2))
    for point in neighbours:
        check = is_pos_sdef(calc_constrM(point, K, m, w, g, double_well), tol)
        if check == ToF:
            neighbours_ToF = np.vstack((neighbours_ToF, point))
    return neighbours_ToF

# find hole region boundary from start point
def find_region_boundary(h1, h2, K, E_arr, exp_x2_arr, start_point, tol, max_iterations, m=1/2, w=2, g=1, double_well=False):
  region_boundary = np.array([start_point])
  start_neighbours_true = np.vstack((check_neighbours(h1, h2, K, E_arr, exp_x2_arr, start_point, tol, True, 1), start_point))
  middle_point = start_point
  last_middle_point = start_point
  ToF = False
  switch = True
  counter = 0
  while switch:
    r = 1
    if counter >= max_iterations:
        status = False
        switch = False
    for point in start_neighbours_true:
        if np.array_equal(middle_point, point) and counter > 10:
                print('Region complete')
                status = True
                switch = False
    neighbours_ToF = check_neighbours(h1, h2, K, E_arr, exp_x2_arr, middle_point, tol, ToF, r, m, w, g, double_well)
    var = middle_point
    for point in region_boundary[-6:-2]:
        neighbours_ToF = delete_point_from_array(neighbours_ToF, point)
    a_count = 0
    while not np.any(neighbours_ToF) and a_count < 200:
        r += 1
        neighbours_ToF = check_neighbours(h1, h2, K, E_arr, exp_x2_arr, var, tol, ToF, r, m, w, g, double_well)
        for point in region_boundary[-6:-2]:
            neighbours_ToF = delete_point_from_array(neighbours_ToF, point)
        a_count += 1
    if not a_count < 200:
        status = False
        print('To many adjustments. Region is assumed to be a line. All results should manually checked for correctness.')
        return region_boundary, status
    middle_point, dist = next_point(var, last_middle_point, neighbours_ToF, ToF)
    if counter%1000 == 0:
        print(counter, middle_point)
    last_middle_point = var
    if ToF:
      region_boundary = np.vstack((region_boundary, neighbours_ToF))
    ToF = not ToF
    counter += 1
  return region_boundary, status

# calculate the spacing between two grid values of E or exp_x2 from the min and max value of the array
def cal_h(start, stop, scale = 1.5e3):
    h = (stop - start)/scale
    return h
