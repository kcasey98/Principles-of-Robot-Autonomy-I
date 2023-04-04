from os import PRIO_PGRP
import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions.
        #       When should each be called? Make use of self.t_before_switch and
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        
        t_final = self.traj_controller.traj_times[-1]
        if t >= (t_final - self.t_before_switch):
            # print("pose", self.pose_controller.compute_control(x,y,t,th))
            V, m = self.pose_controller.compute_control(x,y,t,th)
            return V, m
        else:
            # print("traj", self.traj_controller.compute_control(x,y,th,t))
            return self.traj_controller.compute_control(x,y,th,t)

        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    # assert(path and k > 2 and k < len(path))

    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.

    N = len(path)
    # time = np.zeros(N)
    time = [0]
    x = [path[0][0]]
    y = [path[0][1]]

    # print("path", path[0])
    # print("diff", np.linalg.norm([path[1][0] - path[0][0], path[1][1] - path[0][1]]))
    # print("len(path)", len(path))

    for i in range(1,N):
        d = np.linalg.norm([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
        time.append((1/V_des)*d + time[i-1])
        # time[i] = (1/V_des)*d + time[i-1]
        x.append(path[i][0])
        y.append(path[i][1])

    print("len", len(time))
    print("x",x)
    print("y",y)

    t_smoothed = np.arange(0,time[-1],dt)
    x_spl = scipy.interpolate.splrep(time, x, k = k, s= alpha)
    y_spl = scipy.interpolate.splrep(time, y, k = k, s= alpha)

    x_d = scipy.interpolate.splev(t_smoothed,x_spl,der=0)
    y_d = scipy.interpolate.splev(t_smoothed,y_spl,der=0)

    xd_d = scipy.interpolate.splev(t_smoothed,x_spl,der=1)
    yd_d = scipy.interpolate.splev(t_smoothed,y_spl,der=1)
    theta_d = np.arctan2(yd_d,xd_d)

    xdd_d = scipy.interpolate.splev(t_smoothed,x_spl,der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed,y_spl,der=2)


    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj()
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########

    # final timex
    tf = float(t[-1])

    V, om = compute_controls(traj=traj)
    V_tilde = rescale_V(V,om, V_max, om_max)
    om_tilde = rescale_om(V,om,V_tilde)
    s = compute_arc_length(V,t)
    tau = compute_tau(V_tilde,s)

    # s_f = State(x=traj[-1][0], y=traj[-1][1], V= np.sqrt(traj[-1][3]**2 + traj[-1][4]**2), th=traj[-1][2])
    s_f = State(x=traj[-1][0], y=traj[-1][1], V=V_max, th=traj[-1][2])

    return interpolate_traj(traj,tau,V_tilde,om_tilde,dt,s_f)

