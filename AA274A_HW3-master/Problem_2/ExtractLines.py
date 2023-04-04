#!/usr/bin/env python3

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
from posixpath import split
from re import I
from uuid import RFC_4122
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    print("LB", LineBreak)
    for endIdx in LineBreak:
        print("start", startIdx)
        print("endi", endIdx)
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, 
                                                             rho, 
                                                             (startIdx, endIdx), 
                                                             params['MIN_POINTS_PER_SEGMENT'],
                                                             params['LINE_POINT_DIST_THRESHOLD'])
        N_lines = r_seg.size

        # print("alpha_seg", alpha_seg )
        # print("r_seg", r_seg)
        # print("pointIdx_seg",pointIdx_seg)
        # print("N_lines", N_lines)

        ## Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

        print("N_lines", N_lines)
        # print("alpha", alpha )
        # print("r", r)
        print("pointIdx",pointIdx)

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, indices, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        indices: tuple with (idxStart, idxEnd) representing the starting
                and ending indices of the segment to be split
        MIN_POINTS_PER_SEGMENT: parameter describing min pts per segement.
        LINE_POINT_DIST_THRESHOLD: parameter describing the max distance a point
            can be from a line before the line is split
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########

    # print("recursion call")

    a = indices[0]
    b = indices[1]
    alpha, r = FitLine(theta[a:b],rho[a:b])

    # print("alpha", alpha)
    # print("r", r)
    # print("ind", indices)

    if (b - a) <= MIN_POINTS_PER_SEGMENT:
        return alpha, r, (a, b)

    # find split index
    s = FindSplit(theta[a:b],rho[a:b],alpha,r, MIN_POINTS_PER_SEGMENT,LINE_POINT_DIST_THRESHOLD)

    # s not found 
    if s == -1:
        return alpha, r, (a, b)

    r = np.zeros(0)
    alpha = np.zeros(0)
    idx = np.zeros((0, 2), dtype=int)

    # print("first line")
    # print("a + s", a+s)
    alpha1, r1, idx1 = SplitLinesRecursive(theta,rho, (a, a + s), MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD)
    
    r = np.append(r, r1)
    alpha = np.append(alpha, alpha1)
    idx = np.vstack((idx, idx1))
    # print("added line 1")
    # print(alpha)

    # print("second line")
    # print("a + s", a+s)
    alpha2, r2, idx2 = SplitLinesRecursive(theta,rho, (a + s, b), MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD)
    r = np.append(r, r2)
    alpha = np.append(alpha, alpha2)
    idx = np.vstack((idx, idx2))
    # print("added line2")
    # print(alpha)

    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. You should check if any point exists that satisfies both criteria.
    Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        MIN_POINTS_PER_SEGMENT: parameter describing min pts per segement.
        LINE_POINT_DIST_THRESHOLD: parameter describing the max distance a point
            can be from a line before the line is split
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).

    Hint: the output is the index of the first element in the second line. 
    For instance, if theta = np.array([0,1,2,3,4,5,6]) and splitIdx = 3, 
    then the two line segments will be 
        theta1 = theta[0:splitIdx] = np.array([0,1,2]) and 
        theta2 = theta[splitIdx:] = np.array([3,4,5,6]).
    '''
    ########## Code starts here ##########

    # The best point to split at is the one whose distance from the line is
    # the farthest, as long as this maximum distance exceeds
    # LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    # smaller than MIN_POINTS_PER_SEGMENT. You should check if any point exists that satisfies both criteria.
    # Otherwise, no split should be made.

    max_distance = 0
    N = len(theta)
    splitIdx = -1

    # distance = np.absolute(rho*np.cos(theta - alpha) - r)
    # print("distance", distance)
    # i = np.argmax(distance)
    # max_distance = distance[i]
    # if max_distance > LINE_POINT_DIST_THRESHOLD:
    #     if len(theta[0:i]) > MIN_POINTS_PER_SEGMENT and len(theta[i:]) > MIN_POINTS_PER_SEGMENT:
    #         splitIdx = i

    for i in range(N):
        distance = np.abs(rho[i]*np.cos(theta[i] - alpha) - r)
        if distance > LINE_POINT_DIST_THRESHOLD and len(theta[0:i]) >= MIN_POINTS_PER_SEGMENT and len(theta[i:]) >= MIN_POINTS_PER_SEGMENT:
            if distance > max_distance:
                max_distance = distance
                splitIdx = i

    ########## Code ends here ##########
    return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads). Should be between -pi and pi.
        r: 'r' of best fit for range data (1 number) (m). Should be positive.
    '''
    ########## Code starts here ##########
    s2 = 0
    c2 = 0
    cs = 0
    cij = 0
    r = 0
    alpha = 0

    N = len(theta)
    # s2 = np.sum((rho**2)*np.sin(2*theta))
    # c2 = (2/N)*np.sum((rho**2)*np.cos(2*theta))
    for i in range(N):
        s2 += (rho[i]**2)*np.sin(2*theta[i])
        c2 += (rho[i]**2)*np.cos(2*theta[i])
        for j in range(N):
            cs += rho[i]*rho[j]*np.cos(theta[i])*np.sin(theta[j])
            cij += rho[i]*rho[j]*np.cos(theta[i] + theta[j])

    alpha = 0.5*np.arctan2(s2 - (2/N)*cs,c2 - (1/N)*cij) + np.pi/2

    # print("aaa",alpha)

    # for i in range(N):
    #     r += rho[i]*np.cos(theta[i] - alpha)

    # r *= (1/N)

    r = (1/N)*np.sum(rho*np.cos(theta - alpha))

    # print("rr", r)

    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########

    # rOut = np.zeros(0)
    # alphaOut = np.zeros(0)
    # pointIdxOut = np.zeros((0, 2), dtype=int)

    #N = len(pointIdx)
    i = 0
    j = len(pointIdx) - 1
    while i < j:
        p1 = pointIdx[i]
        a1 = p1[0]
        b1 = p1[1]
        p2 = pointIdx[i+1]
        a2 = p2[0]
        b2= p2[1]
        alpha1, r1 = FitLine(theta[a1:b2],rho[a1:b2])
        s = FindSplit(theta[a1:b2],rho[a1:b2],alpha1,r1, params['MIN_POINTS_PER_SEGMENT'],params['LINE_POINT_DIST_THRESHOLD'])
        if s == -1:
            #merge
            p_first = pointIdx[0:i]
            p_new = np.array([a1,b2])
            p_end = pointIdx[i+2:]
            pointIdx = np.vstack((p_first,p_new,p_end))
            alpha[i] = alpha1
            alpha = np.delete(alpha,i+1)
            r = np.delete(r,i+1)
            r[i] = r1
        else:
            # no merge
            i += 1
        j = len(pointIdx) - 1

    rOut = r
    alphaOut = alpha
    pointIdxOut = pointIdx

    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 4  # minimum number of points per line segment
    MAX_P2P_DIST = 1.0  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    # filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show()

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
