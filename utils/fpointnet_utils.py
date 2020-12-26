# -*- coding: utf-8 -*-
"""
F-Pointnet utils
"""

# imports
import numpy as np
from shapely.geometry import Polygon


# methods

def inter_overlap(I1, I2):
    # given 2 intervals I = [x1, x2], this function
    # returns the overlapo=ing intervals if it exists
    
    # get x-cooords of intervals
    xl1, xh1 = I1
    xl2, xh2 = I2
    # the overlaping interval starts from largest lower point of both
    # and ends at smallest upper point of both
    xl = max(xl1, xl2)
    xh = min(xh1, xh2)
    # interval is 0 if xl >= xh
    return max(xh - xl, 0)
    
# avod 3D boxes are giben as: box = [x,y,z,w,l,h,ry]
# height axis is y, rotaition is around y (in rad)
    
def box2DToPoly(box):
    # transforms a 2D rotated box into a shapeley polygon
    # box = [x, z, w, h, ry]
    # get box
    cx, cy, l, w, theta = box
    cent = np.array([cx, cy])
    # calc vectors corresponding to w, and h
    vec_l = np.array([l * np.cos(theta), l * np.sin(theta)])
    vec_w = np.array([-w * np.sin(theta), w * np.cos(theta)])
    # get the four corners
    c1 = cent - vec_w - vec_l
    c2 = cent - vec_w + vec_l
    c3 = cent + vec_w + vec_l
    c4 = cent + vec_w - vec_l
    return Polygon([c1, c2, c3, c4])

def iou_3Dbox(box1, box2):   
    # computes the iou of 3D boxes
    # box = [x, y, z, w, l, h, theta]
    # boxes can rotate only on y axis
    # get boxes
    cx1, cy1, cz1, l1, w1, h1, theta1 = box1
    cx2, cy2, cz2, l2, w2, h2, theta2 = box2
    # calc volumes
    V1 = w1 * l1 * h1
    V2 = w2 * l2 * h2 
    # take 2D bases and transfprm to shapeley polygons
    base1 = box2DToPoly([cx1, cz1, l1/2, w1/2, theta1])
    base2 = box2DToPoly([cx2, cz2, l2/2, w2/2, theta2])
    # calc intersection base area
    A_ov = base1.intersection(base2).area
    # calc intersection length on y axis
    I1 = [cy1 - h1/2, cy1 + h1/2]
    I2 = [cy2 - h2/2, cy2 + h2/2]
    z_ov = inter_overlap(I1, I2)
    # intersection volume
    V12 = A_ov * z_ov
    # find intersection and union
    inter = V12
    union = V1 + V2 - V12
    if union == 0:
        return 0
    return inter / union

