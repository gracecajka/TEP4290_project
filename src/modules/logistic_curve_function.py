#%%
import os
import pandas as pd
import sys
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
#%%
#define logistic function and noise function
def logistic(x:np.ndarray,ti:float,tau:float,C0:float,C1:float) -> np.ndarray:
    """
    General logistic function.
    Arguments:
    - x: np.ndarray of observation points (time)
    - ti: inflection time
    - tau: transition time coefficient
    - C0: start value
    - C1: end value

    Returns:
    - np.ndarray with len(x) number of points
    """
    return (C1 - C0)/(1 + np.exp(-(x - ti) / tau)) + C0  

#%%
def noise(start: int, stop: int, lo_time_deltas: list, lo_deviations: list) -> np.ndarray:
    '''
    Generates noise for timeseries for a set of time deltas and 
    deviations.This works by setting random deviations at certain 
    intervals and interpolating the points in between.The noise can then
    simply be added to the smooth timeseries curve to generate the final
    timeseries.
    
    Arguments:
    - start: beginning of the timeseries
    - stop: end of the timeseries
    - lo_time_deltas: list of time deltas which set the points at which 
                      noise trends are set
    - lo_deviations: the respective standard deviation from which the 
                     deviation for each point is drawn.

    Returns:
    - np.ndarray with stop-start+1 values of noise, averaging around 0
    '''
    no_time = stop-start +1 #number of discrete time instances
    final_points = np.zeros(no_time)

    for (time_delta, deviation) in zip(lo_time_deltas, lo_deviations):
        no_points = int((no_time-1)/time_delta)+2 #1 more than necessary to extend series
        end_time = start + (no_points-1)*time_delta
        macro_points = np.random.normal(0, deviation, no_points) 
        macro_point_x = np.linspace(start, end_time,no_points)
        macro_point_x = np.delete(macro_point_x, -1) #delete the extra point here


        extended_macro_points = [macro_points[0]]
        for index, macro_point in enumerate(macro_points[1:]):
            connection = np.linspace(macro_points[index], macro_point, time_delta+1, endpoint=True)
            extended_macro_points.extend(connection[1:])
        extended_macro_points = np.array(extended_macro_points[0:no_time])
        macro_points = np.delete(macro_points, -1)

        final_points = np.add(final_points, extended_macro_points)

    return final_points
# %%
