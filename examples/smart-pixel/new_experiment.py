import os
import yaml
import time
import pickle
import argparse
import numpy as np
import fkeras as fk
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from fkeras.metrics.hessian import HessianMetrics

import models
import fault_injection_campaign

def compute_sensitivity_pointer_median(wbi_lists, wbi_list_delta_metrics, last_k_measurements):
    """
    PrioriFI helper function

    Given a list of lists of bit indices, compute the sensitivity pointer of 
    which list to flip next.
    """
    # Compute median of last k delta metrics for each list
    last_delta_metrics = []
    for i in range(len(wbi_lists)):
        if len(wbi_lists[i]) >= last_k_measurements:
            # Take the median of the last k delta metric measurements
            last_delta_metrics.append(np.median(wbi_list_delta_metrics[i][-last_k_measurements:]))
        else: # Take the average
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i]))
    # Take negative to sort in descending order
    sensivity_metric_argsort = np.argsort(-np.array(last_delta_metrics)) 
    
    for i in range(len(sensivity_metric_argsort)):
        if len(wbi_lists[sensivity_metric_argsort[i]]) > 0:
            sensitivity_pointer = sensivity_metric_argsort[i]
            break
    return sensitivity_pointer

def compute_sensitivity_pointer_mean(wbi_lists, wbi_list_delta_metrics, last_k_measurements):
    """
    PrioriFI helper function

    Given a list of lists of bit indices, compute the sensitivity pointer of 
    which list to flip next.
    """
    # Compute median of last k delta metrics for each list
    last_delta_metrics = []
    for i in range(len(wbi_lists)):
        if len(wbi_lists[i]) >= last_k_measurements:
            # Take the mean of the last k delta metric measurements
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i][-last_k_measurements:]))
        else: # Take the average
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i]))
    # Take negative to sort in descending order
    sensivity_metric_argsort = np.argsort(-np.array(last_delta_metrics)) 
    
    for i in range(len(sensivity_metric_argsort)):
        if len(wbi_lists[sensivity_metric_argsort[i]]) > 0:
            sensitivity_pointer = sensivity_metric_argsort[i]
            break
    return sensitivity_pointer

def compute_sensitivity_pointer_mean(wbi_lists, wbi_list_delta_metrics, last_k_measurements,alpha,beta):
    """
    PrioriFI helper function

    Given a list of lists of bit indices, compute the sensitivity pointer of 
    which list to flip next.
    """
    # Compute median of last k delta metrics for each list
    last_delta_metrics = []
    for i in range(len(wbi_lists)):
        if len(wbi_lists[i]) >= last_k_measurements:
            # Take the mean of the last k delta metric measurements
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i][-last_k_measurements:]))
        else: # Take the average
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i]))
    # Take negative to sort in descending order
    sensivity_metric_argsort = np.argsort(-np.array(last_delta_metrics)) 
    
    for i in range(len(sensivity_metric_argsort)):
        if len(wbi_lists[sensivity_metric_argsort[i]]) > 0:
            sensitivity_pointer = sensivity_metric_argsort[i]
            break
    return sensitivity_pointer


for(i in range(1,10):
  
