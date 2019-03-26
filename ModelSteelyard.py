#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats


class ModelSteelyard:
    @staticmethod
    def mean(values):
        return np.mean(values)

    @staticmethod
    def var(values):
        return np.var(values)
    
    @staticmethod
    def sqrt(values):
        return np.sqrt(values)
    
    @staticmethod
    def std(values):
        return np.std(values)
    
    @staticmethod
    def sample_std(values):
        return np.std(values, ddof=1)
    
    @classmethod
    def mean_interval(cls, values, conf_level=0.95):
        interval = ()
        if values:
            mean = cls.mean(values)
            sample_size = len(values)
            if sample_size > 1:
                std = cls.sample_std(values)
                if std > 0:
                    if sample_size >= 30:
                        interval = stats.norm.interval(conf_level, loc=mean, scale=std / np.sqrt(sample_size))
                    else:
                        interval = stats.t.interval(conf_level, df=sample_size - 1, loc=mean,
                                                    scale=std / np.sqrt(sample_size))
                else:
                    return mean, mean
        return interval
    
    @classmethod
    def delta_interval(cls, control_group, test_group, alpha=0.5):
        n2 = len(control_group)
        n1 = len(test_group)
        u2 = cls.mean(control_group)
        u1 = cls.mean(test_group)
        s2 = cls.var(control_group)
        s1 = cls.var(test_group)
        sw = cls.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 -2))
        t_score = stats.t.isf(alpha / 2, df = (n1 + n2 -2) )
        lower_bound = u1 - u2 - t_score * sw * cls.sqrt(1 / n1 + 1 / n2)
        upper_bound = u1 - u2 + t_score * sw * cls.sqrt(1 / n1 + 1 / n2)
        return lower_bound, upper_bound
    
    @classmethod
    def rise_interval(cls, control_group, test_group):
        u1 = cls.mean(control_group)
        lower_bound, upper_bound  = cls.delta_interval(control_group, test_group)
        lower_rise = (lower_bound) / u1
        upper_rise = (upper_bound) / u1
        return lower_rise, upper_rise
    
    @classmethod
    def surpass_rate(cls, control_group, test_group):
        lower_bound, upper_bound  = cls.delta_interval(control_group, test_group)
        if lower_bound >= 0.0:
            p = 1.0
        elif lower_bound < 0.0 and upper_bound > 0.0:
            p = upper_bound / (upper_bound - lower_bound)
        else:
            p = 0.0
        return p
    
        
if __name__ == "__main__":       
    x = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,0.2, 0.2, 0.2, 0.2]
    y = [0.21,0.29,0.242,0.16,0.23,0.19, 0.13, 0.24, 0.17, 0.23, 0.2, 0.13, 0.24, 0.2, 0.2, 0.2]
    ms = ModelSteelyard()
    interval = ms.mean_interval(y)
    lower_bound, upper_bound = ms.delta_interval(x, y)
    lower_rise, upper_rise = ms.rise_interval(x, y)
    p = ms.surpass_rate(x, y)
    