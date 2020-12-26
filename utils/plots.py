# -*- coding: utf-8 -*-
"""
A method to make bar plots
"""

# imports
import numpy as np
import matplotlib.pyplot as plt


def make_barplot(series_list, title, xlabs_list, legends_list, ylab = 'percentage (%)'):
    # a method to make bar plots
    # we create a group of bars for each tiem in series
    
    assert len(series_list) <= 3 # for visability
    n = len(series_list[0]) 
    
    fig, ax = plt.subplots()
    index = np.arange(n)
    bar_width = 0.25
    opacity = 0.9
    
    if len(series_list) >= 1:
        ax.bar(index, series_list[0], bar_width, alpha=opacity, color='r',
                    label = legends_list[0])
    if len(series_list) >= 2:
        ax.bar(index + bar_width, series_list[1], bar_width, alpha=opacity, color='b',
                label = legends_list[1])
    if len(series_list) >= 3:
        ax.bar(index + 2*bar_width, series_list[2], bar_width, alpha=opacity, color='c',
                label = legends_list[2])
    
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(xlabs_list)
    ax.legend()
    plt.show()
# end
    
    

