#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:42:40 2019

@author: jiedeng
"""

def autoscale_y(ax,margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    expample:
        import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(-100,100,1000)
    y = x**2 + np.cos(x)*100
    
    fig,axs = plt.subplots(1,2,figsize=(8,5))
    
    for ax in axs:
        ax.plot(x,y)
        ax.plot(x,y*2)
        ax.plot(x,y*10)
        ax.set_xlim(-10,10)
    
    autoscale_y(axs[1])
    
    axs[0].set_title('Rescaled x-axis')
    axs[1].set_title('Rescaled x-axis\nand used "autoscale_y"')
    
    plt.show()
    """

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)



def set_major_axis_font(ax,fontsize=13):
    """
    set major axis label fontsize

    Params
    -----
    ax : axes
    fontsize : fontsize
    """
    try:
        len(ax)
        tmp = ax.flatten()
        for axx in tmp:
            _set_major_axis_font(axx,fontsize=fontsize)
    except:
        _set_major_axis_font(ax,fontsize=fontsize)


def _set_major_axis_font(ax,fontsize):
    """
    helper function for set_major_axis_font
    """
    for i in range(len(ax.xaxis.get_major_ticks())):
        ax.xaxis.get_major_ticks()[i].label.set_fontsize(fontsize)
    for j in range(len(ax.yaxis.get_major_ticks())):
        ax.yaxis.get_major_ticks()[j].label.set_fontsize(fontsize)

def show_minor_ticks(ax,fontsize=13):
    """
    set major axis label fontsize

    Params
    -----
    ax : list of ax or a single ax
    fontsize : fontsize
    """
    from matplotlib.ticker import AutoMinorLocator
    try:
        len(ax)
        tmp = ax.flatten()
        for axx in tmp:
            axx.xaxis.set_minor_locator(AutoMinorLocator())
            axx.yaxis.set_minor_locator(AutoMinorLocator())
    except:
       ax.xaxis.set_minor_locator(AutoMinorLocator())
       ax.yaxis.set_minor_locator(AutoMinorLocator())

def load_default_setting(font_family='Arial',fontsize=13):
    """
    """
    import matplotlib as mpl
    mpl.rc('font',family = font_family)
    mpl.rc('font',size = fontsize)
    print("set font family as {0}".format(font_family))
    print("set font size as {0}".format(fontsize))
