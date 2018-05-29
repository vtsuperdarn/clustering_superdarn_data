def genCmap(param, scale, colors='lasse', lowGray=False):
    """From DavitPy https://github.com/vtsuperdarn/davitpy

    Generates a colormap and returns the necessary components to use it

    Parameters
    ----------
    param : str
        the parameter being plotted ('velocity' and 'phi0' are special cases,
        anything else gets the same color scale)
    scale : list
        a list with the [min,max] values of the color scale
    colors : Optional[str]
        a string indicating which colorbar to use, valid inputs are
        'lasse', 'aj'.  default = 'lasse'
    lowGray : Optional[boolean]
        a flag indicating whether to plot low velocities (|v| < 15 m/s) in
        gray.  default = False

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        the colormap generated.  This then gets passed to the mpl plotting
        function (e.g. scatter, plot, LineCollection, etc.)
    norm : matplotlib.colors.BoundaryNorm
        the colormap index.  This then gets passed to the mpl plotting
        function (e.g. scatter, plot, LineCollection, etc.)
    bounds : list
        the boundaries of each of the colormap segments.  This can be used
        to manually label the colorbar, for example.

    Example
    -------
        cmap,norm,bounds = genCmap('velocity', [-200,200], colors='aj', lowGray=True)

    Written by AJ 20120820

    """
    import matplotlib,numpy
    import matplotlib.pyplot as plot

    #the MPL colormaps we will be using

    cmj = matplotlib.cm.jet
    cmpr = matplotlib.cm.prism


    if(param == 'velocity'):
        #check for what color scale we want to use
        if(colors == 'aj'):
            if(not lowGray):
                #define our discrete colorbar
                cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                         cmpr(.11), cmpr(.1),
                                                         cmpr(.175), cmpr(.158),
                                                         cmj(.32), cmj(.37)])
            else:
                cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                         cmpr(.11), cmpr(.1),
                                                         '.6', cmpr(.175),
                                                         cmpr(.158), cmj(.32),
                                                         cmj(.37)])
        else:
            if(not lowGray):
                #define our discrete colorbar
                cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8),
                                                         cmj(.7), cmj(.65),
                                                         cmpr(.142), cmj(.45),
                                                         cmj(.3), cmj(.1)])
            else:
                cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8),
                                                         cmj(.7), cmj(.65),
                                                         '.6', cmpr(.142),
                                                         cmj(.45), cmj(.3),
                                                         cmj(.1)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],7))
        if(lowGray):
            bounds[3] = -15.
            bounds = numpy.insert(bounds,4,15.)
        bounds = numpy.insert(bounds,0,-50000.)
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    elif(param == 'phi0'):
        #check for what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.142), cmpr(.125),
                                                     cmpr(.11), cmpr(.1),
                                                     cmpr(.18), cmpr(.16),
                                                     cmj(.32), cmj(.37)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.9), cmj(.8), cmj(.7),
                                                     cmj(.65), cmpr(.142),
                                                     cmj(.45), cmj(.3),
                                                     cmj(.1)])

        #define the boundaries for color assignments
        bounds = numpy.linspace(scale[0],scale[1],9)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    elif(param == 'grid'):
        #check what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.175), cmpr(.17),
                                                     cmj(.32), cmj(.37),
                                                     cmpr(.142), cmpr(.13),
                                                     cmpr(.11), cmpr(.10)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.1), cmj(.3), cmj(.45),
                                                     cmpr(.142), cmj(.65),
                                                     cmj(.7), cmj(.8), cmj(.9)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],8))
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    else:
        # If its a non-velocity plot, check what color scale we want to use
        if(colors == 'aj'):
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmpr(.175), cmpr(.158),
                                                     cmj(.32), cmj(.37),
                                                     cmpr(.142), cmpr(.13),
                                                     cmpr(.11), cmpr(.10)])
        else:
            #define our discrete colorbar
            cmap = matplotlib.colors.ListedColormap([cmj(.1), cmj(.3), cmj(.45),
                                                     cmpr(.142), cmj(.65),
                                                     cmj(.7), cmj(.8), cmj(.9)])

        #define the boundaries for color assignments
        bounds = numpy.round(numpy.linspace(scale[0],scale[1],8))
        bounds = numpy.append(bounds,50000.)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cmap.set_bad('w',1.0)
    cmap.set_over('w',1.0)
    cmap.set_under('.6',1.0)

    return cmap,norm,bounds

def drawCB(fig, coll, cmap, norm, map_plot=False, pos=[0,0,1,1]):
    """ From DavitPy https://github.com/vtsuperdarn/davitpy

    manually draws a colorbar on a figure.  This can be used in lieu of
    the standard mpl colorbar function if you need the colorbar in a specific
    location.  See :func:`pydarn.plotting.rti.plotRti` for an example of its
    use.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        the figure being drawn on.
    coll : matplotlib.collections.Collection
        the collection using this colorbar
    cmap : matplotlib.colors.ListedColormap
        the colormap being used
    norm : matplotlib.colors.BoundaryNorm
        the colormap index being used
    map_plot : Optional[bool]
        a flag indicating the we are drawing the colorbar on a figure with
        a map plot
    pos : Optional[list]
        the position of the colorbar.  format = [left,bottom,width,height]

    Returns
    -------
    cb

    Example
    -------

    Written by AJ 20120820

    """
    import matplotlib,numpy
    import matplotlib.pyplot as plot

    if not map_plot:
        # create a new axes for the colorbar
        cax = fig.add_axes(pos)
        # set the colormap and boundaries for the collection of plotted items
        if(isinstance(coll,list)):
            for c in coll:
                c.set_cmap(cmap)
                c.set_norm(norm)
                cb = plot.colorbar(c,cax=cax,drawedges=True)
        else:
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            cb = plot.colorbar(coll,cax=cax,drawedges=True)
    else:
        if(isinstance(coll,list)):
            for c in coll:
                c.set_cmap(cmap)
                c.set_norm(norm)
                cb = fig.colorbar(c,location='right',drawedges=True)
        else:
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            cb = fig.colorbar(coll,location='right',pad="5%",drawedges=True)

    cb.ax.tick_params(axis='y',direction='out')
    return cb



def plot_clusters(x, x_name, y, y_name, cluster_membership):
    import matplotlib.pyplot as plt
    import numpy as np

    y_size = len(np.unique(y))
    y_min = np.min(y)
    y_max = np.max(y)

    x_size = len(np.unique(x))
    x_min = np.min(x)
    x_max = np.max(x)

    color_mesh = np.zeros((x_size, y_size)) * np.nan
    num_clusters = len(cluster_membership)
    plot_param = 'velocity'

    # Create a (num times) x (num range gates) map of cluster values.
    # The colormap will then plot those values as cluster values.
    # cluster_color = np.linspace(-200, 200, num_clusters)
    for k in range(len(cluster_membership)):
        color = k
        # Cluster membership indices correspond to the flattened data, which may contain repeat time values
        for i in cluster_membership[k]:
            ii = np.where(x == x[i])[0][0]
            matching_y = y[ii]

            for my in matching_y:
                color_mesh[ii, my] = color

    # Create a matrix of the right size
    mesh_x, mesh_y = np.meshgrid(np.linspace(x_min, x_max, x_size), np.linspace(y_min, y_max, y_size))
    invalid_data = np.ma.masked_where(np.isnan(color_mesh.T), color_mesh.T)
    #Zm = np.ma.masked_where(np.isnan(data[:tcnt][:].T), data[:tcnt][:].T)
    # Set colormap so that masked data (bad) is transparent.

    cmap, norm, bounds = genCmap(plot_param, [0, len(cluster_membership)],
                                                 colors = 'lasse',
                                                 lowGray = False)

    cmap.set_bad('w', alpha=0.0)

    pos=[.1, .1, .76, .72]
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes(pos)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    #ax.set_title(title_str+' '+start_time.strftime("%d %b %Y") + ' ' + rad.upper())
    colormesh = ax.pcolormesh(mesh_x, mesh_y, invalid_data, lw=0.01, edgecolors='None', cmap=cmap, norm=norm)


    # Draw the colorbar.
    cb = drawCB(fig, colormesh, cmap, norm, map_plot=0,
                      pos=[pos[0] + pos[2] + .02, pos[1], 0.02, pos[3]])
    plt.show()
    plt.savefig((num_clusters + 2).__str__() + "_GMM_all_clusters_colormesh_" + start_time.__str__() + ".png")