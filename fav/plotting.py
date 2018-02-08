import math

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np
import pandas as pd


from .configValues import CONFIG
from .base import InvalidInput, hist_to_title, adv_getitem

def get_n_different_colors(n):
    return cm.rainbow(np.linspace(0,1,n))

def symbol_gen():
    while True:
        for s in ["s", "o", "+", "d"]:
            yield s
symbol = symbol_gen()

# Register some configuration values
CONFIG.add_item("plotting.hist_bins", 20, int,
                "How many bins should be used for histograms? (None to disable histograms)",
                none_allowed = True)
CONFIG.add_item("plotting.kde_kernel", "epanechnikov", str,
                "What kernel schoud be used for the kernel density estimates used in plots?"
                " (None to disable kde plots)",
                none_allowed = True,
                restricted = ['gaussian', 'tophat', 'epanechnikov', 'exponential',
                              'linear', 'cosine', None])
CONFIG.add_item("plotting.kde_bandwidth", 1., float,
                "The bandwidth used for the kdes. None for auto-detection via cross-validation.",
                none_allowed = True)
CONFIG.add_item("plotting.kde_linewidth", 2, int,
                "The linewidth used for plotting kdes.",
                none_allowed = False)
CONFIG.add_item("plotting.kde_color", "red", str,
                "The color used for plotting kdes.",
                none_allowed = False)
CONFIG.add_item("plotting.legend_loc", 0, int,
                "The location of the legend (As int from 0 to 9). None hids legend",
                none_allowed = True)
CONFIG.add_item("plotting.ylim", None, int,
                "The maximal y value for plots. None for auto.",
                none_allowed = True)
CONFIG.add_item("plotting.xlim", None, int,
                "The maximal x value for plots. None for auto.",
                none_allowed = True)

class PlotMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_commands.update({"HIST":self.plotting_histogram, "ANGHIST": self.plotting_angular_histogram, "SCATTER": self.plotting_scatter})

    def plotting_scatter(self, name_x, name_y, _for = None, *for_filters):
        fig, mainAx = plt.subplots()
        if _for is not None:
            if _for !="FOR":
                raise InvalidInput("Expecting 'FOR' found '{}'".format(for_filters[0]))
            if for_filters[0]=="SAVED":
                colors = get_n_different_colors(len(for_filters)-1)
                for dataname, c in zip(for_filters[1:], colors):
                    data = self.stored[dataname]
                    mainAx.scatter(adv_getitem(data, name_x), adv_getitem(data, name_y), label=dataname, color = c, marker=next(symbol))
            else:
                range_ = self._get_range(*for_filters)
                colors = get_n_different_colors(len(range_))
                for r,c in zip(range_, colors):
                    data = self._filter_from_r(for_filters[0], r)
                    mainAx.scatter(adv_getitem(data, name_x), adv_getitem(data, name_y), color = c, marker=next(symbol), label=hist_to_title(data._fav_history[-2:]))
            if self.settings["plotting.legend_loc"] is not None:
                mainAx.legend(loc=self.settings["plotting.legend_loc"], prop={'size':6})
        elif len(self.filtered_data):
            fig.text(0.2,0.8, "{} datapoints".format(len(self.filtered_data)) )
            mainAx.scatter(adv_getitem(self.filtered_data, name_x), adv_getitem(self.filtered_data, name_y))
        if self.settings["plotting.xlim"] is not None:
            mainAx.set_xlim(right = self.settings["plotting.xlim"])
        if self.settings["plotting.ylim"] is not None:
            mainAx.set_ylim(top = self.settings["plotting.ylim"])

        mainAx.set_xlabel(name_x)
        mainAx.set_ylabel(name_y)

        plt.show(block=False)

    def _plotting_hist_base(self, angular, *args):
        """
        :param angular: Boolean
        """
        try:
            columnname = args[0]
        except IndexError as e:
            raise InvalidInput("HIST requires the columnname you like to show data about.") from e
        if len(args) == 1:
            show_hist(adv_getitem(self.filtered_data,columnname), columnname,
                      self.filtered_data._fav_datasetname +" "+hist_to_title(self.filtered_data._fav_history),
                      self.settings, angular)
        elif args[1] == "FOR":
            if len(args)>2 and args[2]=="SAVED":
                for dataname in args[3:]:
                    if dataname not in self.stored:
                        raise InvalidInput("The command 'HIST columnname FOR SAVED' expect a list of saved "
                                           "datasets, but nothing was saved under the name "
                                           "'{}'.".format(dataname))
                    show_hist(adv_getitem(self.stored[dataname],columnname), columnname,
                              self.stored[dataname]._fav_datasetname +" "+hist_to_title(self.stored[dataname]._fav_history),
                              self.settings, angular)
            elif len(args)>2 and args[3] in self.filtered_data.columns.values:
                range_ = self._get_range(*args[3:])
                for r in range_:
                    data = self._filter_from_r(args[3], r)
                    show_hist(adv_getitem(data,columnname), columnname,
                              data._fav_datasetname +" " + hist_to_title(data._fav_history),
                              self.settings, angular)
            else:
                raise InvalidInput("The command 'HIST columnname FOR' expect either the keyword "
                                   "'SAVED' or a vaild key that identifies one column of the data,"
                                   " not {}".format(args[3]))
        else:
            raise InvalidInput("The third argument to 'HIST columnname' should be 'FOR' or nothing."
                               "Found {}".format(repr(args[2:])))

    def plotting_angular_histogram(self, *args):
        """
        Show a polar histogram and kernel density estimate of a column of the dataset.

        This assumes that the data is angular data in radians.

        Use 'ANGHIST columnname' for a plot of the current dataset
        Use 'ANGHIST columnname FOR SAVED NAME1, NAME2' to show the plots for the saved datasets NAME1, NAME2
        Use 'ANGHIST columnname FOR key from to [step]' to show the plots for subsets of the current dataset where
            the the value in column key falls into different regions of the range given by from, to and optionally step.
        """
        self._plotting_hist_base(True, *args)

    def plotting_histogram(self, *args):
        """
        Show a histogram and kernel density estimate of a column of the dataset.

        Use 'HIST columnname' for a plot of the current dataset
        Use 'HIST columnname FOR SAVED NAME1, NAME2' to show the plots for the saved datasets NAME1, NAME2
        Use 'HIST columnname FOR key from to [step]' to show the plots for subsets of the current dataset where
            the the value in column key falls into different regions of the range given by from, to and optionally step.
        """
        self._plotting_hist_base(False, *args)
def bin_data(data, num_bins):
    min_ = min(data)
    max_ = max(data)
    bins = np.linspace(min_, max_, num_bins+1)
    groups = pd.value_counts(pd.cut(data, bins), sort=False)
    return groups.values, bins

def plot_kde(data, ax, settings):
    try:
        from sklearn.neighbors.kde import KernelDensity
    except:
        warnings.warn("Cannot import sklearn.neighbors.kde. Cannot plot kernel density estimate.")
        return
    x = np.linspace(0, max(data), 200)
    if settings["plotting.kde_bandwidth"] is not None:
        bw = settings["plotting.kde_bandwidth"]
        kde = KernelDensity(kernel=settings["plotting.kde_kernel"],
                            bandwidth=bw).fit(data.values.reshape(-1, 1))

    else:
        grid = GridSearchCV(KernelDensity(kernel=settings["plotting.kde_bandwidth"]),
                  {'bandwidth': np.linspace(math.radians(2), math.radians(30), 40)},
                cv=min(10, len(data))) # 10-fold cross-validation
        try:
            grid.fit(data.values.reshape(-1, 1))
        except ValueError:
            return #Do not plot kde, if we do not have enough datapoints
        #print("Bandwidth = {}".format(grid.best_params_))
        kde = grid.best_estimator_
    ax.plot(x, np.exp(kde.score_samples(x.reshape(-1,1))), label="kde",
            linewidth = settings["plotting.kde_linewidth"], color = settings["plotting.kde_color"])


def polar_twin(ax):
    """
    Replace ax.twin_x for polar plots.

    Thanks to http://stackoverflow.com/a/19620861/5069869
    """
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar',
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    # There should be a method for this, but there isn't... Pull request?
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()
    # Ensure that original axes tick labels are on top of plots in twinned axes
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)
    plt.setp(ax2.get_yticklabels(), color='red')
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    return ax2

def bin_angular(data, num_bins = 100):
    bins = np.linspace(0, 2*math.pi, num_bins+1)
    groups = pd.value_counts(pd.cut(data, bins), sort=False)
    return groups.values, bins


def show_hist(data, xlabel, histtext, settings, angular):
    num_bins = settings["plotting.hist_bins"]
    max_val = settings["plotting.ylim"]
    subplot_kw = {}
    if angular:
        subplot_kw["projection"]="polar"
    fig, mainAx = plt.subplots(subplot_kw=subplot_kw)
    fig.suptitle(histtext)
    mainAx.set_xlabel(xlabel)

    if len(data):
        if num_bins is not None:
            if angular:
                bin_f=bin_angular
                mainAx.set_theta_direction(-1)
                mainAx.set_theta_zero_location("W")
                unit=36
                mainAx.set_thetagrids([unit*r for r in range(6)])
                fig.text(0.05,0.9, "{} datapoints".format(len(data)) )
            else:
                bin_f=bin_data
                fig.text(0.2,0.8, "{} datapoints".format(len(data)) )

            values, bins = bin_f(data, num_bins)
            bars = mainAx.bar(bins[:-1], values, width=bins[1]-bins[0], linewidth=0.25, align="edge")
            mainAx.set_ylabel("count")
        if max_val:
            mainAx.set_ylim(0,max_val)

        maxc = max(values)
        for r, bar in zip(values, bars):
            bar.set_facecolor(plt.cm.jet(r / maxc))

        if settings["plotting.kde_kernel"] is not None:
            if angular:
                kde_ax = polar_twin(mainAx)
            else:
                kde_ax = mainAx.twinx()
                kde_ax.set_ylabel("density")
            plot_kde(data, kde_ax, settings)

    plt.show(block=False)
