import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes


def _get_full_filename(fingerprint_dir, filename):
    if not os.path.exists(fingerprint_dir):
        os.mkdir(fingerprint_dir)
    full_filename = os.path.join(fingerprint_dir, filename)
    return full_filename


def plot_fingerprint(party_votes, valid_votes, registered_voters, title,
                     filename=None, weighted=True, zoom_onto=False,
                     fingerprint_dir="", quiet=False, axes=None):
    """
    Plot electoral fingerprint (a 2D histogram).
    Originally recommended to be used in conjunction with the winner party.

    :param party_votes: Array like of number of votes cast on the party depcted.
    :param valid_votes: Array like of valid votes (ballots) cast.
    :param registered_voters: Array like of all voters eligible to vote.
    :param title: Plot title.
    :param filename: Filename ot save the plot under, None to prevent saving.
        A .png extension seems to work ;)
    :param weighted: Whether to use the number of votes won as a weight on the
        histogram.
    :param zoom_onto: Boolean, allows to magnify the plot by a factor of 5/2
        over the y axis, vote proportion received.
    :param fingerprint_dir: Directory to save the fingerprint plot to.
    :param quiet: Whether to show the resulting plot, False prevents it.
    :param axes: MatPlotLib  axes to plot onto. If left as None, a new plot will
        be generated.
    :return: None
    """
    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]
    if zoom_onto:
        bins[1] = 0.4 * bins[1]

    weights = None if not weighted else party_votes

    if axes is not None:
        dest = axes
    else:
        dest = plt

    dest.hist2d(
        # winner_votes / registered_voters,  # TODO: or valid_votes?
        valid_votes / registered_voters,
        party_votes / valid_votes,  # TODO: or valid_votes?
        bins=bins,
        weights=weights
    )
    if axes is None:
        dest.title(title)
    else:
        dest.set_title(title)
    if filename is not None:
        full_filename = _get_full_filename(fingerprint_dir, filename)
        dest.savefig(full_filename)
        print("plot saved as %s" % full_filename)
    if not quiet and axes is None:
        dest.show()


def plot_animated_fingerprints(party_votes, valid_votes, registered_voters,
                               frame_inclusions,
                               title,
                               filename=None, weighted=True, zoom_onto=False,
                               fingerprint_dir="", quiet=False, axes=None,
                               interval=200, frame_title_exts=None):

    """
    Can be used to plot an animated .gif showing how the distribution of votes
    changes over various subsets of the electoral wards involved. Technically,
    if party_votes, valid_votes, registered_voters are considered columns, the
    subsets boil down to various selections of the rows, specified by the
    frame_inclusions parameter.

    For the description of the rest of the parameters see plot_fingerprint().

    :param frame_inclusions: An array-like of row filters, each specifying a
        frame. Each row filter is just an array-like of booleans telling to
        include the nth row whenever the nth value is True.
    :param axes: axes to plot to, None to create a new plot.
    :param interval: animation interval to elapse between frames, in millisecs.
    :param frame_title_exts: Optional array-like of frame-specific text to append
        to the title. This parameter is experimental and might change.
    :return: None
    """

    """ 
    based on 
    https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/
    """
    fig, ax = plt.subplots()  # type: object, Axes

    def update(frame_index):
        include = frame_inclusions[frame_index]
        plot_fingerprint(
            party_votes[include], valid_votes[include],
            registered_voters[include], title,
            weighted=weighted, zoom_onto=zoom_onto,
            quiet=True, axes=ax
        )
        if frame_title_exts is not None:
            t = ax.text(0.1, 0.9, frame_title_exts[frame_index], color="white")
            t.set_bbox(dict(facecolor='black', alpha=1, edgecolor='black'))

    fig.set_tight_layout(True)
    anim = FuncAnimation(fig, update,
                         frames=list(range(len(frame_inclusions))),
                         interval=interval)
    if filename != "":
        full_filename = _get_full_filename(fingerprint_dir, filename)
        anim.save(full_filename, dpi=80, writer='imagemagick')
    if not quiet:
        plt.show()
