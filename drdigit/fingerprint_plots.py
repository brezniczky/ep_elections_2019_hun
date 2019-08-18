import os
import numpy as np
import matplotlib.pyplot as plt


def plot_fingerprint(party_votes, valid_votes,
                     registered_voters, title, filename=None,
                     weighted=True,
                     zoom_onto=False,
                     fingerprint_dir="",
                     quiet=False):

    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]
    if zoom_onto:
        bins[1] = 0.4 * bins[1]

    weights = None if not weighted else party_votes
    plt.hist2d(
        # winner_votes / registered_voters,  # TODO: or valid_votes?
        valid_votes / registered_voters,
        party_votes / valid_votes,  # TODO: or valid_votes?
        bins=bins,
        weights=weights
    )
    plt.title(title)
    if filename is not None:
        if not os.path.exists(fingerprint_dir):
            os.mkdir(fingerprint_dir)
        full_filename = os.path.join(fingerprint_dir, filename)
        plt.savefig(full_filename)
        print("plot saved as %s" % full_filename)
    if not quiet:
        plt.show()
