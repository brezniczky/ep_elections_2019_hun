import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os


def plot_fingerprint(winner_votes, valid_votes,
                     registered_voters, title, filename=None,
                     weighted=True,
                     zoom_onto=False,
                     fingerprint_dir="",
                     quiet=False):

    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]
    if zoom_onto:
        bins[1] = 0.4 * bins[1]

    weights = None if not weighted else winner_votes
    plt.hist2d(
        # winner_votes / registered_voters,  # TODO: or valid_votes?
        valid_votes / registered_voters,
        winner_votes / valid_votes,  # TODO: or valid_votes?
        bins=bins,
        weights=weights
    )
    plt.title(title)
    if filename is not None:
        full_filename = os.path.join(fingerprint_dir, filename)
        plt.savefig(full_filename)
        print("plot saved as %s" % full_filename)
    if not quiet:
        plt.show()


def plot_histogram2d(d_2_1, d_1_2, binx, biny, show=True, filename=None):
    # plot a numpy histogram2d result via matplotlib

    # TODO: quiet mode mutually exclusively with filename
    # plt.pcolormesh(binx, biny, d_1_2, alpha=0.5)

    # couldn't even try out the , shading="gouraud" due to a
    # "Dimensions of C (99, 99) are incompatible with X (100) and/or Y (100);
    # see help(pcolormesh)" probably due to
    # https://github.com/matplotlib/matplotlib/issues/8422
    #
    # alpha=0.5 otherwise looks horrible unfortunately

    # cmap1 = plt.get_cmap('PiYG')
    div = np.maximum(np.max(d_2_1), np.max(d_1_2)) * 1.3

    cmap1 = plt.get_cmap('copper')
    cmap2 = plt.get_cmap('bone')

    plt.pcolormesh(binx, biny, d_1_2, cmap=cmap1, norm=Normalize(vmax=div))
    d_2_1 = np.ma.masked_array( d_2_1, d_2_1 == 0)
    plt.pcolormesh(binx, biny, d_2_1, cmap=cmap2, norm=Normalize(vmax=div))

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


def get_diffs(x1, y1, w1, x2, y2, w2):
    # TODO: zooming

    # 1. histogram gets x arr, y arr, weights
    #    returns values (nx x ny), x bin edges (n+1), y bin edges (n+1)
    #    will spec the bins on input so that they conform on return for 1 and 2
    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]

    v1, x1, y1 = np.histogram2d(x1, y1, bins=bins, weights=w1)
    v2, x2, y2 = np.histogram2d(x2, y2, bins=bins, weights=w2)

    assert all(x1 == x2) and all(y1 == y2)

    # 2. calc. intersection (inert area)
    def is_(x):
        # median or mean as a fallback to act as a threshold of
        # 'surely sufficiently convincingly included'
        mx = np.median(x)
        if mx == 0:
            mx = np.mean(x)
        # 0..1 value for AND-style multiplicability
        return np.minimum(x / mx, 1)

    # operate on slightly fuzzy logical masks here
    is_intersection = is_(v1) * is_(v2)
    is_v1_only = is_(v1) * (1 - is_intersection)
    is_v2_only = is_(v2) * (1 - is_intersection)

    # the product is still in 0..1 and can be used to slice the A or B-specific
    return v1 * is_v1_only, v2 * is_v2_only, x1, y1


def plot_fingerprint_diff(df, party, top_municipalities, bottom_municipalities,
                          show=True, filename=None, ):

    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quadmesh_demo.html
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    df_top = df[
        df.Telepules.isin(top_municipalities)
    ]
    df_bottom = df[
        df.Telepules.isin(bottom_municipalities)
    ]

    def votes_to_coords(df, party):
        # x: turnout, y: winning vote proportion, weight
        # looks like it needs to be transposed when playing with np.histogram2d
        return (df[party] / df["Ervenyes"],
                df["Ervenyes"] / df["Nevjegyzekben"],
                df[party])

    x1, y1, w1 = votes_to_coords(df_bottom, party)
    x2, y2, w2 = votes_to_coords(df_top, party)
    s1 = sum(df_top[party])
    s2 = sum(df_bottom[party])
    print("votes won when not susp", s2, "susp", s1, "ratio", s1 / s2)
    df1 = df_top
    df2 = df_bottom
    print("mean valid votes:",
          np.mean(df1["Ervenyes"]),
          np.mean(df2["Ervenyes"]))
    print("total valid votes ratio \"achieved:",
          np.sum(df1["Ervenyes"]) /
          np.sum(df2["Ervenyes"]))
    print("vote ratio in susp. areas %.2f" % (s1 / sum(df1["Ervenyes"]) * 100))
    print("additional votes perc. in more susp. areas %.2f %%" %
          ((s1 - s2) / (sum(df1["Ervenyes"]) - sum(df2["Ervenyes"])) * 100))
    # have to go CLT
    #
    # std_indiv / sqrt(n_sampl)   ~   sd of the mean
    # std_indiv * sqrt(n_sampl)   ~   sd of the sum
    # sd of the sum over the sum  ~   rel. uncertainty in nr. of votes
    print("rel. sd ratio (< 1 for suspect targeted manipulation):",
          (np.std(df1[party] * (len(df1) ** 0.5)) / np.sum(df1[party])) /
          (np.std(df2[party] * (len(df2) ** 0.5)) / np.sum(df2[party]))
          )

    d_1_2, d_2_1, binx, biny = get_diffs(x1, y1, w1, x2, y2, w2)
    plot_histogram2d(d_1_2, d_2_1, binx, biny, show, filename)
