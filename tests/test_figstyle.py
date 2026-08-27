"""The figure style's three promises: exact printed size, zero baseline, shared scale."""

import figstyle as fs

CM_PER_INCH = 2.54


def test_figure_is_built_at_its_printed_width():
    """A figure must leave the module at the width it occupies in the PDF."""
    fig, _ = fs.figure("full", ratio=0.5)
    w, h = fig.get_size_inches()
    assert round(w * CM_PER_INCH, 2) == fs.FULL_CM
    assert round(h * CM_PER_INCH, 2) == fs.FULL_CM * 0.5

    fig, _ = fs.figure("narrow")
    assert round(fig.get_size_inches()[0] * CM_PER_INCH, 2) == fs.NARROW_CM


def test_include_zero_extends_the_axis_to_zero():
    _, ax = fs.figure()
    ax.set_ylim(0.43, 0.75)
    fs.include_zero(ax)
    assert ax.get_ylim() == (0.0, 0.75)

    ax.set_ylim(-0.8, -0.2)  # also reaches up to zero from below
    fs.include_zero(ax)
    assert ax.get_ylim() == (-0.8, 0.0)


def test_include_zero_leaves_an_axis_that_already_spans_zero():
    _, ax = fs.figure()
    ax.set_ylim(-1.0, 1.0)
    fs.include_zero(ax)
    assert ax.get_ylim() == (-1.0, 1.0)


def test_match_limits_gives_every_axis_the_union():
    _, axes = fs.figure(ncols=2)
    axes[0].set_ylim(0.0, 0.5)
    axes[1].set_ylim(0.2, 0.9)
    fs.match_limits(axes)
    assert axes[0].get_ylim() == axes[1].get_ylim() == (0.0, 0.9)


def test_palette_survives_greyscale():
    """Consecutive colours must differ enough in luminance to print in grey."""

    def luminance(hex_color):
        rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    lums = sorted(luminance(c) for c in fs.PALETTE)
    assert min(b - a for a, b in zip(lums, lums[1:])) > 0.05


def test_every_colour_carries_its_own_dash():
    """Colour never encodes alone, so the cycle must pair the two."""
    assert len(fs.PALETTE) == len(fs.DASHES)
    entries = list(fs.CYCLE)
    assert {e["color"] for e in entries} == set(fs.PALETTE)
    assert {e["linestyle"] for e in entries} == set(fs.DASHES)
