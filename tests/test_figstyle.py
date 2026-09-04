"""The figure style's two promises: exact printed size and zero baseline."""

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


def test_the_cycle_assigns_the_palette_in_its_declared_order():
    """Colour follows the entity, so the order must be fixed and not cycled."""
    assert [e["color"] for e in fs.CYCLE] == list(fs.PALETTE)
    assert len(set(fs.PALETTE)) == len(fs.PALETTE)
