import torch
from sportsam.analysis import Analyzer


def make_mask(shape, ones_coords=None):
    mask = torch.zeros(shape, dtype=torch.bool)
    if ones_coords:
        for y, x in ones_coords:
            mask[y, x] = 1
    return mask


def test_zeroth_image_moment_area():
    mask = make_mask((5, 5), ones_coords=[(1, 1), (2, 2), (3, 3)])
    assert Analyzer.zeroth_image_moment(mask) == 3


def test_zeroth_image_moment_empty():
    mask = make_mask((4, 4))
    assert Analyzer.zeroth_image_moment(mask) == 0


def test_first_image_moment_centroid():
    mask = make_mask((4, 4), ones_coords=[(1, 1), (1, 2), (2, 1), (2, 2)])
    # Centroid should be (1.5, 1.5) rounded to (2, 2)
    assert Analyzer.first_image_moment(mask) == (2, 2)


def test_first_image_moment_empty():
    mask = make_mask((3, 3))
    assert Analyzer.first_image_moment(mask) == (0, 0)


def test_second_image_moment():
    mask = make_mask((3, 3), ones_coords=[(0, 0), (2, 2)])
    # Area = 2, centroid = (1, 1)
    # xx = ((0-1)^2 + (2-1)^2)/2 = (1+1)/2 = 1
    # yy = ((0-1)^2 + (2-1)^2)/2 = (1+1)/2 = 1
    # xy = ((0-1)*(0-1) + (2-1)*(2-1))/2 = (1+1)/2 = 1
    assert Analyzer.second_image_moment(mask) == (1, 1, 1)


def test_second_image_moment_empty():
    mask = make_mask((2, 2))
    assert Analyzer.second_image_moment(mask) == (0, 0, 0)
