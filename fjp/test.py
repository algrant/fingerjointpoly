import unittest

import numpy as np

from polyhedron import *

square_poly = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
])

fun_poly = np.array([
    [2, 0, 0],
    [2, 2, 0],
    [1, 3, 0],
    [1, 1, 0],
    [0, 2, 0],
    [0, 0, 0],
])

class TestPolyhedron(unittest.TestCase):
    def test_get_triangle_array(self):
        """
        Test that it can sum a list of integers
        """
        
        result = get_triangle_array(square_poly)

        triangulated_square = np.array([
            [0., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.]
        ])

        np.testing.assert_array_equal(result, triangulated_square)

    def test_gta_fun(self):
        result = get_triangle_array(fun_poly)

        triangulated_poly = np.array([
            [2., 0., 0.],
            [2., 2., 0.],
            [1., 3., 0.],

            [2., 0., 0.],
            [1., 3., 0.],
            [1., 1., 0.],

            [1., 1., 0.],
            [0., 2., 0.],
            [0., 0., 0.],

            [0., 0., 0.],
            [2., 0., 0.],
            [1., 1., 0.]
        ])

        np.testing.assert_array_equal(result, triangulated_poly)

    def test_gta_returns_ccw(self):
        # note to self -- we always get triangles back as ccw, even if input polygon is clockwise...
        r1 = get_triangle_array(fun_poly)
        r2 = get_triangle_array(np.flip(fun_poly, 0))

        np.testing.assert_array_equal(r1, r2)


if __name__ == '__main__':
    unittest.main()