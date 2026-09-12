import unittest
import numpy as np
from flypaint.graph import build_matrix, reservoir_step


class GraphTests(unittest.TestCase):
    def test_direction_and_duplicate_aggregation(self):
        matrix = build_matrix(['a', 'b', 'c'], ['a', 'a', 'c'], ['b'] * 3, [1, 1, 2])
        np.testing.assert_allclose(matrix @ np.array([2, 0, 0]), [0, 1, 0])

    def test_invalid_inputs(self):
        for args in [(['a', 'a'], [], [], []), (['a'], ['x'], ['a'], [1]),
                     (['a'], ['a'], ['a'], [-1]), (['a'], ['a'], ['a'], [float('nan')]),
                     (['a'], ['a'], [], [1])]:
            with self.assertRaises(ValueError):
                build_matrix(*args)

    def test_zero_incoming_and_repeatability(self):
        matrix = build_matrix(['a', 'b'], ['a'], ['b'], [1])
        state = np.zeros(2, dtype=np.float32)
        drive = np.array([0.1, 0], dtype=np.float32)
        first = reservoir_step(matrix, state, drive)
        np.testing.assert_array_equal(first, reservoir_step(matrix, state, drive))
        self.assertTrue(np.isfinite(first).all())
        for _ in range(100):
            state = reservoir_step(matrix, state, drive)
        self.assertLessEqual(float(abs(state).max()), 1)


if __name__ == '__main__':
    unittest.main()
