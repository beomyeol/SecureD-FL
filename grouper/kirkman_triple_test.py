"""Kirkman triple tests."""
# pylint: disable=missing-class-docstring,missing-function-docstring
from __future__ import absolute_import, division, print_function

import unittest

from kirkman_triple import find_kirkman_triples


class KirkmanTripleTests(unittest.TestCase):

    def test_9_participants(self):
        solutions = find_kirkman_triples(9)

        self.assertEqual(len(solutions), 4)  # 3 * 1 + 1

        # check non overlapping
        members = {}  # maps id -> others' ids to communicate with

        for solution in solutions:
            for group in solution:
                self.assertEqual(len(group), 3)
                group_set = set(group)
                for node_id in group_set:
                    prev_set = members.get(node_id, set())
                    members[node_id] = prev_set | (group_set - set([node_id]))

        all_indexes = set(range(9))
        self.assertEqual(set(members.keys()), all_indexes)

        for node_id, others_set in members.items():
            self.assertEqual(len(others_set), 8)
            self.assertEqual(others_set.union(set([node_id])), all_indexes)


if __name__ == "__main__":
    unittest.main()
