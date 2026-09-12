import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order
from scripts.extract_task_graph import select_paths
from flypaint.controller import Reservoir

class ExtractionTests(unittest.TestCase):
    def test_whole_directed_paths_and_unreachable_outputs(self):
        graph=sparse.coo_matrix((np.ones(4),([0,1,2,4],[1,2,3,3])),shape=(6,6)).tocsr()
        selected=select_paths(graph,[0],[3,5],4)
        self.assertEqual(selected.tolist(),[0,1,2,3])
        self.assertIn(3,breadth_first_order(graph[selected][:,selected],0,directed=True)[0])
        with self.assertRaises(ValueError):select_paths(graph,[3],[0],4)
        with self.assertRaises(ValueError):select_paths(graph,[0],[3],3)
    def test_endpoint_adapters(self):
        graph=sparse.eye(40,format='csr',dtype=np.float32)
        graph.input_indices=[0,1];graph.output_indices=[30,31]
        controller=Reservoir(graph,10)
        self.assertFalse(controller.encoder[2:].any())
        self.assertEqual(controller.outputs.tolist(),[30,31])
        self.assertEqual(controller.shape,(3,3))
