import unittest
import torch
import pyarrow as pa
from datastar import DataStarGPUFrame

class TestDataStar(unittest.TestCase):
    def test_gpu_frame(self):
        device = "cpu"
        data = {"a": torch.tensor([1, 2, 3])}
        df = DataStarGPUFrame(data, device=device)
        self.assertEqual(len(df), 3)

    def test_arrow_conversion(self):
        t = pa.Table.from_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        df = DataStarGPUFrame.from_arrow_table(t, device="cpu")
        self.assertEqual(len(df), 3)
        tensors = df.to_tensor()
        self.assertTrue("a" in tensors)
        self.assertTrue("b" in tensors)
        self.assertEqual(tensors["a"].dtype, torch.long)
        self.assertEqual(tensors["b"].dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()
