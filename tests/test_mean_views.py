from pathlib import Path
import sys
import unittest
import tempfile
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import SKAPP
from mean_views import values_for, retrieval_setting


class MeanViewsTests(unittest.TestCase):
    def test_zero_branch_preserves_outputs_and_rng(self):
        torch.manual_seed(12)
        old = SKAPP(feature_dim=4, hidden_dim=2).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(12)
        new = SKAPP(feature_dim=4, hidden_dim=2, prediction_mean_views=True).eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        args = (torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 3, 4),
                torch.randn(2, 3, 4), torch.randn(2, 3), torch.randn(2, 3, 4), torch.randn(2, 3))
        torch.testing.assert_close(old(*args), new(*args, torch.ones(2, 1)), rtol=0, atol=0)
        new(*args, torch.ones(2, 1)).sum().backward()
        self.assertGreater(new.direct_metadata.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            new(*args)

    def test_invalid_values_and_mismatched_ids_rejected(self):
        for value in (-1., float('nan')):
            with self.assertRaises(ValueError):
                values_for({'image_id': np.array(['a']), 'mean_views': np.array([value])}, 'train')
        with self.assertRaisesRegex(ValueError, 'missing mean_views'):
            values_for({'image_id': np.array(['a'])}, 'train')

    def test_retrieval_protocol_must_be_declared(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'dataset': 'icip'}
            Path(folder, 'dataset.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                retrieval_setting(folder, manifest)
            manifest['retrieval_mean_views'] = False
            self.assertFalse(retrieval_setting(folder, manifest))


if __name__ == '__main__':
    unittest.main()
