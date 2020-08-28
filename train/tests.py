import unittest

import torch

from stepchart import step_index_to_features, step_sequence_to_targets, step_features_to_str


class TestStepchart(unittest.TestCase):
	def test_tokenization_single(self):
		d = torch.device('cpu')
		for i in range(1024):
			feats = step_index_to_features(i, 'pump-single', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-single', None)[0].item(), i)

	def test_tokenization_double(self):
		d = torch.device('cpu')
		for i in range(20686):
			feats = step_index_to_features(i, 'pump-double', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-double', None)[0].item(), i)

if __name__ == '__main__':
	unittest.main()
