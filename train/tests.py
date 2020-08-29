import unittest

import torch

from stepchart import (step_index_to_features, step_sequence_to_targets, 
						sequence_to_tensor, step_features_to_str)

d = torch.device('cpu')

class TestStepchart(unittest.TestCase):
	@unittest.skip('tmp')
	def test_tokenization_single(self):
		for i in range(1024):
			feats = step_index_to_features(i, 'pump-single', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-single', None)[0].item(), i)

	@unittest.skip('tmp')
	def test_tokenization_double(self):
		for i in range(20686):
			feats = step_index_to_features(i, 'pump-double', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-double', None)[0].item(), i)

	def test_special_tokens(self):
		special = {20686: 'XXXXXXXXXX', 20687: '..HHXHH..'}
		for key, val in special.items():
			feats = step_index_to_features(key, 'pump-double', special, d)
			self.assertEqual(step_features_to_str(feats[0]), val)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-double', special)[0].item(), key)
		
		new_feats = sequence_to_tensor(['XXWWXXWWXX', 'XX..XX..XX'])
		new_targets, new_tokens = step_sequence_to_targets(new_feats, 'pump-double', special)
		self.assertEqual(new_targets[0].item(), 20688)
		self.assertEqual(new_targets[1].item(), 20689)
		self.assertEqual(new_tokens, 2)

		self.assertEqual(special[20689], 'XX..XX..XX')
		self.assertEqual(special[20688], 'XXWWXXWWXX')

if __name__ == '__main__':
	unittest.main()
