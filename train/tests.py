import unittest

import torch

from step_tokenize import (step_features_to_str, get_state_indices, step_index_to_features, 
step_sequence_to_targets, sequence_to_tensor)

d = torch.device('cpu')

class TestStepchart(unittest.TestCase):
	@unittest.skip('temp')
	def test_tokenization_single(self):
		for i in range(1024):
			feats = step_index_to_features(i, 'pump-single', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-single', None)[0].item(), i)

	@unittest.skip('temp')
	def test_tokenization_double(self):
		for i in range(20686):
			feats = step_index_to_features(i, 'pump-double', None, d).unsqueeze(0)
			self.assertEqual(step_sequence_to_targets(feats, 'pump-double', None)[0].item(), i)

	@unittest.skip('temp')
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

	def test_get_state_indices(self):
		h_indices_0 = get_state_indices(0, [0,1], 'pump-single')

		tokens_to_check = ['.....', 'X.W..', 'X.H..', '..W..', 'X....', 'X.HX.', '..WX.']
		tokens_to_omit = ['W.WX.', 'H..X.', 'H....', 'W.WXX', 'H.W..', 'W...X']		

		check_targets, _ = step_sequence_to_targets(sequence_to_tensor(tokens_to_check), 'pump-single', None)
		omit_targets, _ = step_sequence_to_targets(sequence_to_tensor(tokens_to_omit), 'pump-single', None)

		for i, target in enumerate(check_targets):
			self.assertTrue(target.item() in h_indices_0, f'{target.item()}, {tokens_to_check[i]} not in {h_indices_0}')

		for i, target in enumerate(omit_targets):
			self.assertFalse(target.item() in h_indices_0, f'{target.item()}, {tokens_to_omit[i]} not in {h_indices_0}')

		h_indices_1 = get_state_indices(4, [2,3], 'pump-single')

		tokens_to_check = ['....H', 'X.W.W', 'X.H.W', '..W.H', 'X...W', 'X.HXW', '..WXH']
		tokens_to_omit = ['W.WX.', 'H.WX.', 'H..X.', 'W.WXX', 'H....', 'W.WXX', 'H.W..', 'W...X']

		check_targets, _ = step_sequence_to_targets(sequence_to_tensor(tokens_to_check), 'pump-single', None)
		omit_targets, _ = step_sequence_to_targets(sequence_to_tensor(tokens_to_omit), 'pump-single', None)

		for i, target in enumerate(check_targets):
			self.assertTrue(target.item() in h_indices_1, f'{target.item()}, {tokens_to_check[i]} not in {h_indices_1}')

		for i, target in enumerate(omit_targets):
			self.assertFalse(target.item() in h_indices_1, f'{target.item()}, {tokens_to_omit[i]} not in {h_indices_1}')

if __name__ == '__main__':
	unittest.main()
