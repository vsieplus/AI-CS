# utility classes/functions for loading step charts

from functools import lru_cache
import json
import math
import os
import re

import rpy2.robjects as robj
import torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from pathlib import Path
from joblib import Memory

from extract_audio_feats import extract_audio_feats, load_audio
from hyper import (HOP_LENGTH, PAD_IDX, SEED, N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE, SELECTION_VOCAB_SIZES)
from step_tokenize import sequence_to_tensor, step_sequence_to_targets, step_features_to_str, step_index_to_features
from train_util import convert_chartframe_to_melframe

ABS_PATH = str(Path(__file__).parent.absolute())
DATA_DIR = os.path.join('..', 'data')
CACHE_DIR = os.path.join(ABS_PATH, '.dataset_cache/')

R_CHART_UTIL_PATH = os.path.join(ABS_PATH, '..', 'shiny', 'util.R')

memory = Memory(CACHE_DIR, verbose=0, compress=True)
# to reset cache: memory.clear(warn=False)

# add caching to extract_audio_feats and some seq. conversion functions (below)
extract_audio_feats = memory.cache(extract_audio_feats)

# import chart util functions from r script
robj.r.source(R_CHART_UTIL_PATH)	

def collate_charts(batch):
	"""custom collate function for dataloader
		input: dict of chart objects/lengths
		output: dict of batch inputs, targets
	"""
	batch = list(filter(lambda x: x is not None, batch))

	batch_size = len(batch)

	audio_feats = []
	chart_feats = []
	levels = []

	first_frame = 100000000
	last_frame = -1

	placement_targets = []

	step_sequence = []
	step_targets = []

	# list of lengths of examples, for packing
	audio_lengths = []
	step_sequence_lengths = []

	for chart in batch:
		# skip empty charts
		if chart.n_steps == 0:
			continue

		# transpose channels/timestep so timestep comes first for pad_sequence()
		audio_feats.append(chart.get_audio_feats().transpose(0, 1))
		chart_feats.append(torch.tensor(chart.chart_feats).unsqueeze(0))
		placement_targets.append(chart.placement_targets)

		levels.append(chart.level)

		# already tensors
		step_sequence.append(chart.step_sequence)
		step_targets.append(chart.step_targets)

		if chart.first_frame < first_frame:
			first_frame = chart.first_frame
		
		if chart.last_frame > last_frame:
			last_frame = chart.last_frame

		audio_lengths.append(audio_feats[-1].size(0))
		step_sequence_lengths.append(step_sequence[-1].size(0))

	# transpose timestep/channel dims back => [batch, channels, max audio frames, freq]
	audio_feats = pad_sequence(audio_feats, batch_first=True, padding_value=PAD_IDX).transpose(1, 2)
	chart_feats = torch.cat(chart_feats, dim=0) # [batch, chart_feats]

	# [batch, max audio frames]
	placement_targets = pad_sequence(placement_targets, batch_first=True, padding_value=PAD_IDX)

	# [batch, max arrow seq len, arrow features] / [batch, max arrow seq len]
	step_sequence = pad_sequence(step_sequence, batch_first=True, padding_value=PAD_IDX)
	step_targets = pad_sequence(step_targets, batch_first=True, padding_value=PAD_IDX)

	return {'audio_feats': audio_feats,
			'chart_feats': chart_feats,
			'chart_levels': levels,
			'audio_lengths': audio_lengths,
			'placement_targets': placement_targets,
			'first_step_frame': first_frame,
			'last_step_frame': last_frame,
			'step_sequence': step_sequence,
			'step_sequence_lengths': step_sequence_lengths,
			'step_targets': step_targets,
	}

class StepchartDataset(Dataset):
	"""Dataset of step charts"""
	def __init__(self, dataset_json, load_to_memory, first_dataset_load, special_tokens):
		"""dataset_json: path to json file with dataset metadata"""
		assert(os.path.isfile(dataset_json))
		with open(dataset_json, 'r') as f:
			metadata = json.loads(f.read())

		self.name = metadata['dataset_name']
		self.songtypes = metadata['song_types']

		self.chart_type = metadata['chart_type']
		self.step_artists = metadata['step_artists']
		self.mixes = metadata['mixes']
		self.chart_difficulties = metadata['chart_difficulties']
		self.min_level = metadata['min_chart_difficulty']
		self.max_level = metadata['max_chart_difficulty']

		self.permutations = metadata['permutations'] + ['']
		
		self.splits = metadata['splits']

		self.n_unique_songs = 0

        # track special tokens in this dataset (idx -> ucs step (str))
		if special_tokens:
			self.special_tokens = special_tokens
			self.vocab_size = SELECTION_VOCAB_SIZES[self.chart_type] + len(special_tokens)
		else:
			self.special_tokens = {}
			self.vocab_size = SELECTION_VOCAB_SIZES[self.chart_type]

		# store the file paths and chart_indices to load
		self.chart_ids = self.filter_fps([os.path.join(DATA_DIR, fp) for fp in metadata['json_fps']])
		self.print_summary()

		self.load_to_memory = load_to_memory
		if load_to_memory or first_dataset_load:
			self.compute_stats(load_to_memory)
			self.computed_stats = True
		else:
			self.computed_stats = False

	def __len__(self):
		return len(self.chart_ids)

	def __getitem__(self, idx):
		if self.load_to_memory:
			return self.charts[idx]
		else:
			chart_fp, chart_idx, permutation, _ = self.chart_ids[idx]
			try:
				return self.load_chart(chart_fp, chart_idx, permutation)
			except ValueError:
				return None	

	# returns splits of the dataset; assumes 3 way split
	# group songs together to avoid crossover between sets
	def get_splits(self):
		split_songs = {}
		song_ids = list(self.songs.keys())

		for i, split in enumerate(self.splits):
			split_size = round(split * len(self.songs))
			split_start = 0 if i == 0 else last_split_end

			curr_song_ids = song_ids[split_start:split_start + split_size]
			last_split_end = split_start + split_size			

			if i == 0:
				split_songs['train'] = curr_song_ids
			elif i == 1:
				split_songs['valid'] = curr_song_ids
			else:
				split_songs['test'] = curr_song_ids

		train_indices, valid_indices, test_indices = [], [], []

		# return indices of chart_ids used in the __getitem function__
		for j, (_, _, _, song_path) in enumerate(self.chart_ids):
			if song_path in split_songs['train']:
				train_indices.append(j)
			elif song_path in split_songs['valid']:
				valid_indices.append(j)
			elif song_path in split_songs['test']:
				test_indices.append(j)

		return train_indices, valid_indices, test_indices

	def print_summary(self):
		print(f'Dataset name: {self.name}')
		print(f'Chart Type: {self.chart_type}')
		print(f'Total # charts: {self.__len__()}')
		if self.chart_difficulties:
			print(f'Chart levels: {self.chart_difficulties}')
		else:
			print(f'Minimum chart level: {self.min_level}')
			print(f'Maximum chart level: {self.max_level}')
		print(f'Chart Permutations: {self.permutations}')

	def compute_stats(self, load_to_memory):
		print("Caching dataset...")
		
		# load once at start to compute overall vocab size, various stats + cache tensors
		charts = []
		for fp, idx, perm, _ in self.chart_ids:
			chart = self.load_chart(fp, idx, perm, first=True)
			if chart:
				charts.append(chart)

		if load_to_memory:
			self.charts = charts

		self.n_unique_charts = self.__len__() // len(self.permutations)
		self.n_steps = 0
		self.n_steps = sum([chart.n_steps for chart in charts])
		self.n_audio_hours = sum([song.n_minutes for song in self.songs.values()]) / 60

		if not self.step_artists:
			self.step_artists = set()
			for chart in charts:
				self.step_artists.add(chart.step_artist)

		steps_per_second = [chart.steps_per_second for chart in charts]
		self.avg_steps_per_second = sum(steps_per_second) / len(steps_per_second)

	# filter charts to include in the dataset; store path to json + chart index num.
	def filter_fps(self, json_fps):
		self.songs = {}
		chart_ids = []

		for fp in json_fps:
			with open(fp, 'r') as f:
				attrs = json.loads(f.read())
			
			# check attrs for each chart to see if we should add it to the dataset
			chart_indices = self.filter_charts(attrs)

			if chart_indices:
				# create new song if needed
				song_path = os.path.join(DATA_DIR, attrs['music_fp'])
				if song_path not in self.songs:
					self.songs[song_path] = Song(song_path, attrs['title'], attrs['artist'], attrs['genre'], attrs['songtype'])
					self.n_unique_songs += 1

				for chart_idx in chart_indices:
					for permutation in self.permutations:
						chart_ids.append((fp, chart_idx, permutation, song_path))

		print('Done filtering!')
		return chart_ids

	def load_chart(self, chart_json_fp, chart_idx, permutation, first=False):
		with open(chart_json_fp, 'r') as f:
			attrs = json.loads(f.read())

		# .ssc (may contain multiple charts/same song) or .ucs (always 1 chart)    
		orig_filetype = attrs['chart_fp'].split('.')[-1]

		song_path = os.path.join(DATA_DIR, attrs['music_fp'])

		try:
			chart =  Chart(attrs['charts'][chart_idx], self.songs[song_path], orig_filetype, permutation, self.special_tokens)
		except ValueError:
			print('Error while loading chart..., skipping')
			return None	

		if first:
			self.vocab_size += chart.n_special_tokens

		return chart
	
	def filter_charts(self, attrs):
		""" determine which charts in the given attrs belongs in this dataset
			return list of indices, w/length between 0 <= ... <= len(attrs['charts'])
	   	"""
		chart_indices = []

		if not self.songtypes or self.songtypes and attrs['songtype'] in self.songtypes:
			for i, chart_attrs in enumerate(attrs['charts']):
				# skip ucs/missions from ssc files
				if 'description' in chart_attrs and re.search('(ucs|mission|quest)', chart_attrs['description']):
					continue

				if chart_attrs['stepstype'] != self.chart_type:
					continue

				chart_level = int(chart_attrs['meter'])
				if self.chart_difficulties:
					valid_level = chart_level in self.chart_difficulties
				else:
					valid_level = (self.min_level <= chart_level and chart_level <= self.max_level)

				if not valid_level:
					continue

				valid_author = (not self.step_artists or self.step_artists and chart_attrs['credit'] in self.step_artists)

				if valid_author:
					chart_indices.append(i)

		return chart_indices

class Song:
	"""A song object, corresponding to some audio file. May be relied on by
	multiple charts."""

	def __init__(self, audio_fp, title, artist, genre, songtype):
		self.audio_fp = audio_fp
		self.title = title
		self.artist = artist
		self.genre = genre
		self.songtype = songtype

		# shape [3, ?, 80]
		self.audio_feats, self.sample_rate = extract_audio_feats(self.audio_fp)

		# secs = (hop * melframe) / sample_rate
		self.n_minutes = (HOP_LENGTH * self.audio_feats.size(1) / self.sample_rate) / 60

CHART_PERMUTATIONS = {
	'pump-single': {
	    #normal:        '01234'
		'flip':         '43210',
		'mirror':       '34201',
		'flip_mirror':  '10243'
	},

	'pump-double': {
		#normal:        '0123456789'
		'flip':         '9876543210',
		'mirror':       '9875643201',
		'flip_mirror':  '1023465789'
	}
}

STEP_PATTERNS = re.compile('[XMW]')

@lru_cache(maxsize=4096)
def permute_steps(steps, chart_type, permutation_type):
	# permutation numbers signify moved location of original step
	#   ex) steps = '10010'
	#       permt = '43210' -> (flip horizontally)
	#    newsteps = '01001'	
	if not permutation_type:
		return steps

	permutation = CHART_PERMUTATIONS[chart_type][permutation_type]

	new_step = ''.join([steps[int(permutation[i])] for i in range(len(permutation))])

	return new_step

@memory.cache
def parse_notes(notes, chart_type, permutation_type, filetype):
	# [# frames] track which (10 ms) chart frames have steps
	step_placement_frames = []

	# [# frames, # step features] - sequence of non-empty steps with associated frame numbers
	step_sequence = []

	num_arrows = 5 if chart_type == 'pump-single' else 10

	# use UCS notation for less ambiguity
	do_convert = filetype == 'ssc'
	if do_convert:
		converted_sequence = robj.r.convertToUCS([steps for _, _, _, steps in notes])

	for i, (_, _, time, steps) in enumerate(notes):
		if time < 0:
			continue

		if len(steps) != num_arrows:
			raise ValueError('invalid steps, skipping chart...')	

		step_to_check = converted_sequence[i] if do_convert else steps

		# for each frame, track absolute frame number
		step_this_frame = STEP_PATTERNS.search(step_to_check)
		if step_this_frame:
			step_placement_frames.append(int(round(time * CHART_FRAME_RATE)))

		# only store non-empty steps in the sequence
		if step_this_frame:
			step_sequence.append(permute_steps(step_to_check, chart_type, permutation_type))

	return step_placement_frames, step_sequence

def placement_frames_to_targets(placement_frames, audio_length, sample_rate):
	# get chart frame numbers of step placements, [batch, placements (variable)]
	placement_target = torch.zeros(audio_length, dtype=torch.long)

	mel_placement_frames = []
	for f in placement_frames:
		melframe = convert_chartframe_to_melframe(f, sample_rate)

		# ignore steps that come after the song has ended
		if melframe < audio_length:
			mel_placement_frames.append(melframe)

	mel_placement_frames = torch.tensor(mel_placement_frames, dtype=torch.long)

	# and set target to '1' at corresponding melframes
	placement_target[mel_placement_frames] = 1

	any_placements = mel_placement_frames.size(0) > 0
	first_frame = mel_placement_frames[0].item() if any_placements else 0
	last_frame = mel_placement_frames[-1].item() if any_placements else 0

	return placement_target, first_frame, last_frame


class Chart:
	"""A chart object, with associated data. Represents a single example"""
	def __init__(self, chart_attrs, song, filetype, permutation_type, special_tokens):
		self.song = song
		self.filetype = filetype

		self.step_artist = chart_attrs['credit'] if 'credit' in chart_attrs else ''
		self.level = int(chart_attrs['meter'])
		self.chart_type = chart_attrs['stepstype']
		assert(self.chart_type in CHART_PERMUTATIONS)

		# concat one-hot encodings of chart_type/level
		chart_type_one_hot = [0 if self.chart_type == 'pump-single' else 1
							  for _ in range(N_CHART_TYPES)]
		chart_type_one_hot = [0] * N_CHART_TYPES
		if self.chart_type == 'pump-single':
			chart_type_one_hot[0] = 1
		else:
			chart_type_one_hot[1] = 1
		level_one_hot = [0] * N_LEVELS
		level_one_hot[self.level - 1] = 1
		self.chart_feats = chart_type_one_hot + level_one_hot

		self.permutation_type = permutation_type

		try:
			step_placement_frames, step_sequence = parse_notes(chart_attrs['notes'], self.chart_type, self.permutation_type, self.filetype)
		except ValueError:
			raise

		(self.placement_targets,
		 self.first_frame, 
		 self.last_frame) = placement_frames_to_targets(step_placement_frames, self.song.audio_feats.size(1), self.song.sample_rate)

		self.step_sequence = sequence_to_tensor(step_sequence)
		(self.step_targets,
		 self.n_special_tokens) = step_sequence_to_targets(self.step_sequence, self.chart_type, special_tokens)

		# ignore steps in sequence after song has ended
		# (make sure length matches num of placement targets)
		self.step_sequence = self.step_sequence[:self.placement_targets.sum()]
		self.step_targets = self.step_targets[:self.placement_targets.sum()]

		self.n_steps = self.step_sequence.size(0)
		self.steps_per_second = self.n_steps / (self.song.n_minutes * 60)

	# return tensor of audio feats for this chart
	def get_audio_feats(self):
		return self.song.audio_feats
