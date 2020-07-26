# from https://github.com/chrisdonahue/ddc/blob/master/dataset/...
def ez_name(x):
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

# below adapted from "/smdataset/abstime.py
_EPSILON = 1e-6

def bpm_to_spb(bpm):
    return 60.0 / bpm

def calc_segment_lengths(bpms):
    assert len(bpms) > 0
    segment_lengths = []
    for i in xrange(len(bpms) - 1):
        spb = bpm_to_spb(bpms[i][1])
        segment_lengths.append(spb * (bpms[i + 1][0] - bpms[i][0]))
    return segment_lengths

# computes the absolute time for a particular beat
def calc_abs_for_beat(offset, bpms, segment_lengths, beat):
    bpm_idx = 0
    while bpm_idx < len(bpms) and beat + _EPSILON > bpms[bpm_idx][0]:
        bpm_idx += 1
    bpm_idx -= 1

    full_segment_total = sum(segment_lengths[:bpm_idx])
    partial_segment_spb = bpm_to_spb(bpms[bpm_idx][1])
    partial_segment = partial_segment_spb * (beat - bpms[bpm_idx][0])

    return full_segment_total + partial_segment - offset

def calc_note_beats_and_abs_times(offset, bpms, note_data):
    segment_lengths = calc_segment_lengths(bpms)

    # copy bpms
    bpms = bpms[:]
    inc = None
    inc_prev = None
    time = offset

    # beat loop
    note_beats_abs_times = []
    beat_times = []

    for measure_num, measure in enumerate(note_data):
        ppm = len(measure)
        for i, code in enumerate(measure):
            beat = measure_num * 4.0 + 4.0 * (float(i) / ppm)
            # TODO: This could be much more efficient but is not the bottleneck for the moment.
            beat_abs = calc_abs_for_beat(offset, bpms, segment_lengths, beat)
            note_beats_abs_times.append(((measure_num, ppm, i), beat, beat_abs, code))
            beat_times.append(beat_abs)

    # handle negative stops
    beat_time_prev = float('-inf')
    del_idxs = []
    for i, beat_time in enumerate(beat_times):
        if beat_time_prev > beat_time:
            del_idxs.append(i)
        else:
            beat_time_prev = beat_time
    for del_idx in sorted(del_idxs, reverse=True):
        del note_beats_abs_times[del_idx]
        del beat_times[del_idx]

    return note_beats_abs_times
