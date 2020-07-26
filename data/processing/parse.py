# functionality to parse ssc/ucs stepcharts
# adapted from https://github.com/chrisdonahue/ddc/blob/master/dataset/smdataset/parse.py

import logging
import re
import util

from collections import OrderedDict

int_parser = lambda x: int(x.strip()) if x.strip() else None
bool_parser = lambda x: True if x.strip() == 'YES' else False
str_parser = lambda x: x.strip().lower()
float_parser = lambda x: float(x.strip()) if x.strip() else None

def unsupported_parser(attr_name):
    def parser(x):
        raise ValueError('Unsupported attribute: {} with value {}'.format(attr_name, x))
        return None
    return parser

# key-value parser (like: key=value)
def kv_parser(k_parser, v_parser):
    def parser(x):
        if not x:
            return (None, None)
        k, v = x.split('=', 1)
        return k_parser(k), v_parser(v)
    return parser

# parse each element of a 'list' (a,b,c,...) using 'x_parser'
def list_parser(x_parser):
    def parser(l):
        l_strip = l.strip()
        if len(l_strip) == 0:
            return []
        else:
            return [x_parser(x) for x in l_strip.split(',')]
    return parser

def bpms_parser(x):
    bpms = list_parser(kv_parser(float_parser, float_parser))(x)

    if len(bpms) == 0:
        raise ValueError('No BPMs found in list')
    if bpms[0][0] != 0.0:
        raise ValueError('First beat in BPM list is {}'.format(bpms[0][0]))

    # make sure changes are nonnegative, take last for equivalent
    beat_last = -1.0
    bpms_cleaned = []
    for beat, bpm in bpms:
        if beat == None or bpm == None:
            raise ValueError('Empty BPM found')
        if bpm <= 0.0:
            raise ValueError('Non positive BPM found {}'.format(bpm))
        if beat == beat_last:
            bpms_cleaned[-1] = (beat, bpm)
            continue
        bpms_cleaned.append((beat, bpm))
        if beat <= beat_last:
            raise ValueError('Descending list of beats in BPM list')
        beat_last = beat
    if len(bpms) != len(bpms_cleaned):
        parlog.warning('One or more (beat, BPM) pairs begin on the same beat, using last listed')

    return bpms_cleaned

def stops_parser(x):
    stops = list_parser(kv_parser(float_parser, float_parser))(x)

    beat_last = -1.0
    for beat, stop_len in stops:
        if beat == None or stop_len == None:
            raise ValueError('Bad stop formatting')
        if beat < 0.0:
            raise ValueError('Bad beat in stop')
        if stop_len == 0.0:
            continue
        if beat <= beat_last:
            raise ValueError('Nonascending list of beats in stops')
        beat_last = beat
    return stops

# scroll 'speeds' format: (beat=multiplier=duration=0, ...)
def speeds_parser(x):
    # really only need the beat + speed multiplier
    speeds = list_parser(kv_parser(float_parser, kv_parser(float_parser, kv_parser(float_parser, float_parser))))(x)

    speeds_clean = []
    for speed in speeds:
        beat = speed[0]
        mult = speed[1][0]

        speeds_clean.append((beat, mult))

    return speeds_clean

# represent notes as list of lists. sublist ~ measure, sub-elem ~ beat
def notes_parser(x, chart_type = 'ssc'):

    if(chart_type not in ['ucs', 'ssc']):
        return
    
    # parse/clean measures
    measures = [measure.splitlines() for measure in x.split(',')]
    measures_clean = []
    for measure in measures:
        measure_clean = filter(lambda pulse: not pulse.strip().startswith('//') and len(pulse.strip()) > 0, measure)
        measures_clean.append(list(measure_clean))

    if len(measures_clean) > 0 and len(measures_clean[-1]) == 0:
        measures_clean = measures_clean[:-1]

    # check measure lengths
    for measure in measures_clean:
        if len(measure) == 0:
            raise ValueError('Found measure with 0 notes')

    return measures_clean

def ucs_notes_parser(x):
    measures_clean = []

    return measures_clean

ATTR_NAME_TO_PARSER = {
    # song attrs.
    'title': str_parser,
    'subtitle': str_parser,
    'artist': str_parser,
    'titletranslit': str_parser,
    'subtitletranslit': str_parser,
    'artisttranslit': str_parser,
    'genre': str_parser,
    'credit': str_parser,
    'banner': str_parser,
    'background': str_parser,
    'previewvid': str_parser,
    'editable': bool_parser,
    'version': str_parser,
    'origin': str_parser,
    'jacket': str_parser,
    'cdimage': str_parser,
    'discimage': str_parser,
    'lyricspath': str_parser,
    'cdtitle': str_parser,
    'music': str_parser,
    'offset': float_parser,
    'samplestart': float_parser,
    'samplelength': float_parser,
    'selectable': bool_parser,
    'songtype': str_parser,
    'songcategory': str_parser,
    'volume': int_parser,
    'displaybpm': float_parser,
    'stops': stops_parser,
    'delays': str_parser,
    'warps': str_parser,
    'timesignatures': list_parser(kv_parser(float_parser, kv_parser(int_parser, int_parser))),
    'tickcounts': list_parser(kv_parser(float_parser, int_parser)),
    'labels': list_parser(kv_parser(float_parser, str_parser)),
    'lastsecondhint': float_parser,
    'bgchanges': str_parser,
    'fgchanges': str_parser,
    'keysounds': str_parser,
    'attacks': str_parser,
    
    # chart attrs.
    'notedata': str_parser,
    'chartname': str_parser,
    'stepstype': str_parser,                    # 'pump-single'/'pump-double'
    'description': str_parser,                  #  has 'oucs' distinction after chart type/level
    'chartstyle': str_parser,
    'difficulty': str_parser,                   # diff. as in basic mode (easy/med/hard/...)
    'meter': int_parser,                        # difficulty level (numeric)
    'radarvalues': list_parser(float_parser),
    'credit': str_parser,                       # chart author (if known)
    'patchinfo': str_parser,
    'offset': float_parser,

    # bpms used in chart stored as (beat=bpm, ...)
    'bpms': bpms_parser,

    'combos': str_parser,
    'speeds': speeds_parser,
    
    # scroll speeds for holds?
    'scrolls': list_parser(kv_parser(float_parser, float_parser)),

    'fakes': str_parser,
    'notes': unsupported_parser('notes'),       # handle manually
}

# distinguish song and chart attributes
ATTR_SONG = []
ATTR_CHART = ['notedata', 'chartname', 'stepstype', 'description', 'chartstyle', 
    'stops', 'warps', 'delays', 'difficulty', 'meter', 'radarvalues', 'credit',
    'patchinfo', 'offset', 'displaybpm', 'bpms', 'tickcounts', 'combos', 'speeds',
    'scrolls', 'fakes', 'labels', 'attacks', 'timesignatures', 'notes']
ATTR_CHART_REQUIRED = ['stepstype', 'description', 'meter', 'credit', 'offset',
    'stops', 'bpms', 'speeds', 'notes']

# attributes with potentially multiple sets (for each of the chart types)
ATTR_NOTES = 'notes'

def parse_chart_txt(chart_txt, chart_type):
    attrs = {}

    # parse each attribute in the txt
    for attr_name, attr_val in re.findall(r'#([^:]*):([^;]*);', chart_txt):
        attr_name = attr_name.lower()

        if attr_name not in ATTR_NAME_TO_PARSER:
            logging.warning('Found unexpected attribute {}:{}, ignoring'.format(attr_name, attr_val))
            continue

        # if attribute is 'notes', parse differently depending on chart type
        if attr_name == ATTR_NOTES:
            attr_val_parsed = notes_parser(attr_val, chart_type)
        else:
            attr_val_parsed = ATTR_NAME_TO_PARSER[attr_name](attr_val)

        # check for duplicate song attrs
        if attr_name in attrs:
            if attr_val_parsed == attrs[attr_name]:
                continue
            else:
                raise ValueError('Song Attribute {} defined multiple times'.format(attr_name))
        elif attr_name in ATTR_CHART:

            if attr_name not in ATTR_CHART_REQUIRED:
                continue

            if 'charts' in attrs and len(attrs['charts']) > 0:
                latest_chart = attrs['charts'][-1]
                latest_chart_attr = next(reversed(latest_chart))

                # check if last chart finished
                if latest_chart_attr == 'notes':
                    # start new chart if applicable
                    if attr_name == 'stepstype':
                        next_chart = OrderedDict([(attr_name, attr_val_parsed)])
                        attrs['charts'].append(next_chart)
                else:
                    # skip if a UCS chart in an ssc
                    if attr_name == 'description' and len(re.findall('ucs', attr_val_parsed)) > 0:
                        del attrs['charts'][-1]
                        continue

                    # if notes,  convert -> ([measure, beat, split], abs_beat,
                    # abs_time, notes [as str ~ '1001'])
                    if attr_name == 'notes':
                        attr_val_parsed = util.calc_note_beats_and_abs_times(
                            latest_chart['offset'], latest_chart['bpms'],
                            latest_chart['stops'], attr_val_parsed)
                    
                    latest_chart[attr_name] = attr_val_parsed
            else:                
                attrs['charts'] = []

                if attr_name == 'stepstype':
                    first_chart = OrderedDict([(attr_name, attr_val_parsed)])
                    attrs['charts'].append(first_chart)

        else:
            attrs[attr_name] = attr_val_parsed

    return attrs