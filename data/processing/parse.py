# functionality to parse ssc/ucs stepcharts
# adapted from https://github.com/chrisdonahue/ddc/blob/master/dataset/smdataset/parse.py

import logging
import re

int_parser = lambda x: int(x.strip()) if x.strip() else None
bool_parser = lambda x: True if x.strip() == 'YES' else False
str_parser = lambda x: x.strip() if x.strip() else None
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

def notes_parser(x):
    pattern = r'([^:]*):' * 5 + r'([^;:]*)'
    notes_split = re.findall(pattern, x)
    if len(notes_split) != 1:
        raise ValueError('Bad formatting of notes section')
    notes_split = notes_split[0]
    if (len(notes_split) != 6):
        raise ValueError('Bad formatting within notes section')

    # parse/clean measures
    measures = [measure.splitlines() for measure in notes_split[5].split(',')]
    measures_clean = []
    for measure in measures:
        measure_clean = filter(lambda pulse: not pulse.strip().startswith('//') and len(pulse.strip()) > 0, measure)
        measures_clean.append(measure_clean)
    if len(measures_clean) > 0 and len(measures_clean[-1]) == 0:
        measures_clean = measures_clean[:-1]

    # check measure lengths
    for measure in measures_clean:
        if len(measure) == 0:
            raise ValueError('Found measure with 0 notes')
        if not len(measure) in VALID_PULSES:
            parlog.warning('Nonstandard subdivision {} detected, allowing'.format(len(measure)))

    chart_type = str_parser(notes_split[0])
    if chart_type not in ['dance-single', 'dance-double', 'dance-couple', 'lights-cabinet']:
        raise ValueError('Nonstandard chart type {} detected'.format(chart_type))

    return (str_parser(notes_split[0]),
        str_parser(notes_split[1]),
        str_parser(notes_split[2]),
        int_parser(notes_split[3]),
        list_parser(float_parser)(notes_split[4]),
        measures_clean
    )

ATTR_NAME_TO_PARSER = {
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
    'bpms': bpms_parser,
    'stops': stops_parser,
    'delays': unsupported_parser('delays'),
    'warps': unsupported_parser('warps'),
    'timesignatures': list_parser(kv_parser(float_parser, kv_parser(int_parser, int_parser))),
    'tickcounts': list_parser(kv_parser(float_parser, int_parser)),
    'combos': list_parser(kv_parser(float_parser, int_parser)),
    'speeds': unsupported_parser('speeds'),
    'scrolls': unsupported_parser('scrolls'),
    'fakes': unsupported_parser('fakes'),
    'labels': list_parser(kv_parser(float_parser, str_parser)),
    'lastsecondhint': float_parser,
    'bgchanges': unsupported_parser('bgchanges'),
    'keysounds': str_parser,
    'attacks': str_parser,
    
    'notes': unsupported_parser('notes'), # handle manually


}

# attributes with potentially multiple sets (for each of the chart types)
ATTR_MULTI = ['notedata', 'chartname', 'stepstype', 'description', 'chartstyle',
    'difficulty', 'meter', 'radarvalues', 'credit', 'patchinfo', 'offset', 
    'bpms', 'stops', 'delays', 'warps', 'timesignatures', 'tickcounts', 'combos',
    'speeds', 'scrolls', 'fakes', 'labels', 'notes']
ATTR_NOTES = 'notes'

def parse_chart_txt(chart_txt, chart_type):
    attrs = {attr_name: [] for attr_name in ATTR_MULTI}

    # parse each attribute in the txt
    for attr_name, attr_val in re.findall(r'#([^:]*):([^;]*);', chart_txt):
        attr_name = attr_name.lower()

        if attr_name not in ATTR_NAME_TO_PARSER:
            logging.warning('Found unexpected attribute {}:{}, ignoring'.format(attr_name, attr_val))
            continue

        # if attribute is 'notes', parse differently depending on chart type
        if attr_name == ATTR_NOTES:
            if chart_type == 'ucs':
                attr_val_parsed = ucs_parser(attr_val)
            elif chart_type == 'ssc':
                attr_val_parsed = ssc_parser(attr_val)
            else:
                raise NotImplementedError('Chart type {} invalid'.format(chart_type))
        else:
            attr_val_parsed = ATTR_NAME_TO_PARSER[attr_name](attr_val)
        
        if attr_name in attrs:
            if attr_name not in ATTR_MULTI:
                if attr_val_parsed == attrs[attr_name]:
                    continue
                else:
                    raise ValueError('Attribute {} defined multiple times'.format(attr_name))
            attrs[attr_name].append(attr_val_parsed)
        else:
            attrs[attr_name] = attr_val_parsed

    # remove empty attributes
    for attr_name, attr_val in attrs.items():
        if attr_val == None or attr_val == []:
            del attrs[attr_name]

    return attrs