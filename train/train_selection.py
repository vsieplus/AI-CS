# train model to perform step selection


CHART_PERMUTATIONS = {
    'pump-single': {
       #normal:         '01234'
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

# https://github.com/chrisdonahue/ddc/blob/master/dataset/filter_json.py
def add_permutations(chart_attrs):
    for chart in chart_attrs.get('charts'):
        chart['permutation'] = 'normal'

        chart_type = chart['stepstype']
        if chart_type == 'pump-routine':
            continue

        for permutation_name, permutation in CHART_PERMUTATIONS[chart_type].items():
            chart_copy = copy.deepcopy(chart)
            notes_cleaned = []
            for meas, beat, time, note in chart_copy['notes']:

                # permutation numbers signify moved location
                #   ex) note = '10010'
                #       perm = '43210' -> (flip horizontally)
                #       note_new = '01001'

                note_new = ''.join([note[int(permutation[i])] for i in range(len(permutation))])

                notes_cleaned.append((meas, beat, time, note_new))
                chart_copy['notes'] = notes_cleaned
                chart_copy['permutation'] = permutation_name

            chart_attrs['charts'].append(chart_copy)
    return chart_attrs