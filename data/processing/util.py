# from https://github.com/chrisdonahue/ddc/blob/master/dataset/util.py
def ez_name(x):
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)