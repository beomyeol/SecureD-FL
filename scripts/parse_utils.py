from __future__ import absolute_import, division, print_function

import time


def get_value(line, name):
    idx = line.find(name)
    if idx == -1:
        return None

    start_idx = idx + len(name) + 1
    end_idx = line.find(',', idx)

    value = line[start_idx:end_idx].strip()
    if value.startswith('['):
        value = value[1:-1].split('/')
    return value


FORMAT = '%Y-%m-%d %H:%M:%S,%f'


def get_timestamp(line):
    idx = line.find(' - ')
    if idx == -1:
        return None
    return time.mktime(time.strptime(line[:idx].strip(), FORMAT))
