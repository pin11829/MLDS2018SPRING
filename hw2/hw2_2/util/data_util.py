import collections
import pickle

import numpy as np

def _build_dict_and_save(all_words, filename_int2str, filename_str2int, min_count):
    counter = collections.Counter(all_words)
    print('total words:', len(counter.most_common()))
    int2str = ['<unk>', '<pad>', '<bos>', '<eos>'] + [x[0] for x in counter.most_common() if x[1]>min_count]
    str2int = {x:i for i, x in enumerate(int2str)}
    print('dict size:', len(int2str))

    with open(filename_int2str, 'wb') as f:
        pickle.dump(int2str, f)
    with open(filename_str2int, 'wb') as f:
        pickle.dump(str2int, f)

    return int2str, str2int


def _load_dict(filename_int2str, filename_str2int):
    with open(filename_int2str, 'rb') as f:
        int2str = pickle.load(f)
    with open(filename_str2int, 'rb') as f:
        str2int = pickle.load(f)

    return int2str, str2int


def _prepare_pairs(
    data,
    int2str,
    str2int,
    valid=True,
    reshuffle=True,
    shuffle_file='shuffle.npy',
    input_max_len=35,
    num_valid=200000):

    max_len = 0
    for para in data:
        for line in para:
            max_len = max(max_len, len(line))
    print('max original sentence len:', max_len)

    data_len = []
    bos_token = str2int['<bos>']
    eos_token = str2int['<eos>']
    pad_token = str2int['<pad>']

    for i in range(len(data)):
        data_len.append([])
        for j in range(len(data[i])):
            data[i][j] = [bos_token] + data[i][j]
            data[i][j] = data[i][j][0:input_max_len-1] + [eos_token]
            data_len[i].append(len(data[i][j]))
            data[i][j] += [pad_token] * (input_max_len-len(data[i][j]))
    print(data[0])

    data_x = []
    data_y = []
    data_x_len = []
    data_y_len = []
    data_y_shift = []
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data_x.append(data[i][j])
            data_y.append(data[i][j+1])
            data_x_len.append(data_len[i][j])
            data_y_len.append(data_len[i][j+1])
            data_y_shift.append(data[i][j+1][1:] + [pad_token])
    print('total data pairs:', len(data_x))
    print(data_x[0], data_y[0])
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    data_x_len = np.asarray(data_x_len, dtype=np.int32)
    data_y_len = np.asarray(data_y_len, dtype=np.int32)
    data_y_shift = np.asarray(data_y_shift)

    if reshuffle:
        s = np.arange(len(data_x))
        np.random.shuffle(s)
        np.save(shuffle_file, s)
        data_x = data_x[s]
        data_y = data_y[s]
        data_x_len = data_x_len[s]
        data_y_len = data_y_len[s]
        data_y_shift = data_y_shift[s]
    else:
        s = np.load(shuffle_file)
        data_x = data_x[s]
        data_y = data_y[s]
        data_x_len = data_x_len[s]
        data_y_len = data_y_len[s]
        data_y_shift = data_y_shift[s]

    data_input = {}
    data_output = {}
    data_input_len = {}
    data_output_len = {}
    data_output_shift = {}

    if valid:
        data_input['valid'] = data_x[-num_valid:]
        data_output['valid'] = data_y[-num_valid:]
        data_input['train'] = data_x[:-num_valid]
        data_output['train'] = data_y[:-num_valid]

        data_input_len['valid'] = data_x_len[-num_valid:]
        data_output_len['valid'] = data_y_len[-num_valid:]
        data_input_len['train'] = data_x_len[:-num_valid]
        data_output_len['train'] = data_y_len[:-num_valid]

        data_output_shift['valid'] = data_y_shift[-num_valid:]
        data_output_shift['train'] = data_y_shift[:-num_valid]
    else:
        data_input['valid'] = None
        data_output['valid'] = None
        data_input['train'] = data_x
        data_output['train'] = data_y

        data_input_len['valid'] = None
        data_output_len['valid'] = None
        data_input_len['train'] = data_x_len
        data_output_len['train'] = data_y_len

        data_output_shift['valid'] = None
        data_output_shift['train'] = data_y_shift

    return data_input, data_output, data_input_len, data_output_len, data_output_shift


def read_data(
    filename='data/clr_conversation.txt',
    build_dict=True,
    filename_int2str='int2str.pkl',
    filename_str2int='str2int.pkl',
    valid=True,
    reshuffle=True,
    shuffle_file='shuffle.npy',
    input_max_len=35,
    min_count=10):

    with open(filename, 'r') as f:
        data = f.read()
    data = data.strip().split('+++$+++')[:-1]
    data = [
        [
            #'''[c for c in line.strip() if c!=' ']'''
            list(line.replace(' ', '')) for line in lines.strip().split('\n')
        ] for lines in data
    ]
    print('total paragraph:',len(data))
    print(data[0])
    print()
    print(data[-1])

    if build_dict:
        all_words = []
        for para in data:
            for line in para:
                all_words += line
        int2str, str2int = _build_dict_and_save(all_words, filename_int2str, filename_str2int, min_count)
    else:
        int2str, str2int = _load_dict(filename_int2str, filename_str2int)

    print('first 100 words:', int2str[:100])

    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                data[i][j][k] = str2int.get(data[i][j][k], 0)
    print(data[0])

    data_input, data_output, data_input_len, data_output_len, data_output_shift = _prepare_pairs(
        data,
        int2str,
        str2int,
        valid=valid,
        reshuffle=reshuffle,
        shuffle_file=shuffle_file,
        input_max_len=input_max_len
    )

    return data_input, data_output, data_input_len, data_output_len, data_output_shift, int2str, str2int


def read_test_data(filename, filename_int2str, filename_str2int, input_max_len):
    int2str, str2int = _load_dict(filename_int2str, filename_str2int)
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    for i in range(len(lines)):
        lines[i] = list(lines[i].replace(' ', ''))
        for j in range(len(lines[i])):
            lines[i][j] = str2int.get(lines[i][j], str2int['<unk>'])

    data_len = []
    data = []
    for i in range(len(lines)):
        data.append([str2int['<bos>']] + lines[i])
        data[i] = data[i][:input_max_len-1] + [str2int['<eos>']]
        data[i] = data[i] + [str2int['<pad>']]*(input_max_len-len(data[i]))
        data_len.append(min(len(lines[i])+2, input_max_len))

    print('Total testing data:', len(data))
    print(lines[0])
    print(data[0], data_len[0])
    print(lines[-1])
    print(data[-1], data_len[-1])

    return data, data_len
