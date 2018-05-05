def int_to_str(int2str, idx):
    res = ''
    for i in range(len(idx)):
        res += int2str[idx[i]]
    return res

def adjust_int_output(idx):
    res = []
    for i in range(len(idx)):
        if idx[i]==3:
            break
        res.append(idx[i])
    return res