import scipy
import os
from copy import deepcopy

def get_label_dict(key):
    dir_name = "datasets/worm_data/"
    labels_fn = filter_filename(dir_name, include = ["labels-BrainScanner", key])
    assert len(labels_fn) == 1
    labels_fn = labels_fn[0]
    label_dict = {}
    label_inverse_dict = {}
    with open(dir_name + labels_fn, "r") as f:
        for line in f.readlines():
            neuron, id = line.split(" ")
            id = eval(id)
            label_dict[id] = neuron
            label_inverse_dict[neuron] = id
    return label_dict, label_inverse_dict


def get_rankings(related_neurons, rankings_ori):
    related_neurons_ids = []
    rankings = []
    for neuron in related_neurons:
        if neuron in label_inverse_dict:
            related_neurons_id = label_inverse_dict[neuron]
            related_neurons_ids.append(related_neurons_id)
            rankings.append(rankings_ori[related_neurons_id])
        else:
            related_neurons_ids.append(np.NaN)
            rankings.append(np.NaN)
    return rankings


def filter_filename(dirname, include = [], exclude = [], array_id = None):
    """Filter filename in a directory"""
    def get_array_id(filename):
        array_id = filename.split("_")[-2]
        try:
            array_id = eval(array_id)
        except:
            pass
        return array_id
    filename_collect = []
    if array_id is None:
        filename_cand = [filename for filename in os.listdir(dirname)]
    else:
        filename_cand = [filename for filename in os.listdir(dirname) if get_array_id(filename) == array_id]
    
    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]
    
    for filename in filename_cand:
        is_in = True
        for element in include:
            if element not in filename:
                is_in = False
                break
        for element in exclude:
            if element in filename:
                is_in = False
                break
        if is_in:
            filename_collect.append(filename)
    return filename_collect


def sort_two_lists(list1, list2, reverse = False):
    """Sort two lists according to the first list."""
    from operator import itemgetter
    if reverse:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0), reverse=True))])
    else:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0)))])
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]