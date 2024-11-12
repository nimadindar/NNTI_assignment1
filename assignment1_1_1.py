import numpy as np


def gen_array(shape):

    array = np.random.randn(shape[0], shape[1])
    return array

def shift_mean(array, shift_vector):

    array += shift_vector
    return array

def gen_label(len):

    return np.ones((len,1)), -1 * np.ones((len,1))

def stack_arrays(arr1, arr2):

    return np.vstack((arr1, arr2))

def gen_shuffled_indeces(index_size):

    indeces = np.arange(index_size)
    np.random.shuffle(indeces)
    return indeces

def create_dataset(cluster1_input_shape, cluster2_input_shape, cluster1_shift_vec, cluster2_shift_vec):

    cluster1, cluster2 = gen_array(cluster1_input_shape), gen_array(cluster2_input_shape)
    label_cluster1, label_cluster2 = gen_label(cluster1_input_shape[0])

    cluster1 = shift_mean(cluster1, cluster1_shift_vec)
    cluster2 = shift_mean(cluster2, cluster2_shift_vec)


    stacked_clusters = stack_arrays(cluster1, cluster2)
    stacked_labels = stack_arrays(label_cluster1, label_cluster2)

    indeces = gen_shuffled_indeces(cluster1_input_shape[0]+cluster2_input_shape[0])

    return stacked_clusters[indeces], stacked_labels[indeces]
