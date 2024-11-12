import numpy as np

def gen_xor(input_shape):
    
    return np.random.randint(2, size = input_shape)

def modify_result(input_array):

    return np.where(input_array == 0, -1, input_array)

def gen_label(input_array):

        result = np.logical_xor(input_array[:, 0],input_array[:, 1]).astype(int)
        return modify_result(result)