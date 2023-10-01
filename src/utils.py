import numpy as np

def sort_array_based_on_other(main_array, order_array):
    """
    Sort 'order_array' based on the sorted order of 'main_array'.
    """
    zipped_arrays = zip(main_array, order_array)
    sorted_arrays = sorted(zipped_arrays, key=lambda x: x[0])
    sorted_order_array = np.array([item[1] for item in sorted_arrays])
    return sorted_order_array