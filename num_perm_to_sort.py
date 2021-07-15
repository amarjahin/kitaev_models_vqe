from numpy import argmax

def num_perm_to_sort(a): 
    a = list(a)
    num_perm = 0
    for _ in range(len(a) - 1): 
        max_indx = argmax(a)
        num_perm = num_perm + max_indx
        a.pop(max_indx)
    return num_perm
