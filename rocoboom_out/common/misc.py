def get_entry(arr, i, j, p, m):
    # Helper function to deal with array indexing
    # arr is the array e.g. fig, arr = plt.subplots(nrows=p, ncols=m)
    # i, j are the row, column index
    # p, m are the number of rows, columns respectively
    if p > 1:
        if m > 1:
            ax = arr[i, j]
        else:
            ax = arr[i]
    else:
        if m > 1:
            ax = arr[j]
        else:
            ax = arr
    return ax