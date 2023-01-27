def reduction(I1):

    '''

    Reduction of number of thresholds
    Input:
    I1: array of original thresholds
    Output:
    B: array of reduced thresholds

    length of piece = round(len(I1)/499 + 0.5) - 1
    Here we divide the given array into 499 equally distanced pieces and set x_0 = len(I1) - 499 * length of piece
    Thus x_i = x_0 + i * length of piece, for i=1, 2, ..., 499



    '''


    I1 = sorted(I1)
    N = 499
    m = len(I1)
    M = round(m / N + 0.5) - 1   # the length of each piece
    r = m - N * M                # x_0
    B = list()
    for i in range(N + 1):
        B.append(I1[(i) * M + r - 1])

    return B