def reduce_list(input_list, num_segm):
    # Sort the input list in ascending order
    input_list = sorted(input_list)
    
    N = num_segm  # Desired number of segments
    m = len(input_list)  # Length of the input list
    M = round(m / N + 0.5) - 1  # Calculate the size of each segment
    r = m - N * M  # Calculate the remainder to distribute across segments

    # Initialize the result list with the first element
    reduced_list = [input_list[0]]
    
    # Add elements to the reduced list based on the calculated size and remainder
    for i in range(N + 1):
        reduced_list.append(input_list[i * M + r - 1])
    
    return reduced_list
