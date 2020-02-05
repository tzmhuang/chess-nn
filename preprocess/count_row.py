def count_row(file_n, start, n):
    count = 0
    for i in range(n-1):
        index = start + 250*i
        temp = pd.read_csv("//Volumes/DiskA/atk_map/atk_map_{}_{}".format(file_n,index))
        count = count + temp.shape[0]
    index = start+810
    temp = pd.read_csv("//Volumes/DiskA/atk_map/atk_map_{}_{}".format(file_n,index))
    count = count + temp.shape[0]
    return count

def sum_row():
    count = 0
    for i in range(1,34):
        temp = pd.read_csv("//Volumes/DiskA/atk_map/atk_map_{}".format(i))
        count = count + temp.shape[0]
        print (i)
        del(temp)
    return count
