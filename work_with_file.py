def read_data(path):
    file = open(path, 'r')
    data = file.readlines()
    file.close()
    return data
