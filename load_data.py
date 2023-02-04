import numpy

def load_data(file_name):
    label_list = []
    data_list = []

    with open(file_name) as file:
        for line in file:
            try:
                features_list = line.split(',')
                features = numpy.array(features_list[0:11], dtype='float32').reshape(-1,1)
                label = int(features_list[-1])
                data_list.append(features)
                label_list.append(label)
            except:
                print('Error while reading file!')
                pass

    return numpy.hstack(data_list), numpy.array(label_list, dtype=numpy.int32)
