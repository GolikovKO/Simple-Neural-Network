import numpy
import matplotlib.pyplot


def build_graph(data_list):
    #values = data_list[0].split(',')
    image_array = numpy.asfarray(data_list[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
