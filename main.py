import glob
import matplotlib
import imageio.v2 as imageio
import numpy

from neural_network import NeuralNetwork
from graph import build_graph


def main():
    #data = open("mnist_dataset/mnist_train_100.csv", 'r')
    #list = data.readlines()
    #build_graph(list)

    #data.close()
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 2
    for epoch in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)

    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    #all_values = test_data_list[0].split(',')
    #print(all_values[0])
    #build_graph(all_values)

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        #print(correct_label, ' - истинный маркер.')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        label = numpy.argmax(outputs)
        #print(label, ' - ответ сети.')

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print("эффективность сети - ", scorecard_array.sum() / scorecard_array.size)

    our_own_dataset = []

    # load the png image data as test data set
    for image_file_name in glob.glob('numbers/2828_my_own_?.png'):
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])

        # load image data from png files into an array
        print("loading ... ", image_file_name)
        img_array = imageio.imread(image_file_name, as_gray=True)

        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))

        # append label and image data  to test data set
        record = numpy.append(label, img_data)
        our_own_dataset.append(record)

        pass

    item = 0

    # plot image
    #for item in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')

    # correct answer is first value
    correct_label = our_own_dataset[item][0]
    # data is remaining values
    inputs = our_own_dataset[item][1:]

    # query the network
    outputs = network.query(inputs)
    #print(outputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("network says ", label)
    print("correct label is ", correct_label)
    # append correct or incorrect to list
    if (label == correct_label):
        print("match!")
    else:
        print("no match!")
        pass


if __name__ == '__main__':
    main()
