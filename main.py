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
        print(correct_label, ' - истинный маркер.')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        label = numpy.argmax(outputs)
        print(label, ' - ответ сети.')

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print("эффективность сети - ", scorecard_array.sum() / scorecard_array.size)


if __name__ == '__main__':
    main()
