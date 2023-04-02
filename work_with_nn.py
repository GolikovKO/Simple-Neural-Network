import numpy


def train_network(training_data_list, output_nodes, network):
    epochs = 2
    for epoch in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)


def test_network(test_data_list, network):
    score = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        network_label = numpy.argmax(outputs)

        if network_label == correct_label:
            score.append(1)
        else:
            score.append(0)

    print("Эффективность сети по тестовым данным MNIST - ", sum(score) / len(score))
