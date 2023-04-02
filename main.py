import glob
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy

from neural_network import NeuralNetwork
from work_with_file import read_data
from work_with_nn import train_network, test_network


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Обучаем нейронную сеть
    training_data_list = read_data("mnist_dataset/mnist_train.csv")
    train_network(training_data_list, output_nodes, network)

    # Тестируем нейронную сеть на данных mnist
    test_data_list = read_data("mnist_dataset/mnist_test.csv")
    test_network(test_data_list, network)

    # Подготавливаем данные моих цифр
    my_dataset = []
    for image_file_name in glob.glob('numbers/my_paint_images/?.png'):
        label_of_number = int(image_file_name[-5:-4])

        img_array = imageio.imread(image_file_name, as_gray=True)
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01

        record = numpy.append(label_of_number, img_data)
        my_dataset.append(record)

    # Тестируем сеть моими цифрами
    correct_numbers_count = 0
    for item in range(10):
        plt.imshow(my_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')

        correct_label = my_dataset[item][0]
        inputs = my_dataset[item][1:]
        outputs = network.query(inputs)

        network_label = numpy.argmax(outputs)
        print("-----")
        print("Цифра", int(correct_label))
        print("Нейросеть говорит это цифра", network_label)

        if network_label == correct_label:
            print("Правильно!")
            correct_numbers_count += 1
        else:
            print("Ошибка!")

    print("Эффективность сети -", correct_numbers_count / 10)


if __name__ == '__main__':
    main()
