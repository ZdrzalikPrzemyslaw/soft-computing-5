import random

import matplotlib.image
import numpy
import time

import matplotlib.pyplot as plt

from random import randrange

import os
import imageio
import numpy as np

# wspolczynnik uczenia
eta = 0.001
# momentum
alfa = 0.1

ERROR_FILE = "mean_squared_error.txt"


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        if self.is_bias:
            return "hidden_layer (wiersze - neurony) :\n" + str(
                self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(
                self.output_layer) + "\nbiashiddenlayer\n" + str(
                self.bias_hidden_layer) + "\nbiasoutputlayer\n" + str(self.bias_output_layer)
        return "hidden_layer (wiersze - neurony) :\n" + str(
            self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer)

    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs, is_bias):
        # czy uruchomilismy bias
        self.is_bias = is_bias
        self.iteration = 0
        # warstwy ukryta i wyjściowa oraz odpowiadające im struktury zapisujące zmianę wagi w poprzedniej iteracji, używane do momentum
        self.hidden_layer = (2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)).T - 1)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs, number_of_neurons_hidden_layer)).T
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)).T - 1
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer, number_of_neurons_output)).T
        # jesli wybralismy że bias ma byc to tworzymy dla każdej warstwy wektor wag biasu
        if is_bias:
            self.bias_hidden_layer = (2 * numpy.random.random(number_of_neurons_hidden_layer) - 1)
            self.bias_output_layer = (2 * numpy.random.random(number_of_neurons_output) - 1)
        # jesli nie ma byc biasu to tworzymy takie same warstwy ale zer. Nie ingerują one potem w obliczenia w żaden sposób
        else:
            self.bias_hidden_layer = numpy.zeros(number_of_neurons_hidden_layer)
            self.bias_output_layer = numpy.zeros(number_of_neurons_output)
        # taka sama warstwa delty jak dla layerów
        self.bias_output_layer_delta = numpy.zeros(number_of_neurons_output)
        self.bias_hidden_layer_delta = numpy.zeros(number_of_neurons_hidden_layer)

    # Wzór funkcji
    def linear_fun(self, inputcik):
        return inputcik

    def linear_fun_deriative(self, inputcik):
        return 1

    # Wzór funkcji
    def sigmoid_fun(self, inputcik):
        return 1 / (1 + numpy.exp(-inputcik))

    def sigmoid_fun_deriative(self, inputcik):
        return inputcik * (1 - inputcik)

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    # Jak wiadomo bias to przesunięcie wyniku o stałą więc jeżeli wybraliśmy że bias istnieje to on jest po prostu dodawany do odpowiedniego wyniku iloczynu skalarnego
    def calculate_outputs(self, inputs):

        hidden_layer_output = self.linear_fun(numpy.dot(inputs, self.hidden_layer.T) + self.bias_hidden_layer)
        output_layer_output = self.linear_fun(
            numpy.dot(hidden_layer_output, self.output_layer.T) + self.bias_output_layer)

        return hidden_layer_output, output_layer_output

    # trening, tyle razy ile podamy epochów
    # dla każdego epochu shufflujemy nasze macierze i przechodzimy przez nie po każdym wierszu z osobna
    def train(self, inputs, expected_outputs, epoch_count, fileName):
        error_list = []
        for it in range(epoch_count):

            # Shuffle once each iteration
            joined_arrays = numpy.copy(inputs)
            numpy.random.shuffle(joined_arrays)

            mean_squared_error = 0
            ite = 0

            for k, j in zip(joined_arrays, joined_arrays):

                hidden_layer_output, output_layer_output = self.calculate_outputs(k)

                # błąd dla wyjścia to różnica pomiędzy oczekiwanym wynikiem a otrzymanym
                output_error = output_layer_output - j
                mean_squared_error += output_error.dot(output_error) / 2
                ite += 1

                # output_delta - współczynnik zmiany wagi dla warstwy wyjściowej. Otrzymujemy jeden współczynnik dla każdego neronu.
                # aby potem wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                output_delta = output_error * self.linear_fun_deriative(output_layer_output)

                # korzystamy z wcześniej otrzymanego współczynniku błędu aby wyznaczyć błąd dla warstwy ukrytej
                hidden_layer_error = output_delta.T.dot(self.output_layer)
                # jak dla warstwy wyjściowej hidden_layer_delta jest jeden dla każdego neuronu i
                # aby wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                hidden_layer_delta = hidden_layer_error * self.linear_fun_deriative(hidden_layer_output)

                output_layer_adjustment = []
                for i in output_delta:
                    output_layer_adjustment.append(hidden_layer_output * i)
                output_layer_adjustment = numpy.asarray(output_layer_adjustment)

                hidden_layer_adjustment = []
                for i in hidden_layer_delta:
                    hidden_layer_adjustment.append(k * i)
                hidden_layer_adjustment = numpy.asarray(hidden_layer_adjustment)

                # jeżeli wybraliśmy żeby istniał bias to teraz go modyfikujemy
                if self.is_bias:
                    hidden_bias_adjustment = eta * hidden_layer_delta + alfa * self.bias_hidden_layer_delta
                    output_bias_adjustment = eta * output_delta + alfa * self.bias_output_layer_delta
                    self.bias_hidden_layer -= hidden_bias_adjustment
                    self.bias_output_layer -= output_bias_adjustment
                    self.bias_hidden_layer_delta = hidden_bias_adjustment
                    self.bias_output_layer_delta = output_bias_adjustment

                # wyliczamy zmianę korzystając z współczynnika uczenia i momentum
                hidden_layer_adjustment = eta * hidden_layer_adjustment + alfa * self.delta_weights_hidden_layer
                output_layer_adjustment = eta * output_layer_adjustment + alfa * self.delta_weights_output_layer

                # modyfikujemy wagi w warstwach
                self.hidden_layer -= hidden_layer_adjustment
                self.output_layer -= output_layer_adjustment

                # zapisujemy zmianę wag by użyć ją w momentum
                self.delta_weights_hidden_layer = hidden_layer_adjustment
                self.delta_weights_output_layer = output_layer_adjustment

            mean_squared_error = mean_squared_error / ite
            error_list.append(mean_squared_error)

        # po przejściu przez wszystkie epoki zapisujemy błędy średniokwadratowe do pliku
        with open(fileName, "w") as file:
            for i in error_list:
                file.write(str(i) + "\n")


# otwieramy plik errorów i go plotujemy
def plot_file(fileName):
    with open(fileName, "r") as file:
        lines = file.read().splitlines()
    values = []
    ilosc = []
    liczba = 1
    for i in lines:
        values.append(float(i))
        liczba += 1
        ilosc.append(liczba)

    # plt.plot(values, 'o', markersize=1)
    plt.xlabel('Iteration')
    plt.ylabel('Error for epoch')
    plt.title("Mean square error change")
    plt.plot(ilosc, values)
    plt.show()


def main():
    files = []
    path = "./images"
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        files.append(imageio.imread(os.path.join(path, '0' + i.__str__() + '.bmp')).flatten())

    # TODO: podzielic obraz na 128 losowych fragmentow 8x8
    #  uczyc w losowej kolejnosci

    random.shuffle(files)

    test_per_photo = 128
    chunk_size = 64
    hidden_neurons_count = 32

    train = []
    test = []
    for i in range(2):
        for j in range(test_per_photo):
            random_num = randrange(512 * 512 - chunk_size)
            if random_num < 0:
                random_num = 0
            train.append(files[i][random_num:random_num + chunk_size])
    for i in range(2, 8):
        test.append([])
        for j in range(0, 512 * 512, chunk_size):
            test[i - 2].append(files[i][j:j + chunk_size])

    for i, j in enumerate(train):
        train[i] = train[i] * (1 / 255)

    for ite1, i in enumerate(test):
        for ite2, j in enumerate(i):
            test[ite1][ite2] = test[ite1][ite2] * (1 / 255)

    Network = NeuralNetwork(number_of_neurons_hidden_layer=hidden_neurons_count, is_bias=True,
                            number_of_neurons_output=chunk_size, number_of_inputs=chunk_size)
    iterations = 1000

    # dane wejściowe, dane wyjściowe, ilość epochów

    Network.train(train, train, iterations,
                  ERROR_FILE)

    plot_file(ERROR_FILE)

    res = []
    for _ in test:
        res.append([])

    for ite1, i in enumerate(test):
        for ite2, j in enumerate(i):
            res[ite1].append(Network.calculate_outputs(j)[1])

    pics = []
    for _ in res:
        pics.append([])
    for ite1, i in enumerate(res):
        for ite2, j in enumerate(i):
            for k in j:
                pics[ite1].append(int(k * 255))

    pics_reshape = []
    for i in pics:
        pics_reshape.append(np.reshape(i, (512, 512)))
    xd = 12

    for i, j in enumerate(pics_reshape):
        imageio.imsave('./out/' + str(i) + '.bmp', j)


if __name__ == "__main__":
    main()
