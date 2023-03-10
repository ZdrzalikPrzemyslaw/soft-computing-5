import os
from random import randrange

import imageio
import matplotlib.pyplot as plt
import numpy
import numpy as np

# wspolczynnik uczenia
eta = 0.00000001
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
        weight_possible = 0.2
        weight_possible2 = 0.1
        # warstwy ukryta i wyjściowa oraz odpowiadające im struktury zapisujące zmianę wagi w poprzedniej iteracji, używane do momentum
        self.hidden_layer = (weight_possible * numpy.random.random(
            (number_of_inputs, number_of_neurons_hidden_layer)).T - weight_possible2)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs, number_of_neurons_hidden_layer)).T
        self.output_layer = weight_possible * numpy.random.random(
            (number_of_neurons_hidden_layer, number_of_neurons_output)).T - weight_possible2
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer, number_of_neurons_output)).T
        # jesli wybralismy że bias ma byc to tworzymy dla każdej warstwy wektor wag biasu
        if is_bias:
            self.bias_hidden_layer = (
                    weight_possible * numpy.random.random(number_of_neurons_hidden_layer) - weight_possible2)
            self.bias_output_layer = (
                    weight_possible * numpy.random.random(number_of_neurons_output) - weight_possible2)
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
        ilosc.append(liczba)
        liczba += 1

    # plt.plot(values, 'o', markersize=1)
    plt.xlabel('Iteration')
    plt.ylabel('Error for epoch')
    plt.title("Mean square error change")
    plt.plot(ilosc, values)
    plt.show()


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def main(size):
    numpy.random.seed(0)
    files = []
    path = "./images"
    # pobieranie zdjec z plikow
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        files.append(np.asarray(imageio.imread(os.path.join(path, '0' + i.__str__() + '.bmp'))).astype(int))

    _files = []

    # zmiana zdjec na fragmenty 8 na 8
    for i in files:
        _files.append(blockshaped(i, 8, 8))

    # splaszczenie wektorów (teraz z 8 na 8 jest 64 na 1)
    flat = []
    for ite1, i in enumerate(_files):
        flat.append([])
        for ite, j in enumerate(i):
            flat[ite1].append(j.flatten())

    averages = []
    flat2 = np.asarray(flat)
    for i in flat2:
        averages.append(np.mean(i, axis=0))

    averWages = numpy.asarray(averages)

    for ite, val in enumerate(flat):
        for ite2, val2 in enumerate(val):
            flat[ite][ite2] = flat[ite][ite2] - averages[ite]

    test_per_photo = 1024
    chunk_size = 64
    hidden_neurons_count = size

    # Stworzenie wektorów treningowyvch i testowych
    train = []
    test = []
    for i in range(3, 5):
        for j in range(test_per_photo):
            random_num = randrange(4096)
            if random_num < 0:
                random_num = 0
            train.append(flat[i][random_num])
    for i in range(0, 8):
        test.append([])
        for j in range(0, 4096):
            test[i].append(flat[i][j])

    # # Normalizacja wartosci w wektorach
    # for i, j in enumerate(train):
    #     train[i] = train[i] * (1 / 255)
    #
    # for ite1, i in enumerate(test):
    #     for ite2, j in enumerate(i):
    #         test[ite1][ite2] = test[ite1][ite2] * (1 / 255)

    Network = NeuralNetwork(number_of_neurons_hidden_layer=hidden_neurons_count, is_bias=False,
                            number_of_neurons_output=chunk_size, number_of_inputs=chunk_size)
    iterations = 1000

    # dane wejściowe, dane wyjściowe, ilość epochów

    Network.train(train, train, iterations,
                  ERROR_FILE)

    plot_file(ERROR_FILE)

    res = []

    for ite1, i in enumerate(test):
        res.append([])
        for ite2, j in enumerate(i):
            res[ite1].append(Network.calculate_outputs(j)[1] + averages[ite1])

    reshaped = []
    for _ in res:
        reshaped.append([])
    for ite1, i in enumerate(res):
        for ite2, j in enumerate(i):
            reshaped[ite1].append(np.reshape(j, (8, 8)))

    joined = []
    for i in reshaped:
        for j in range(0, 4096, 64):
            tmp = []
            for k in range(64):
                tmp.append(i[j + k])
            joined.append(np.concatenate(tmp, axis=1))

    joined2 = []
    for i in range(0, len(joined), 64):
        tmp = []
        for k in range(64):
            tmp.append(joined[i + k])
        joined2.append(np.concatenate(tmp, axis=0))

    # pics = []
    # for _ in res:
    #     pics.append([])
    # for ite1, i in enumerate(res):
    #     for ite2, j in enumerate(i):
    #         for k in j:
    #             pics[ite1].append(int(k * 255))
    #
    # pics_reshape = []
    # for i in pics:
    #     pics_reshape.append(np.reshape(i, (512, 512)))
    # xd = 12

    # for i, j in enumerate(pics_reshape):
    #     imageio.imsave('./out/' + str(i) + '.bmp', j)
    name = "./out/" + hidden_neurons_count.__str__()
    os.mkdir(name)
    name += "/"
    for i, j in enumerate(joined2):
        imageio.imsave(name + '0' + str(i + 1) + '.bmp', j)


if __name__ == "__main__":
    for i in [1, 2, 3, 4, 8, 16, 32, 64, 128]:
        print(i)
        main(i)
