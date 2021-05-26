from GeneticAlgorithm import GeneticAlgorithm  # Импорты модулей и алгоритма
import time
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sqrt
import Antcolony as Ant
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import basinhopping

startTime = time.time()


def function(x, y):  # Функция
    return 5 * x ** 2 + 3 * y ** 2 - np.cos(x / 2) * np.cos(y / np.sqrt(3)) + 1


numberOfPopulationMembers = 1000  # стартовые значение
percentOfBestOnesToLive = 0.8
searchingSection = [-10, 10]
# Иницииализация
GA = GeneticAlgorithm(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, function)

print("Started...")
minimumValue = GA.searchMinimum(iterations=1000)
minimumPoint = GA.getArgumentsOfMinimumValue()
print("Found minimum ", minimumValue, " at point ", minimumPoint)
print("Searching time: %s seconds." % (time.time() - startTime))


def func2d(x):
    f = 5 * x[0] ** 2 + 3 * x[1] ** 2 - np.cos(x[0] / 2) * np.cos(x[1] / np.sqrt(3)) + 1
    df = np.zeros(2)
    df[0] = 10 * x[0] + np.cos(x[1] / np.sqrt(3)) * np.sin(x[0] / 2)
    df[1] = 2. * x[1] + 0.2
    return f, df


def random_search():
    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    z = 5 * x ** 2 + 3 * y ** 2 - np.cos(x / 2) * np.cos(y / np.sqrt(3)) + 1
    min = 100
    iter = 0
    print("Our random Data\n")
    print(z)
    for i in range(len(z)):
        if z[i] < min:
            min = z[i]
            iter = i
    print("Random Minimum Function - ", min, "At x = ", x[iter], " y = ", y[iter])
    print("Searching time: %s seconds." % (time.time() - startTime))


random_search()
minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
x0 = [1.0, 1.0]
ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
print("Метод отжига: x = [%.4f, %.4f], f(x, y) = %.4f" % (ret.x[0], ret.x[1], ret.fun))


def makeData():  # Построение графика
    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    xgrid, ygrid = np.meshgrid(x, y)

    zgrid = 5 * xgrid ** 2 + 3 * ygrid ** 2 - np.cos(xgrid / 2) * np.cos(ygrid / np.sqrt(3)) + 1

    return xgrid, ygrid, zgrid


x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

axes.plot_surface(x, y, z)

pylab.show()

n = 50
m = 100
way = []
a = 0
X = np.random.uniform(a, m, n)
Y = np.random.uniform(a, m, n)
f = open('asdf.txt', 'r')
X = []
Y = []
for line in f:
    temp = line.split()
    X.append(float(temp[1]))
    Y.append(float(temp[2]))

f.close()
n = len(X)
RS = [];
RW = [];
RIB = []
s = []
for ib in np.arange(0, n, 1):
    M = np.zeros([n, n])
    for i in np.arange(0, n, 1):
        for j in np.arange(0, n, 1):
            if i != j:
                M[i, j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            else:
                M[i, j] = float('inf')
    way = []
    way.append(ib)
    for i in np.arange(1, n, 1):
        s = []
        for j in np.arange(0, n, 1):
            s.append(M[way[i - 1], j])
        way.append(s.index(min(s)))
        for j in np.arange(0, i, 1):
            M[way[i], way[j]] = float('inf')
            M[way[i], way[j]] = float('inf')
    S = sum([sqrt((X[way[i]] - X[way[i + 1]]) ** 2 + (Y[way[i]] - Y[way[i + 1]]) ** 2) for i in
             np.arange(0, n - 1, 1)]) + sqrt((X[way[n - 1]] - X[way[0]]) ** 2 + (Y[way[n - 1]] - Y[way[0]]) ** 2)
    RS.append(S)
    RW.append(way)
    RIB.append(ib)
S = min(RS)
way = RW[RS.index(min(RS))]
ib = RIB[RS.index(min(RS))]
X1 = [X[way[i]] for i in np.arange(0, n, 1)]
Y1 = [Y[way[i]] for i in np.arange(0, n, 1)]
plt.title(
    'Жадный алгоритм Общий путь-%s.Номер города-%i.Всего городов -%i.\n Координаты X,Y заданы' % (round(S, 3), ib, n),
    size=14)
plt.plot(X1, Y1, color='r', linestyle=' ', marker='o')
plt.plot(X1, Y1, color='b', linewidth=1)
X2 = [X[way[n - 1]], X[way[0]]]
Y2 = [Y[way[n - 1]], Y[way[0]]]
plt.plot(X2, Y2, color='g', linewidth=2, linestyle='-', label='Метод случайного поиска')
plt.legend(loc='best')
plt.grid(True)
plt.show()
Z = sqrt((X[way[n - 1]] - X[way[0]]) ** 2 + (Y[way[n - 1]] - Y[way[0]]) ** 2)
Y3 = [sqrt((X[way[i + 1]] - X[way[i]]) ** 2 + (Y[way[i + 1]] - Y[way[i]]) ** 2) for i in np.arange(0, n - 1, 1)]
X3 = [i for i in np.arange(0, n - 1, 1)]
plt.title('Метод случайного поиска')
plt.plot(X3, Y3, color='b', linestyle=' ', marker='o')
plt.plot(X3, Y3, color='r', linewidth=1, linestyle='-', label='Без учёта замыкающего пути - %s' % str(round(Z, 3)))
plt.legend(loc='best')
plt.grid(True)
plt.show()

f = open('asdf.txt', 'r')
distance_array = []
for line in f:
    temp = line.split()
    arr = []
    arr.append(float(temp[1]))
    arr.append(float(temp[2]))
    distance_array.append(arr)

f.close()
data = distance_array
ctys = [i for i in range(len(distance_array))]
df = pd.DataFrame(data, columns=['xcord', 'ycord'], index=ctys)
print(pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index))
dist = (distance_matrix(df.values, df.values))

problem = np.array(dist)
optimizer = Ant.AntColonyOptimizer(ants=10, evaporation_rate=.1, intensification=2, alpha=1, beta=1,
                                   beta_evaporation_rate=0, choose_best=.1)

best = optimizer.fit(problem, 100)
optimizer.plot()


def Min(lst, myindex):
    return min(x for idx, x in enumerate(lst) if idx != myindex)


def Delete(matrix, index1, index2):
    del matrix[index1]
    for i in matrix:
        del i[index2]
    return matrix


def PrintMatrix(matrix):
    print("---------------")
    for i in range(len(matrix)):
        print(matrix[i])
    print("---------------")


f = open('asdf.txt', 'r')
distance_array = []
for line in f:
    temp = line.split()
    arr = []
    arr.append(float(temp[1]))
    arr.append(float(temp[2]))
    distance_array.append(arr)

f.close()
data = distance_array
ctys = [i for i in range(len(distance_array))]
df = pd.DataFrame(data, columns=['xcord', 'ycord'], index=ctys)
# print(pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index))
dist = (distance_matrix(df.values, df.values))
dist = np.array(dist)
true_matrix = []
for i in range(len(dist)):
    temp = []
    for j in range(len(dist[0])):
        temp.append(dist[i][j])
    true_matrix.append(temp)

n = len(true_matrix)
matrix = true_matrix
H = 0
PathLenght = 0
Str = []
Stb = []
res = []
result = []
StartMatrix = []

for i in range(n):
    Str.append(i)
    Stb.append(i)

for i in range(n): StartMatrix.append(matrix[i].copy())

for i in range(n): matrix[i][i] = float('inf')

while True:
    for i in range(len(matrix)):
        temp = min(matrix[i])
        H += temp
        for j in range(len(matrix)):
            matrix[i][j] -= temp

    for i in range(len(matrix)):
        temp = min(row[i] for row in matrix)
        H += temp
        for j in range(len(matrix)):
            matrix[j][i] -= temp
    NullMax = 0
    index1 = 0
    index2 = 0
    tmp = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 0:
                tmp = Min(matrix[i], j) + Min((row[j] for row in matrix), i)
                if tmp >= NullMax:
                    NullMax = tmp
                    index1 = i
                    index2 = j
    res.append(Str[index1] + 1)
    res.append(Stb[index2] + 1)

    oldIndex1 = Str[index1]
    oldIndex2 = Stb[index2]
    if oldIndex2 in Str and oldIndex1 in Stb:
        NewIndex1 = Str.index(oldIndex2)
        NewIndex2 = Stb.index(oldIndex1)
        matrix[NewIndex1][NewIndex2] = float('inf')
    del Str[index1]
    del Stb[index2]
    matrix = Delete(matrix, index1, index2)
    if len(matrix) == 1: break

# Формируем порядок пути
for i in range(0, len(res) - 1, 2):
    if res.count(res[i]) < 2:
        result.append(res[i])
        result.append(res[i + 1])
for i in range(0, len(res) - 1, 2):
    for j in range(0, len(res) - 1, 2):
        if result[len(result) - 1] == res[j]:
            result.append(res[j])
            result.append(res[j + 1])
print("Генетический алгоритм")
print("----------------------------------")

# Считаем длину пути
for i in range(0, len(result) - 1, 2):
    if i == len(result) - 2:
        PathLenght += StartMatrix[result[i] - 1][result[i + 1] - 1]
        PathLenght += StartMatrix[result[i + 1] - 1][result[0] - 1]
    else:
        PathLenght += StartMatrix[result[i] - 1][result[i + 1] - 1]
print("Расстояние от 1 до последнего", PathLenght)
