import numpy as numpy


class GeneticAlgorithm:
    # Конструктор принимающий логику алгоритма
    def __init__(self, numberOfPopulationMembers, percentOfBestOnesToLive, section, function):
        self.numberOfPopulationMembers = numberOfPopulationMembers
        self.percentOfBestOnesToLive = percentOfBestOnesToLive
        self.section = section
        self.function = function
        self.probability = 0.000001
        self.populationX = []
        self.populationY = []

    # Генерация проходимых путей

    def generateInitialPopulation(self):
        self.populationX = self.createNewPopulation()
        self.populationY = self.createNewPopulation()

    # Генерация нового прохода

    def createNewPopulation(self):
        minimum = self.section[0]
        maximum = self.section[1]
        randomCoefficients = numpy.random.random(self.numberOfPopulationMembers)
        population = minimum + (randomCoefficients * (maximum - minimum))
        return population

    # Выборка меньшего

    def getBestMembers(self):
        functionValues = self.function(self.populationX, self.populationY)
        sortedIndexes = functionValues.argsort()
        amountOfBestValues = int(len(functionValues) * self.percentOfBestOnesToLive)
        bestPopulationX = self.populationX[sortedIndexes[:amountOfBestValues]]
        bestPopulationY = self.populationY[sortedIndexes[:amountOfBestValues]]
        return [bestPopulationX, bestPopulationY]

    # Подсчет наших чисел

    def mutate(self):  # Мутация
        minimalPopulationX = numpy.min(self.populationX)  # Берем минимальное кол-во X
        minimalPopulationY = numpy.min(self.populationY)  # Берем минимальное кол-во Y
        # Cчитаем произведение наших элемментов( мутация ) для X и Y
        self.populationX += minimalPopulationX * (self.probability * numpy.random.normal(0, 0.0001, len(self.populationX)))
        self.populationY += minimalPopulationY * (self.probability * numpy.random.normal(0, 0.0001, len(self.populationY)))

    # Спуск по функции

    def crossover(self):    # скрещения
        populationXLength = len(self.populationX)  # берем длину популяции по X
        populationYLength = len(self.populationY)  # берем длину популяции по Y
        newPopulationX = numpy.zeros(self.numberOfPopulationMembers)  # создаем пустой массив для новой популяции
        newPopulationY = numpy.zeros(self.numberOfPopulationMembers)  # создаем пустой массив для новой популяции
        for i in range(populationXLength):  # Цикл прохода по всей популяции X - Y
            newPopulationX[i] = self.populationX[i]  # Переписуем эллементы
            newPopulationY[i] = self.populationY[i]
        i = populationXLength
        while i < self.numberOfPopulationMembers:   # Цикл пока не пройдемся по каждой паре

            randomMemberIndexX = numpy.random.randint(0, populationXLength - 1)  # Берем случайный индес(скрещиваем)
            firstRandomMemberX = self.populationX[randomMemberIndexX]   # Скрещиваем эллементы X
            randomMemberIndexX = numpy.random.randint(0, populationXLength - 1)  # Берем случайный индес(скрещиваем)
            secondRandomMemberX = self.populationX[randomMemberIndexX]  # Скрещиваем эллементы X
            averageMemberX = (firstRandomMemberX + secondRandomMemberX) / 2.0  #  Высчитываем среднее значение скрещивания
            # Сохраняем в новую популяцию наше значение
            newPopulationX[i] = averageMemberX
            # Повторяем такую же процедуру для Y
            randomMemberIndexY = numpy.random.randint(0, populationYLength - 1)
            firstRandomMemberY = self.populationY[randomMemberIndexY]
            randomMemberIndexY = numpy.random.randint(0, populationYLength - 1)
            secondRandomMemberY = self.populationY[randomMemberIndexY]
            averageMemberY = (firstRandomMemberY + secondRandomMemberY) / 2.0
            # Сохраняем в новую популяцию наше значение
            newPopulationY[i] = averageMemberY

            i += 1
        self.populationX = newPopulationX   # По окончанию получаем нашу новую популяцию
        self.populationY = newPopulationY

    # Логика поиска минимума

    def searchMinimum(self, iterations):
        self.generateInitialPopulation()
        partOfIterations = 0.3
        i = 0
        while i < iterations:
            tempPopulation = self.getBestMembers()
            self.populationX = tempPopulation[0]
            self.populationY = tempPopulation[1]
            self.crossover()
            self.mutate()
            i += 1
            if i > partOfIterations*iterations:
                self.probability = 0.00000001

        minimumValueIndex = self.getMinimalValueIndex()
        return self.function(self.populationX[minimumValueIndex], self.populationY[minimumValueIndex])

    # Свойство, которое хранит индекс минимума функции и сохраняет и выводит

    def getMinimalValueIndex(self):
        values = self.function(self.populationX, self.populationY)
        sortedValuesIndexes = values.argsort()
        minimumValueIndex = sortedValuesIndexes[0]
        return minimumValueIndex

    # Свойство, которое хранит аргументы x и y по нашему минимуму

    def getArgumentsOfMinimumValue(self):
        minimumValueIndex = self.getMinimalValueIndex()
        minX = self.populationX[minimumValueIndex]
        minY = self.populationY[minimumValueIndex]
        return [minX, minY]