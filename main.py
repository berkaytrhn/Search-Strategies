from random import shuffle
from sys import setrecursionlimit
import time
import math
import numpy as np
from tqdm import tqdm

# setting recursion limit to high level
setrecursionlimit(10 ** 6)

class Cell:
    # maze creation
    connected = None
    row = None
    column = None
    connectedCells = None

    # iterative deepening search
    pathNext = None

    # uniform cost search
    costFromStart = None

    # A* search
    heuristic = None

    # both A* and Uniform Cost Search
    parentCell = None

    def __repr__(self):
        return f"({self.row},{self.column})"

    def __init__(self, row, column):
        self.connected = False
        self.row = row
        self.column = column
        self.connectedCells = []
        self.costFromStart = 0

    def defineHeuristic(self, which, row, column):
        # select heuristic value according to given text
        if which == "euclidean":
            self.heuristic = math.sqrt(((self.row-row)**2)+((self.column-column)**2))
        elif which == "manhattan":
            self.heuristic = abs(self.row-row) + abs(self.column-column)

class Maze:
    width = None
    height = None
    cells = None
    stack = None

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = []
        self.stack = []




    def visualizeHelper(self, array):
        # put blank space string between to connected nodes
        for i in range(len(array)):
            for k in range(len(array[i])):
                value = array[i][k]
                if value == "x":
                    if array[i + 1][k] == " ":
                        array[i + 2][k] = " "
                    if array[i - 1][k] == " ":
                        array[i - 2][k] = " "
                    if array[i][k + 1] == " ":
                        array[i][k + 2] = " "
                    if array[i][k - 1] == " ":
                        array[i][k - 2] = " "

        # put blank space string over nodes
        for i in range(len(array)):
            for k in range(len(array[i])):
                if array[i][k] == "x":
                    array[i][k] = " "

    def visualizeMaze(self):
        # creating array for visualization
        tempArray = []
        for i in range(len(self.cells)*3 + 3):
            tempArray.append(" ")
        visualizeArray = np.array(
            [
                tempArray
            ]
        )
        for i in range(len(self.cells)):
            row = self.cells[i]
            visualizeRow = np.array(
                [
                    [" ", " ", " "],
                    [" ", " ", " "],
                    [" ", " ", " "]
                ]
            )

            for k in range(len(row)):
                cell = self.cells[i][k]
                tempArray = [
                    ["B", "B", "M"],
                    ["4", "0", "5"],
                    ["F", "A", "I"]
                ]
                for l in range(3):
                    for m in range(3):
                        tempArray[l][m] = "#"

                tempArray[1][1] = "x"

                for neighbor in cell.connectedCells:
                    row = neighbor.row
                    column = neighbor.column
                    if row == cell.row:
                        if column > cell.column:
                            tempArray[1][2] = " "
                        else:
                            tempArray[1][0] = " "
                    elif column == cell.column:
                        if row > cell.row:
                            tempArray[2][1] = " "
                        else:
                            tempArray[0][1] = " "



                tempArray = np.array(tempArray)
                # print(tempArray)
                visualizeRow = np.hstack((visualizeRow, tempArray))

            visualizeArray = np.vstack((visualizeArray, visualizeRow))

        # deleting unnecessary rows and columns
        visualizeArray = np.delete(visualizeArray, 0, axis=0)
        visualizeArray = np.delete(visualizeArray, 0, axis=1)
        visualizeArray = np.delete(visualizeArray, 0, axis=1)
        visualizeArray = np.delete(visualizeArray, 0, axis=1)
        self.visualizeHelper(visualizeArray)

        # writing to a file
        file = open(f"./{self.width}x{self.height}_maze.txt", "w")
        for i in visualizeArray:
            for k in i:
                file.write(f"{k}{' '}")
            file.write("\n")
        file.close()

    def defineHeuristics(self, which):
        # defining heuristic for every cell in maze
        for row in self.cells:
            for cell in row:
                cell.defineHeuristic(which, self.height - 1, self.width - 1)

    def createCells(self):
        # creating unconnected cells
        for i in tqdm(range(self.height)):
            row = []
            for j in range(self.width):
                row.append(Cell(i, j))
            self.cells.append(row)

    def createMaze(self):
        # starter function for randomizedDFS
        # top left corner
        startCell = self.cells[0][0]
        self.stack.append(startCell)
        self.randomizedDFS()

    def randomizedDFS(self):
        # iterative randomized dfs algorithm
        cell = self.stack.pop()
        self.randomNeighbor(cell)
        cell.connected = True
        while len(self.stack) != 0:

            couple = self.stack.pop()
            cell = couple[1]
            if not cell.connected:
                cell.connected = True
                self.randomNeighbor(cell)
                self.connectCells(couple[0], cell)
            else:
                pass

    def connectCells(self, cell, neighbor):
        # make connection between cells
        cell.connectedCells.append(neighbor)

    def checkCells(self, cell1, cell2):
        # check if given cells are equal or not
        if (cell1.row == cell2.row) and (cell1.column == cell2.column):
            return True
        else:
            return False

    def selectValidNeighbors(self, cell):
        # selecting neighbor cells according to row-column numbers
        row = cell.row
        column = cell.column

        tempNeighborIndexes = [
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1)
        ]
        neighborCells = []
        for couple in tempNeighborIndexes:
            newRow = couple[0]
            newColumn = couple[1]
            if not ((newRow < 0) or (newColumn < 0) or (newRow >= self.height) or (newColumn >= self.width)):
                # valid index value, append to neighbors
                candidateCell = self.cells[newRow][newColumn]
                if not candidateCell.connected:
                    neighborCells.append(self.cells[newRow][newColumn])

        return neighborCells

    def randomNeighbor(self, cell):
        # putting valid neighbors into stack randomly
        neighborCells = self.selectValidNeighbors(cell)

        # edit array as it will keep connection data
        for i in range(len(neighborCells)):
            neighborCells[i] = [cell, neighborCells[i]]

        # shuffling array
        shuffle(neighborCells)

        # printing unvisited neighbors
        # print(f"{cell.row} {cell.column}, {neighborCells}")

        self.stack.extend(neighborCells)

class IterativeDeepeningSearch:
    graph = None
    stack = None
    path = None
    expanded = None
    pathLength = None

    def __init__(self, graph):
        self.graph = graph
        self.stack = []
        self.expanded = 0
        self.pathLength = 1


    def depthFirstSearch(self, source, target, depth):
        # dfs part of ids algorithm
        self.expanded += 1
        if self.graph.checkCells(source, target):
            return True

        if depth <= 0:
            return False

        for cell in source.connectedCells:
            source.pathNext = cell
            if self.depthFirstSearch(cell, target, depth - 1):
                return True
        return False


    def iterativeDeepeningSearch(self, source, target, depthLimit):
        # depth limit part of  recursive ids algorithm
        for i in range(depthLimit):
            returnedValue = self.depthFirstSearch(source, target, i)

            if returnedValue:
                return True
        return False


    def printPath(self):
        # printing found path to file
        source = self.graph.cells[0][0]
        target = self.graph.cells[self.graph.height - 1][self.graph.width - 1]

        file = open(f"./ids_{self.graph.width}x{self.graph.height}_path.txt", mode="w")
        try:
            while not self.graph.checkCells(source, target):
                self.pathLength += 1
                file.write(f"({source.row},{source.column})->")
                source = source.pathNext
            file.write(f"({target.row},{target.column})")
            file.close()
        except AttributeError:
            pass

class UniformCostSearch:
    graph = None
    queue = None
    expanded = None
    pathLength = None

    def __init__(self, graph):
        self.queue = []
        self.graph = graph
        self.expanded = 0

    def uniformCostSearch(self, source, target):
        # uniform cost search implementation with priority queue
        for i in source.connectedCells:
            self.queue.append([source, [i, source.costFromStart + 1]])
        self.expanded += 1

        while len(self.queue) != 0:
            # sorting queue according to path cost by reverse order
            self.queue = sorted(self.queue, key=lambda element: element[1][1], reverse=True)

            # expanding node
            temp = self.queue.pop()

            # increasing expanded number
            self.expanded += 1
            temp[1][0].parentCell = temp[0]
            source = temp[1][0]
            source.costFromStart = temp[1][1]

            if self.graph.checkCells(source, target):
                return True, source.costFromStart

            for i in source.connectedCells:
                self.queue.append([source, [i, source.costFromStart + 1]])

        return False

    def printPath(self):
        # writing found path to file
        target = self.graph.cells[0][0]
        source = self.graph.cells[self.graph.height - 1][self.graph.width - 1]
        path = []
        while not self.graph.checkCells(source, target):
            path.append(source)
            source = source.parentCell
        path.append(target)

        path.reverse()
        file = open(f"./ucs_{self.graph.width}x{self.graph.height}_path.txt", mode="w")
        self.pathLength = len(path)
        for i in range(len(path)):
            if i == len(path)-1:
                file.write(f"({path[i].row},{path[i].column})")
            else:
                file.write(f"({path[i].row},{path[i].column})->")
        file.close()

class AStarSearch:
    graph = None
    queue = None
    expanded = None
    pathLength = None

    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.expanded = 0


    def aStarSearch(self, source, target):

        for i in source.connectedCells:
            self.queue.append([source, [i, source.costFromStart + 1, source.costFromStart + i.heuristic]])
        self.expanded += 1

        while len(self.queue) != 0:

            # queue sorted by (heuristic + path cost)
            self.queue = sorted(self.queue, key=lambda element: element[1][2], reverse=True)

            # expanding node
            temp = self.queue.pop()

            # increasing expanded number
            self.expanded += 1
            temp[1][0].parentCell = temp[0]
            source = temp[1][0]
            source.costFromStart = temp[1][1]

            if self.graph.checkCells(source, target):
                return True, source.costFromStart

            for i in source.connectedCells:
                self.queue.append([source, [i, source.costFromStart + 1, source.costFromStart + i.heuristic]])

    def printPath(self, heuristic):

        target = self.graph.cells[0][0]
        source = self.graph.cells[self.graph.height - 1][self.graph.width - 1]
        path = []
        while not self.graph.checkCells(source, target):
            path.append(source)
            source = source.parentCell
        path.append(target)

        path.reverse()
        file = open(f"./astar_{heuristic}_{self.graph.width}x{self.graph.height}_path.txt", mode="w")
        self.pathLength = len(path)
        for i in range(len(path)):
            if i == len(path) - 1:
                file.write(f"({path[i].row},{path[i].column})")
            else:
                file.write(f"({path[i].row},{path[i].column})->")
        file.close()

def IDS(maze):
    #IDS Method

    # top left of maze
    source = maze.cells[0][0]
    # right bottom of maze
    target = maze.cells[maze.height - 1][maze.width - 1]

    searcher = IterativeDeepeningSearch(maze)
    startTime = time.time()

    returnedValue = False
    try:
        returnedValue = searcher.iterativeDeepeningSearch(source, target, maze.width * maze.height)
    except MemoryError:
        endTime = time.time()
        searcher.printPath()

        print("****************")
        print("Memory Error")
        print(f"Took {round(endTime - startTime, 7)} seconds.")
        print(f"Path Length Until Now: {searcher.pathLength}")
        print(f"Expanded Node Number Until Now: {searcher.expanded}")


    endTime = time.time()
    if returnedValue:
        # path found
        searcher.printPath()

        # printing operations
        print("****************")
        print(f"Took {round(endTime - startTime, 7)} seconds.")
        print("Path Found For Iterative Deepening Search")
        print(f"Path Length: {searcher.pathLength}")
        print(f"Expanded Node Number: {searcher.expanded}")


    else:
        print("Could not find path with given depth limit.")

def UCS(maze):
    # UCS method


    # top left of maze
    source = maze.cells[0][0]
    # right bottom of maze
    target = maze.cells[maze.height - 1][maze.width - 1]

    searcher = UniformCostSearch(maze)
    startTime = time.time()
    returnedValue, cost = searcher.uniformCostSearch(source, target)
    endTime = time.time()
    if returnedValue:
        # path found

        searcher.printPath()

        # printing operations
        print("****************")
        print(f"Took {round(endTime - startTime, 7)} seconds.")
        print("Path Found For Uniform Cost Search")
        print(f"Path Length: {searcher.pathLength}")
        print(f"Expanded Node Number: {searcher.expanded}")
    else:
        # optimal search algorithm, will find a path; no else condition
        pass

def ASTAR(maze, heuristic):
    # A* method

    # top left of maze
    source = maze.cells[0][0]
    # right bottom of maze
    target = maze.cells[maze.height - 1][maze.width - 1]

    searcher = AStarSearch(maze)
    startTime = time.time()
    returnedValue, cost = searcher.aStarSearch(source, target)
    endTime = time.time()
    if returnedValue:
        # path found
        searcher.printPath(heuristic)

        print("****************")
        print(f"Took {round(endTime - startTime, 7)} seconds.")
        print(f"Path Found For A* with {heuristic.capitalize()} Distance")
        print(f"Path Length: {searcher.pathLength}")
        print(f"Expanded Node Number: {searcher.expanded}")
    else:
        # optimal search algorithm, will find a path; no else condition
        pass

def generalTestFunction(maze):
    IDS(maze)
    UCS(maze)
    maze.defineHeuristics("euclidean")
    ASTAR(maze, "euclidean")
    maze.defineHeuristics("manhattan")
    ASTAR(maze, "manhattan")
    maze.visualizeMaze()

def main():
    maze = None
    userInput = input("1)Iterative Deepening Search\n2)Uniform Cost Search\n3)A* Search with Euclidean Distance\n4)A* Search with Manhattan Distance\n5)Test All Algorithms For a Single Maze\nSelect an Algorithm to Search:")
    size = input("Maze size(widthxheight): ").lower().split("x")
    width = int(size[0])
    height = int(size[1])
    if userInput not in ["1", "2", "3", "4", "5"]:
        print("Invalid Choice!")
    else:
        print("Creating Maze...")
        maze = Maze(width, height)
        maze.createCells()
        maze.createMaze()

    if userInput == "1":
        IDS(maze)
        maze.visualizeMaze()
    elif userInput == "2":
        UCS(maze)
        maze.visualizeMaze()

    elif userInput == "3":
        maze.defineHeuristics("euclidean")
        ASTAR(maze, "euclidean")
        maze.visualizeMaze()

    elif userInput == "4":
        maze.defineHeuristics("manhattan")
        ASTAR(maze, "manhattan")
        maze.visualizeMaze()

    elif userInput == "5":
        generalTestFunction(maze)

main()
