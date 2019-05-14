from typing import List, Any

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

K = 50
# The number of Cluster

N = 200
# The number of iterate times

Distance_per_point = 0.5
# set a point per airline

future_num = 3
# the number of futures' number

w1 = 1.5
# the importance of distance between two points

w2 = 100
# the importance of airplanes number

class Airplane:
    name = "null"
    # the name of this point
    type = 0
    # the type  1 for airlines  2 for airport

    belong = np.zeros(K, dtype=np.int_, order='C')
    # the number belongs to every cluster
    # would be 00001000 e.t.c for point
    # would be 10204005 e.t.c for airlines

    cluster = 1
    # belong to which Cluster

    Fcoord = np.zeros(2, dtype=np.float_, order='C')
    Tcoord = np.zeros(2, dtype=np.float_, order='C')
    # Coord from && to

    data = np.zeros(34, dtype=np.float_, order='C')

    def __init__(self, tlist):
        self.name = tlist[0]
        self.type = int(tlist[1])
        self.Fcoord = np.asarray(tlist[2:4], dtype=np.float_)
        self.Tcoord = np.asarray(tlist[4:6], dtype=np.float_)
        self.data = np.asarray(tlist[6:], dtype=np.float_)

    # update Cluster
    def updataBelong(self):
        tCluster = self.cluster
        temp = 0
        for index, item in enumerate(self.belong):
            if (temp < item):
                temp = item
                tCluster = index
        self.cluster = tCluster

    def display(self):
        print(self.name, self.type, self.Fcoord, self.Tcoord, self.data)


class Point:

    def __init__(self, air: Airplane, Coord=np.zeros(2, dtype=np.float_), parent=0):
        self.data = air.data
        self.name = air.name
        self.coord = np.zeros(2, dtype=np.float_)
        self.coord = Coord
        self.parent = parent
        self.type = air.type

    def display(self):
        print(self.name, self.coord, self.belongsTo, self.data, self.color)

    def mkData(self, left, right):
        self.cData = np.zeros(36, dtype=np.float_)
        self.cData[:2] = self.coord * w1
        self.cData[2+left*2:4+right*2] = self.data[left*2:2+right*2]
        for index,i in enumerate(self.cData):
            if index > 1 and index % 2 == 1:
                self.cData[index] *= w2


airsList = []  # type: list[Airplane]
pointList = []  # type: list[point]
PointN = 0
ansList = []


# read data from file
def readFile():
    file = open('t.txt')
    FileContex = file.readlines()
    for i in FileContex:
        tlist = i.split(' ')
        airsList.append(Airplane(tlist))


# calc Euclidean Distance for vector
def calcDis(Veca, Vecb):
    return np.linalg.norm(Veca - Vecb)


def mkPoint():
    global PointN
    global pointList
    global airsList
    for index, i in enumerate(airsList):
        if i.type == 0:
            pointList.append(Point(i, parent=index, Coord=i.Fcoord))
            PointN += 1
        else:
            From = i.Fcoord
            To = i.Tcoord
            delta = To - From
            pointNumber = calcDis(From, To) / Distance_per_point
            pointNumber = int(pointNumber)
            delta /= pointNumber
            for j in range(pointNumber):
                Coord = From + delta * j
                pointList.append(Point(i, parent=index, Coord=Coord))
                PointN += 1
            pointList.append(Point(i, parent=index, Coord=To))
            PointN += 1


def mkPic(cList, result):
    plt.scatter(x=cList[:, 0], y=cList[:, 1], c=result, s=10)
    plt.show()


def k_means(left=0, right=17):
    cList = []
    pList = []
    i: Point
    for index, i in enumerate(pointList):
        pointList[index].mkData(left, right)
        pList.append(i.parent)
        cList.append(pointList[index].cData)
    cList = np.asarray(cList, dtype=np.float_)
    result = KMeans(n_clusters=K, max_iter=N).fit_predict(cList)
    # print(result)
    a = []
    for i in range(K):
        a.append([])
    for index, j in enumerate(result):
        a[j].append(np.sum(np.asarray(cList[index][2:])))

    c = []
    for i in a:
        c.append(np.mean(np.asarray(i)))

    tpos = np.argmax(np.asarray(c))
    tnum = c[tpos]
    sum = 0
    for i in c:
        sum += tnum - i;

    slist = []
    dlist = []
    for index, i in enumerate(result):
        if i == tpos:
            slist.append(index)
            dlist.append(pointList[index])

    return sum, slist, dlist


SUM = []
RESULT = []
DATA = []


def main_process():
    index = 0
    for l in range(17):
        for r in range(l, 17):
            tSum, tResult, tDATA = k_means(l, r)
            SUM.append(tSum)
            RESULT.append(tResult)
            DATA.append(tDATA)
            index += 1


def query(op=1, len=1, left=6, right=7):

    ans = 0
    left = left - 6
    right = right - 7
    ansl = left
    ansr = right
    index = 0
    if op == 1:  # [L,R]
        for l in range(17):
            for r in range(l, 17):
                if l == left and r == right:
                    ans = index
                index += 1

    if op == 2:
        tsum = 0
        for l in range(17):
            for r in range(l, 17):
                if r - l + 1 == len:
                    if tsum < SUM[index]:
                        tsum = SUM[index]
                        ans = index
                        ansl = l
                        ansr = r
                index += 1
    return ans,ansl+6,ansr+7

readFile()
mkPoint()
main_process()
ans,l,r = query(op=1,left=15,right=16,len=7)

print('The number of POINTs   is:',str(PointN))
print('The number of CLUSTERs is:',str(K))
print(l,r)
for i in DATA[ans]:
    print(i.name + ' ' + str(i.coord))
