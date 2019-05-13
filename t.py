from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
ClusterNum = 5

Distance_per_point = 1.0
# set a point per airline

future_num = 3
# the number of futures' number

importanceOfDistance = 2
# the importance of distance between two points


colorList = []

class Airplane:
    name = "null"
    # the name of this point
    type = 0
    # the type  1 for airlines  2 for airport

    belong = np.zeros(ClusterNum, dtype=np.int_, order='C')
    # the number belongs to every cluster
    # would be 00001000 e.t.c for point
    # would be 10204005 e.t.c for airlines

    cluster = 1
    # belong to which Cluster

    Fcoord = np.zeros(2, dtype=np.float_, order='C')
    Tcoord = np.zeros(2, dtype=np.float_, order='C')
    # Coord from && to

    data = np.zeros(17, dtype=np.float_, order='C')

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
        self.coord = Coord
        self.parent = parent
        self.type = air.type
        self.color = colorList[parent]

    def display(self):
        print(self.name, self.coord, self.belongsTo, self.data, self.color)

    def mkData(self,rank):
        self.cData = np.zeros(3,dtype=np.float_)
        self.cData[:2] = self.coord * importanceOfDistance
        self.cData[2:] = self.data[rank]

airsList = []  # type: list[Airplane]
pointList = [] # type: list[point]
PointN = 0


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


# read data from file
def readFile():
    file = open('t.txt')
    FileContex = file.readlines()
    for i in FileContex:
        tlist = i.split(' ')
        airsList.append(Airplane(tlist))
    for i in range(20):
        colorList.append(randomcolor())

# calc Euclidean Distance for vector
def calcDis(Veca, Vecb):
    return np.linalg.norm(Veca - Vecb)


def mkPoint():
    global PointN
    global pointList
    global airsList
    for index, i in enumerate(airsList):
        if i.type == 0:
            pointList.append(Point(i, parent=index,Coord=i.Fcoord))
            PointN +=1
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


def drawOrigin(flag:bool):
    if(flag == False):return
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    xlist = []
    ylist = []
    clist = []
    slist = []
    i:Point
    for i in pointList:
        xlist.append(i.coord[0])
        ylist.append(i.coord[1])
        clist.append(i.color)
        if(i.type == 1):
            slist.append(1)
        else:
            slist.append(15)
    #print(xlist,'\n',ylist,'\n',clist)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xlist,ylist,c=clist,s=slist,marker='o')
    plt.show()


def k_means(K = ClusterNum,iterNum = 10,ShowPic = True,UpdateCluster = False):#The number of Cluster,The number of iter calc times
    dataList = []
    i: Point
    for i in pointList:
        i.mkData(0)
        dataList.append(i.cData)
    dataList = np.array(dataList,dtype=np.float_)
    colorL = KMeans(n_clusters=K,max_iter=iterNum).fit_predict(dataList)
    #draw pic

    if UpdateCluster == True:

        newList = np.zeros([16,K],dtype=np.int_)
        for index,i in enumerate(pointList):
            if i.type == 1:
                newList[i.parent][colorL[index]] += 1

        tdic = np.zeros(16,dtype=np.int_)

        for index,i in enumerate(airsList):
            if i.type == 1:
                tdic[index] = np.argmax(newList[index])

        for index,i in enumerate(pointList):
            if i.type == 1:
                colorL[index] = tdic[i.parent]

    if ShowPic == True:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(dataList[:,0],dataList[:,1],dataList[:,2],c=colorL)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('Data')
        plt.show()


readFile()
mkPoint()
#disPlayList(pointList)
drawOrigin(False)
k_means(K=4,ShowPic=True,UpdateCluster=True)
#plt.show()
