import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import global_var as gl

RList = [150, 105, 60]
FileName = 'pic3.png'
is_FULL_NEEDED = 0

areaWide = 500  #-westeast
areaHeight = 460 #north-south

MAXDISTANCE = 1000




IM = np.zeros(0)

disB = 13

ALL = 0

def ReadFile():
    global IM
    global disB
    global ALL
    IM = mpimg.imread(FileName)
    im = []
    for i in IM:
        t = []
        for j in i:
            if np.sum(j) == 4:
                t.append(0)
            else:
                t.append(1)
        im.append(t)
    IM = im
    IM = np.asarray(IM)
    for i in IM:
        for j in i:
            ALL += j
    ALL *= len(RList)
    x,y = IM.shape
    t1 = float(areaWide/y)
    t2 = float(areaHeight/x)
    disB = (t1+t2)/2
    print(y,x,disB)
    print('The full area = ',ALL)

def calc_dis(Veca, Vecb):
    return np.linalg.norm(Veca - Vecb) * disB


def limit(x, y):
    s = IM.shape
    xl = s[0]
    yl = s[1]
    if x < 0 or x >= xl: return False
    if y < 0 or y >= yl: return False
    return True


def updateIM(xList: list, yList: list, rList: list):
    global IM
    im = []
    for i in range(3):
        im.append(np.zeros(IM.shape))


    for index, x in enumerate(xList):
        y = yList[index]
        for tr in rList[index]:
            r = RList[tr]

            if not IM[x][y] == 1:
                continue
            delr = int(r / disB)
            for ty in range(y - delr, y + delr):
                for tx in range(x - delr, x + delr):
                    c1 = np.asarray([tx, ty], dtype=np.float_)
                    c2 = np.asarray([x, y], dtype=np.float_)
                    if not limit(tx, ty):
                        continue
                    if calc_dis(c1, c2) <= r and IM[tx][ty] == 1:
                        im[tr][tx][ty] = 1
    return im


def setCover(xList,yList):
    ansList = []


    for r in RList:
        stations = {}
        needed = set()
        finial = set()
        for index,x in enumerate(xList):
            y = yList[index]
            connect = []
            delr = int(r / disB)
            for ty in range(y - delr, y + delr):
                for tx in range(x - delr, x + delr):
                    c1 = np.asarray([tx, ty], dtype=np.float_)
                    c2 = np.asarray([x, y], dtype=np.float_)
                    if not limit(tx, ty):
                        continue
                    if calc_dis(c1, c2) <= r and IM[tx][ty] == 1:
                        connect.append((tx,ty))
                        needed.add((tx,ty))
            stations[(x,y)] = set(connect)


        while needed:
            best = None
            covered = set()
            station_covered = set()
            for station,states in stations.items():
                covered = needed & states
                if len(covered) > len(station_covered):
                    best = station
                    station_covered = covered
            needed -= station_covered
            finial.add(best)

        ansList.append(finial)

    return ansList



def calc_area(xList: list, yList: list,rList = []):
    if len(rList) < 1:
        for i in xList:
            rList.append([0,1,2])
    im = updateIM(xList, yList, rList)
    sum = 0
    for k in range(3):
        for i in im[k]:
            for j in i:
                sum += j

    return sum


def mkPic(xList, yList, im):
    tm = np.asarray(IM, dtype=np.float_)
    for ix, x in enumerate(tm):
        for iy, y in enumerate(x):
            sum = 0
            for k in range(3):
                sum += im[k][ix][iy] * 0.2
            tm[ix][iy] -= sum

    for index, x in enumerate(xList):
        y = yList[index]
        tm[x][y] = 1

    plt.imshow(tm, cmap=plt.cm.gray)
    plt.show()


def finial(xList,yList):

    result = setCover(xList,yList)
    rList = []
    for i in xList:
        rList.append([])

    for index,x in enumerate(xList):
        y = yList[index]
        for j in range(3):
            tset = result[j]
            if (x,y) in tset:
                rList[index].append(j)


    tim = updateIM(xList,yList,rList)

    print('Area = ' + str(calc_area(xList,yList,rList)))

    k = gl.Max_Point_Num

    tx = []
    ty = []
    tr = []
    for index in range(len(xList)):
        x = xList[index]
        y = yList[index]
        r = rList[index]
        if r == []:
            continue
        else:
            tx.append(x)
            ty.append(y)
            tr.append(r)
        print(x,y,r)

    xList = tx[:k]
    yList = ty[:k]
    rList = tr[:k]

    ans = calc_area(xList,yList,rList)

    print('The soultion is ',ans,len(xList))

    if not is_FULL_NEEDED:
        return True


    if ans < ALL:
        print('False,retry')

    mkPic(xList,yList,tim)

    if ans < ALL:
        return False

    return True

if __name__ == '__main__':
    ReadFile()
