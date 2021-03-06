# K-means support functions
from numpy import *

def distEclud(vecA, vecB):
    # calculate the euclidean distance between two vectors
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    # input a data set and get the first k initial points
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while(clusterChanged):
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if (distJI < minDist):
                    minDist = distJI
                    minIndex = j
            if (clusterAssment[i, 0] != minIndex):
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m ,2)))
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = \
                dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = \
                kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = \
                sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit < lowestSSE):
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = \
            len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = \
            bestCentToSplit
        print("The bestCentToSplit is ", bestCentToSplit)
        print("The len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    centList = [squeeze(i).tolist()[0] for i in centList]
    return mat(centList), clusterAssment
# centList ??????????????????????????????
# clusterAssment????????????????????? ????????????cluster??? ???cluster?????????????????????