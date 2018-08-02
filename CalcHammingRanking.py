import numpy as np

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

def CalcReAcc(qB, rB,topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{mxq}
    # for test set
    num_query = qB.shape[0]
    topkacc = 0
    count = 0
    for index in range(num_query):
        hamm = CalcHammingDist(qB[index, :], rB)
        ind = np.argsort(hamm)
        tind = ind[0:topk]
        index_s = index
        count += index_s in tind
    topkacc = count / (num_query*1.0)

    return topkacc

def CalcDist(B1, B2):
    num = B2.shape[0]
    dis = []
    for i in range(num):
        dis_ = np.sum((B1 - B2[i,:])**2)
        dis.append(dis_)
    dis = np.array(dis)
    return dis

def CalcReAcc_2(qB, rB,topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{mxq}
    # for test set
    num_query = qB.shape[0]
    topkacc = 0
    count = 0
    for index in range(num_query):
        hamm = CalcDist(qB[index, :], rB)
        ind = np.argsort(hamm)
        tind = ind[0:topk]
        index_s = index
        count += index_s in tind
    topkacc = count / (num_query*1.0)

    return topkacc



def CalcTopAcc(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkacc = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        
        gnd = gnd[ind]
        
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        topkacc += tsum / topk
    topkacc = topkacc / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkacc


    

if __name__=='__main__':
    qB = np.load('64TU-T_S.npy')
    rB = np.load('64TU-T_I.npy')
    acc =CalcReAcc(qB, rB,10)
    print(acc)

