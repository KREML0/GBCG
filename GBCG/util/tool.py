import os
import random
import numpy as np
import torch

def isClass(obj, classList):
    for i in classList:
        if isinstance(obj, i):
            return True
    return False

def dataSave(ratings, fileName, id2user, id2item):
    ratingList = []
    ind = ratings.nonzero()
    for i,j in zip(ind[0].tolist(),ind[1].tolist()):
            ratingList.append((i,j,ratings[i,j]))

    text = []
    for i in ratingList:
        if i[0] in id2user.keys():
            userId = id2user[i[0]]
        else:
            userId = "fakeUser" + str(i[0])
        itemId = id2item[i[1]]
        new_line = '{} {} {}'.format(userId, itemId, i[2]) + '\n'
        text.append(new_line)
    with open(fileName, 'w') as f:
        f.writelines(text)

def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False