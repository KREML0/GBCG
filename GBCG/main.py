import time
from conf.attack_parser import attack_parse_args
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from ARLib import ARLib
import os
import torch
import numpy as np
import random
import winsound
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

if __name__ == '__main__':

    recommend_args = recommend_parse_args()
    attack_args = attack_parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
    seed = recommend_args.seed
    seedSet(seed)

    import_recommend = 'from recommender.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_recommend)
    
    import_attack = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_attack)

    data = DataLoader(recommend_args)

    recommend_model = eval(recommend_args.model_name)(recommend_args, data)

    if "GBCG" in import_attack:
        LightGCN_model_path = 'modelsaved/LightGCN/LightGCN_64_2_ml-100k'
        LightGCN_model = torch.load(LightGCN_model_path)
        attack_model = eval(attack_args.attackModelName)(LightGCN_model, attack_args, data)
    else:
        attack_model = eval(attack_args.attackModelName)(attack_args, data)

    arlib = ARLib(recommend_model, attack_model, recommend_args, attack_args)

    start_time = time.time()

    arlib.RecommendTrain()
    arlib.RecommendTest()

    arlib.PoisonDataAttack()
    for step in range(arlib.times):
        arlib.RecommendTrain(attack=step)
        arlib.RecommendTest(attack=step)

    arlib.ResultAnalysis()

    end_time = time.time()
    print("Running time: %f s" % (end_time - start_time))
    winsound.Beep(800, 1000)