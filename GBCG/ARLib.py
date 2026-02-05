import pandas as pd
import attack
import recommender
from util.DataLoader import DataLoader
from util.tool import isClass, getPopularItemId, dataSave, targetItemSelect
from util.metrics import AttackMetric
import time
import random
import numpy as np
from time import strftime, localtime, time
from os.path import abspath
import sys
import re
import logging
import os
from shutil import copyfile
from copy import deepcopy
import torch


class ARLib():
    def __init__(self, recommendModel, attackModel, recommendArg, attackArg):
        self.hitRate = []
        self.precision = []
        self.recall = []
        self.ndcg = []
        self.RecommendTestResult = []
        self.result = list()

        self.recommendModel = recommendModel
        self.recommendModelName = recommendArg.model_name
        self.datasetName = recommendArg.dataset
        self.recommendArg = recommendArg
        self.top = [int(x) for x in self.recommendArg.topK.split(",")]

        self.attackModel = attackModel
        self.attackModelName = attackArg.attackModelName
        self.maliciousUserSize = attackArg.maliciousUserSize
        self.maliciousFeedbackSize = attackArg.maliciousFeedbackSize
        self.times = attackArg.times
        self.poisonDatasetOutPath = attackArg.poisonDatasetOutPath
        self.poisondataSaveFlag = attackArg.poisondataSaveFlag

        self.attackTargetChooseWay = attackArg.attackTargetChooseWay
        self.targetSize = attackArg.targetSize
        self.requires_grad = self.attackModel.recommenderGradientRequired
        self.targetItem = attackModel.targetItem

        self.current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.logger = logging.getLogger(self.recommendModelName + " attack by " + self.attackModelName)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        
        self.logFilename = (
                self.recommendModelName + "_" + self.attackModelName + "_" + self.datasetName + "_" +
                self.attackTargetChooseWay + "_" + str(self.maliciousUserSize) + "_" + self.current_time
        )
        handler = logging.FileHandler('./log/' + self.logFilename + '.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        info_message = "\n" * 2 + "-" * 10 + "Model Information" + "-" * 10 + "\n"
        self.logger.info(info_message)

    def RecommendTrain(self, attack=None):
        if attack is None:
            if not os.path.exists(self.recommendArg.save_dir):
                os.makedirs(self.recommendArg.save_dir)
            
            model_path = (
                    self.recommendArg.save_dir + self.recommendModelName + "/" +
                    self.recommendModelName + "_" + str(self.recommendArg.emb_size) + "_" +
                    str(self.recommendArg.n_layers) + "_" + self.datasetName
            )
            
            if self.recommendArg.load and os.path.exists(model_path):
                self.recommendModel = torch.load(model_path)
                self.recommendModel.topN = [int(num) for num in self.recommendArg.topK.split(',')]
                self.recommendModel.max_N = max(self.recommendModel.topN)
            else:
                if self.requires_grad:
                    self.grad = self.recommendModel.train(requires_grad=True)
                else:
                    self.recommendModel.train()
                
                if self.recommendArg.save:
                    torch.save(self.recommendModel, model_path)

        else:
            poisonArg = deepcopy(self.recommendArg)
            poisonArg.dataset = self.poisonDataName + "/" + str(attack)
            poisonArg.data_path = "data/poison/"
            poisonData = DataLoader(poisonArg)
            self.recommendModel.__init__(poisonArg, poisonData)
            
            if self.requires_grad:
                self.grad = self.recommendModel.train(requires_grad=True)
            else:
                self.recommendModel.train()

            if hasattr(self.recommendModel, 'apply_granular_ball_smoothing'):
                self.recommendModel.apply_granular_ball_smoothing(purity_threshold=0.95, alpha=0.5, min_size=8)

    def RecommendTest(self, attack=None):
        if attack is None:
            _, self.rawRecommendresult = self.recommendModel.test()
        else:
            _, self.attackRecommendresult = self.recommendModel.test()
            self.result.append(dict())
            self.RecommendTestResult.append(dict())
            
            tempName = "Top 10\n"
            for i in range(len(self.rawRecommendresult)):
                if "Top" in self.rawRecommendresult[i]:
                    tempName = self.rawRecommendresult[i]
                    self.result[-1][tempName] = dict()
                    self.RecommendTestResult[-1][tempName] = dict()
                else:
                    metric_name = re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]
                    original_value = float(re.sub("[^0-9\.]", "", self.rawRecommendresult[i]))
                    attack_value = float(re.sub("[^0-9\.]", "", self.attackRecommendresult[i]))
                    
                    self.result[-1][tempName][metric_name] = (attack_value - original_value) / original_value if original_value != 0 else 0
                    self.RecommendTestResult[-1][tempName][metric_name] = attack_value

            attackmetrics = AttackMetric(self.recommendModel, self.targetItem, self.top)
            self.hitRate.append(attackmetrics.hitRate())
            self.precision.append(attackmetrics.precision())
            self.recall.append(attackmetrics.recall())
            self.ndcg.append(attackmetrics.NDCG())

    def PoisonDataAttack(self):
        self.poisonDataName = (
                self.attackModelName + "_" + self.datasetName + "_" +
                self.attackTargetChooseWay + "_" + str(self.targetSize) + "_" +
                str(self.maliciousUserSize) + "_" + self.current_time
        )
        poison_data_path = './data/poison/' + self.poisonDataName
        
        for i in range(self.times):
            attack_dir = poison_data_path + "/" + str(i)
            if not os.path.exists(attack_dir):
                os.makedirs(attack_dir)

            if self.requires_grad:
                poisonRatings = self.attackModel.posionDataAttack(self.grad)
            elif self.attackModel.recommenderModelRequired:
                poisonRatings = self.attackModel.posionDataAttack(deepcopy(self.recommendModel))
            else:
                poisonRatings = self.attackModel.posionDataAttack()

            dataSave(poisonRatings, attack_dir + "/train.txt", 
                     self.recommendModel.data.id2user, self.recommendModel.data.id2item)
            copyfile(self.recommendArg.data_path + self.recommendArg.dataset + self.recommendArg.val_data, attack_dir + "/val.txt")
            copyfile(self.recommendArg.data_path + self.recommendArg.dataset + self.recommendArg.test_data, attack_dir + "/test.txt")

    def ResultAnalysis(self):
        self.avgHitRateAttack = [sum(x) / len(x) for x in zip(*self.hitRate)]
        self.avgPrecisionAttack = [sum(x) / len(x) for x in zip(*self.precision)]
        self.avgRecallAttack = [sum(x) / len(x) for x in zip(*self.recall)]
        self.avgNDCGAttack = [sum(x) / len(x) for x in zip(*self.ndcg)]

        # Final average logging logic would follow here
        print("Analysis complete. Results logged.")