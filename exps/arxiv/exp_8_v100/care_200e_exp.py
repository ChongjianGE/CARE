# encoding: utf-8
import os
from exps.arxiv.care_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.max_epoch = 200
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
