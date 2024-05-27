'''
Author: Yi Xu <xu.yi@northeastern.edu>
Model selector
'''

from models.unitraj import UniTraj

def select_model(name):
    name = name.lower()
    if name == 'unitraj':
        return UniTraj