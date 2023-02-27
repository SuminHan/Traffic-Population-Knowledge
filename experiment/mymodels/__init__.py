
import sys
from submodules import *
from mymodels.BASE import *
from mymodels.MATURE import *
from mymodels.DCGRU import * # MyDCGRUSTE0ZCF, MyDCGRUSTE0ZC, MyDCGRUSTE0ZF
from mymodels.GMAN import * # MyGM0ZCF, MyGM0ZC, MyGM0ZF
from mymodels.ASTGCN import *
from mymodels.OURS import *
from mymodels.OURS_CRAZY import *
from mymodels.OURS_AE import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def ModelSet(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    return (model(kwargs) ) * extdata['maxval']

def ModelSetOurs(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    return model(kwargs)

def ModelSetSTD(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    # return (model(kwargs) ) * extdata['maxval']
    return (model(kwargs) ) * extdata['stdval'] + extdata['meanval']
