from django.shortcuts import render
from django.http import JsonResponse, FileResponse
import json
import numpy as np
import io
from numpy import unicode

# Create your views here.


#class NumpyAwareJSONEncoder(json.JSONEncoder):
#    """
#        This class facilitates the JSON encoding of Numpy onjects.
#        e.g. numpy arrays are not supported by the standard json encoder - dumps.
#        """
#        def default(self, obj):
#            if isinstance(obj, np.ndarray):
#                return obj.tolist()
#            return json.JSONEncoder.default(self, obj)
#


def index(request):
    pass


async def postState(request):
    # This is where the training logic is going to go


        

    pass


async def getAgent(request):

    # get the agent from the file
   
    model = open('smashBot/recentWeights.hdf5', 'rb')

    response = FileResponse(model)

    response['Access-Control-Allow-Origin'] = '*'

    return response
    
    #model = {}
    #response = JsonResponse({'model': model})



