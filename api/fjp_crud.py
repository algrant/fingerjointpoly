from bottle import request, response
from bottle import get

import os
import fnmatch
import json

dmcooeyModels = []

i = 0

for root, dirnames, filenames in os.walk('./dmccooey/polyhedra'):
  for filename in fnmatch.filter(filenames, '**.txt'):
    path = os.path.join(root, filename)
    name = filename[:-4]
    
    dmcooeyModels.append({
          "id": i,
          "name": name,
          "type": "dmcooey",
          "path": path
        })
    i += 1

@get('/models')
def listing_handler():
    '''Handles name listing'''

    response.headers['Content-Type'] = 'application/json'
    response.headers['Cache-Control'] = 'no-cache'
    return json.dumps({
      'models': dmcooeyModels
    })

@get('/models/<model_id:int>')
def individual_handler(model_id):
    '''Handles name listing'''

    response.headers['Content-Type'] = 'application/json'
    response.headers['Cache-Control'] = 'no-cache'

    for m in dmcooeyModels:
      if m["id"] == model_id:
        return json.dumps({
          'model': m
        })
    return json.dumps({
      "model": "not found"
    })
