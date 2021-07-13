from bottle import get, request, response

import os
import fnmatch
import json
import sys

sys.path.insert(0,'../..')

from fjp.dmccooey_parser import load_from_file
from fjp.polyhedron import Polyhedron

dmcooeyModels = []

i = 0

for root, dirnames, filenames in os.walk('../../data/dmccooey/polyhedra'):
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

    model = "not found"

    for m in dmcooeyModels:
      if m["id"] == model_id:
        model = m
        data = load_from_file(m["path"])
        poly = Polyhedron(data["vertices"], data["faces"])
        model["data"] = {
          "vertices": [list(v) for v in poly.vertices],
          "faces": [list(f) for f in poly.faces],
          "triangles": [list(v) for v in poly.save_3d_fjp()]
        }

        break
      
    return json.dumps({
      "model": model
    })
