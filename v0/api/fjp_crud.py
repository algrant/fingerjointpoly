from bottle import get, request, response

import os
import fnmatch
import json
import sys

sys.path.insert(0,'..')

from dmcooeyParser import *

dmcooeyModels = []

i = 0

for root, dirnames, filenames in os.walk('../dmccooey/polyhedra'):
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
        polygon_scale = 28
        overlap = 2
        material_thickness=3
        tab_width = 2
        border_width = 3

        ti = DmCooeyFJP(
          scale=polygon_scale, 
          overlap=overlap, 
          material_thickness=material_thickness, 
          tab_width=tab_width,
          border_width=border_width,
          poly_path="poly/dmcooeyAPI"
        )

        ti.load_from_file(m["path"])

        model["data"] = {
          "vertices": [list(v) for v in ti.vertices],
          "faces": ti.faces
        }

        break
      
    return json.dumps({
      "model": model
    })
