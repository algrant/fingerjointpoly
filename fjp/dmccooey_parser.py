from collections import namedtuple  
import numpy as np
from polyhedron import Polyhedron

def load_from_file(filename):
  polyhedron_info = namedtuple('PolyhedronData', ['title', 'coefficients', 'vertices', 'faces'])
  with open(filename) as fp:
    state = "TITLE"
    for line in fp:
      if line == "\n":
        continue
      if state == "TITLE":
        polyhedron_info.title = line
        state = "COEFFICIENTS"
        polyhedron_info.coefficients = {}
        polyhedron_info.vertices = []
      elif state == "COEFFICIENTS":
        if line[0] == "C":
          a = line.strip().split(" = ")
          if not a[0].strip() in polyhedron_info.coefficients:
            polyhedron_info.coefficients[a[0].strip()] = { "value":float(a[1]) }
        elif line[0] == "V":
          a = line.strip().split(" = ")
          coords = a[1][1:-1].split(", ")
          vert = []
          for c in coords:
            co = c.strip()
            try: 
              vert.append(float(co))
            except:
              mult = 1
              if co[0] == "-":
                co = co[1:]
                mult = -1
              vert.append(mult*polyhedron_info.coefficients[co]["value"])
          polyhedron_info.vertices.append(np.array(vert))
        else:
          state = "FACES"
          polyhedron_info.faces = []
      if state == "FACES" and line[0] == "{":
        polyhedron_info.faces.append([int(f) for f in line[1:-2].split(", ")])

    return {
      "faces": np.array(polyhedron_info.faces),
      "vertices": np.array(polyhedron_info.vertices)
    }

if __name__ == "__main__":
  # "./data/dmccooey/polyhedra/Platonic/Cube.txt"
  info = load_from_file("./data/dmccooey/polyhedra/Platonic/Cube.txt")
  poly = Polyhedron(info["vertices"], info["faces"])

  print(poly.as_triangles())