from fjp import *
# import pywavefront
import numpy as np

offset_multi = 5 # 14.5
polygon_scale = 63
overlap = 1
tab_width = 3
material_thickness = 3
border_width = 3.5

files = {
  "egg": "./obj/egg5.obj",
  "donut": "./obj/donut.obj"
}


class ObjPoly(Polyhedron):
  def load_from_file(self, filename):
    self.vertices = []
    self.faces = []
    # dumbest .obj reader I can imagine! only care about v & f...
    with open(filename) as fp:
      for line in fp:
        spl = line.split(" ")
        if spl[0] == "v":
          self.vertices.append(np.array([float(d.strip()) for d in spl[1:]]))
        elif spl[0] == "f":
          self.faces.append([int(d.strip().split("/")[0]) - 1 for d in spl[1:]])
          self.faces[-1].reverse()

ti = ObjPoly(
  scale=polygon_scale, 
  overlap=overlap, 
  material_thickness=material_thickness, 
  tab_width=tab_width,
  border_width=border_width
)

ti.load_from_file(files["donut"])
ti.generate_half_edges()
ti.find_lengths()
ti.find_dihedrals()
ti.determine_offsets()
ti.get_2d_faces()
ti.print_face_connectivity()
