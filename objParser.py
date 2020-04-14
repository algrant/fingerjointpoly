from fjp import *
import pywavefront
import numpy as np

offset_multi = 5 # 14.5
polygon_scale = 70
overlap = 2

files = {
  "egg": "./obj/egg5.obj"
}


class StlPoly(Polyhedron):
  def load_from_file_obj(self, filename):
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    # print(scene.vectors.shape)
    # scene.mesh_list[0].materials[0].vertices
    # scene.mesh_list[0].faces
    terrible_vertices = scene.mesh_list[0].materials[0].vertices
    count = len(terrible_vertices)/6
    self.vertices = [ np.array([terrible_vertices[i*6+3], terrible_vertices[i*6+4], terrible_vertices[i*6+5]]) for i in range(int(count))]
    # print(self.vertices)
    self.faces = scene.mesh_list[0].faces
    print(self.faces)
  def load_from_file(self, filename):
    self.vertices = []
    self.faces = []
    with open(filename) as fp:
      for line in fp:
        spl = line.split(" ")

        if spl[0] == "v":
          self.vertices.append(np.array([float(d.strip()) for d in spl[1:]]))
        elif spl[0] == "f":
          self.faces.append([int(d.strip().split("/")[0]) - 1 for d in spl[1:]])
          self.faces[-1].reverse()

ti = StlPoly()
ti.load_from_file(files["egg"])
ti.generate_half_edges()
ti.find_lengths()
ti.find_dihedrals()
ti.get_2d_faces(polygon_scale, offset_multi, overlap)