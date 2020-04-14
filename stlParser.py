from fjp import *
import stl
offset_multi = 10 # 14.5
polygon_scale = 100
overlap = 2

files = {
  "egg": "./stl/egg_best.stl"
}


class StlPoly(Polyhedron):
  def load_from_file(self, filename):
    poly_mesh = stl.mesh.Mesh.from_file(filename)
    print(poly_mesh.vectors.shape)
    print(dir(poly_mesh))
    """ 
      giving up on this cause
      stl only provides triangle
      data and I'm very interested in
      getting full polygons

      in principle one could connect
      edges on faces and junk, but that
      sounds difficult... and blender
      can output to .obj which is a
      way dumber file format :)
    """



ti = StlPoly()
ti.load_from_file(files["egg"])
