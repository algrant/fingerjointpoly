from fjp import *

offset_multi = 10 # 14.5
polygon_scale = 28
overlap = 2
material_thickness=3
tab_width = 2
border_width = 3

fn = "./dmccooey/TruncatedIcosahedron.txt"
# fn = "./dmccooey/DualGeo_3_0.txt"
# fn = "./dmccooey/Rhombicosidodecahedron.txt"
# fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron2.txt"
# fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron3.txt"
# fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron4.txt"
fn = "./dmccooey/polyhedra/Catalan/RpentagonalHexecontahedron.txt"

class DmCooeyFJP(FingerJointPolyhedron):
  def load_from_file(self, filename):
    with open(filename) as fp:
      state = "TITLE"
      for line in fp:
        if line == "\n":
          continue
        if state == "TITLE":
          self.title = line
          state = "COEFFICIENTS"
          self.coefficients = {}
          self.vertices = []
        elif state == "COEFFICIENTS":
          if line[0] == "C":
            a = line.strip().split(" = ")
            if not a[0].strip() in self.coefficients:
              self.coefficients[a[0].strip()] = { "value":float(a[1]) }
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
                vert.append(mult*self.coefficients[co]["value"])
            self.vertices.append(np.array(vert))
          else:
            state = "FACES"
            self.faces = []
        if state == "FACES" and line[0] == "{":
          self.faces.append([int(f) for f in line[1:-2].split(", ")])


ti = DmCooeyFJP(
  scale=polygon_scale, 
  overlap=overlap, 
  material_thickness=material_thickness, 
  tab_width=tab_width,
  border_width=border_width,
  poly_path="poly/dmcooey"
)

ti.load_from_file(fn)
ti.generate_half_edges()
ti.find_lengths()
ti.find_dihedrals()
ti.determine_offsets()
#ti.get_2d_faces()
# ti.save_3d_orig()
ti.save_3d_fjp()

# handy to double check dihedral against know dihedrals...
# for k, he in ti.half_edges.items():
#   print(k, he["length"]*100, 180*he["dihedral"]/pi)

