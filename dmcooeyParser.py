import numpy as np
import svgwrite
from math import pi, cos, tan


offset_multi = 5
polygon_scale = 55
# fn = "./dmccooey/TruncatedIcosahedron.txt"
# fn = "./dmccooey/DualGeo_3_0.txt"
# fn = "./dmccooey/Rhombicosidodecahedron.txt"
fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron2.txt"
def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return np.array([float(m.T[0]), float(m.T[1])])

def svg_poly(face, filename, dihedrals, min_length):
  lines = []
  min_x = 1000000000
  min_y = 1000000000
  max_x = -100000000
  max_y = -100000000
  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]
    length = np.linalg.norm(curr-prev)
    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= np.linalg.norm(v)
    a_o = v*offset_multi

    norm = rotate_via_numpy(v, pi/2)
    norm /= np.linalg.norm(norm)
    l = 3
    bump = 0
    outer = norm*(l/tan(dihedrals[i]) + bump)
    inner = norm*(l/cos(dihedrals[i]) - bump)

    lines.append([[prev, curr], svgwrite.rgb(10, 10, 16, '%')])
    out_start, out_end = prev + outer + a_o, curr + outer - a_o
    in_start, in_end = prev - inner + a_o, curr - inner - a_o

    # start = prev
    # out_start
    # out_start*(1-t)+out_end(t)
    # in_start*(1-t)+in_end(t)
    # in_start*(1-t)+in_end(t)
    squiggles = [prev, prev + a_o]
    divs = 2
    for i in range(divs):
      t_0 = i/divs
      t_2 = (i+1)/divs
      t_1 = (t_0 + t_2)/2
      squiggles.append(out_start*(1-t_0) + out_end*t_0)
      squiggles.append(out_start*(1-t_1) + out_end*t_1)
      squiggles.append(in_start*(1-t_1) + in_end*t_1)
      squiggles.append(in_start*(1-t_2) + in_end*t_2)

    squiggles.append(curr - a_o)
    squiggles.append( curr)

    lines.append([squiggles, svgwrite.rgb(100, 10, 16, '%')])
    min_x = min(prev[0], curr[0],(prev+outer)[0], (curr+outer)[0],(prev-inner)[0], (curr-inner)[0], min_x)
    min_y = min(prev[1], curr[1],(prev+outer)[1], (curr+outer)[1],(prev-inner)[1], (curr-inner)[1], min_y)
    max_x = max(prev[0], curr[0],(prev+outer)[0], (curr+outer)[0],(prev-inner)[0], (curr-inner)[0], max_x)
    max_y = max(prev[1], curr[1],(prev+outer)[1], (curr+outer)[1],(prev-inner)[1], (curr-inner)[1], max_y)
    # dwg.add(dwg.line(prev+outer, curr+outer, stroke=svgwrite.rgb(100, 10, 16, '%')))
    # dwg.add(dwg.line(prev-inner, curr-inner, stroke=svgwrite.rgb(100, 10, 16, '%')))
  minp = np.array([min_x, min_y])
  print("%imm"%max_x, "%imm"%max_y)

  size = ("%imm"%(max_x - min_x),"%imm"%(max_y- min_y))
  # size = (max_x, max_y)
  dwg = svgwrite.Drawing(filename, size=size, profile="tiny")
  pixies_per_mm = 3.543307
  for line in lines:
    dwg.add(dwg.polyline([(point - minp)*pixies_per_mm for point in line[0]], stroke=line[1], fill="white"))
  dwg.save()

""" https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python """
def new_dihedral(p0, p1, p2, p3):
  """Praxeolitic formula
  1 sqrt, 1 cross product"""
  b0 = -1.0*(p1 - p0)
  b1 = p2 - p1
  b2 = p3 - p2

  # normalize b1 so that it does not influence magnitude of vector
  # rejections that come next
  b1 /= np.linalg.norm(b1)

  # vector rejections
  # v = projection of b0 onto plane perpendicular to b1
  #   = b0 minus component that aligns with b1
  # w = projection of b2 onto plane perpendicular to b1
  #   = b2 minus component that aligns with b1
  v = b0 - np.dot(b0, b1)*b1
  w = b2 - np.dot(b2, b1)*b1

  # angle between v and w in a plane is the torsion angle
  # v and w may not be normalized but that's fine since tan is y/x
  x = np.dot(v, w)
  y = np.dot(np.cross(b1, v), w)
  return np.arctan2(y, x)

# def get_normal(p0, p1, p2):



class Polyhedron:
  def __init__(self, vertices=[], faces=[]):
    self.vertices = vertices
    self.faces = faces
    self.half_edges = {}
    self.faces_2d = []

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
          face_len = len(self.faces[-1])
          for i in range(face_len):
            pre_m1 = self.faces[-1][i-2]
            prev = self.faces[-1][i-1]
            curr = self.faces[-1][i]
            curr_1 = self.faces[-1][(i+1)%face_len]
            self.half_edges["%i_%i"%(prev, curr)] = { "id": "%i_%i"%(prev, curr), "s": prev, "e": curr, "opp": "%i_%i"%(curr, prev), "prev":"%i_%i"%(pre_m1, prev),  "next":"%i_%i"%(curr, curr_1), "face_id": len(self.faces) }

  def find_lengths(self):
    self.min_length = 1000000000
    self.max_length = 0
    for face in self.faces:
      for i in range(len(face)):
        prev = self.vertices[face[i-1]]
        curr = self.vertices[face[i]]
        length = np.linalg.norm(curr-prev)
        self.half_edges["%i_%i"%(face[i-1], face[i])]["length"] = length
        self.min_length = min(length, self.min_length)
        self.max_length = max(length, self.max_length)
  def find_dihedrals(self):
    for key, he in self.half_edges.items():
      # print(key, he)
      if "dihedral" not in he:
        opp_he = self.half_edges[he["opp"]]
        p0_id = self.half_edges[opp_he["next"]]["e"]
        p1_id = he["s"]
        p2_id = he["e"]
        p3_id = self.half_edges[he["next"]]["e"]
        # print(he["id"], opp_he["id"])
        p0 = self.vertices[p0_id]
        p1 = self.vertices[p1_id]
        p2 = self.vertices[p2_id]
        p3 = self.vertices[p3_id]
        he["dihedral"] = new_dihedral(p0, p1, p2, p3)
        opp_he["dihedral"] = he["dihedral"]

  def get_2d_face(self, index):
    # polygon in 3D
    face_vert_ids = self.faces[index]

    face = [self.vertices[f] for f in face_vert_ids]

    dihedrals = []
    for i in range(len(face_vert_ids)):
      prev = face_vert_ids[i-1]
      curr = face_vert_ids[i]
      dihedrals.append(self.half_edges["%i_%i"%(prev,curr)]["dihedral"])    

    p0, p1, p2 = face[0], face[1], face[2]
    a = p0 - p1
    b = p2 - p1
    norm = np.cross(a, b)
    norm /= np.linalg.norm(norm)
    u = a/np.linalg.norm(a)
    v = np.cross(u, norm)
    v /= np.linalg.norm(v)
    # print(u, v)
    scale = polygon_scale
    new_face = [np.array([scale + np.dot(u, f)*scale, scale + np.dot(v, f)*scale]) for f in face]
    self.min_length
    svg_poly(new_face, "./poly/num%i.svg"%index, dihedrals, self.min_length)

  def get_2d_faces(self):
    for i in range(len(self.faces)):
      self.get_2d_face(i)


ti = Polyhedron()
ti.load_from_file(fn)
ti.find_lengths()
ti.find_dihedrals()
ti.get_2d_faces()

# handy to double check dihedral against know dihedrals...
for k, he in ti.half_edges.items():
  print(k, he["length"]*100, 180*he["dihedral"]/pi)
# print(ti.half_edges)
# a = "0.809016994374947424102293417183"
# print(np.float128(a))

# b = "08090169943749474.24102293417183"
# print(float(b))
