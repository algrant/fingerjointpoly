import numpy as np
import svgwrite
from math import pi, cos, tan, sin

offset_multi = 10 # 14.5
polygon_scale = 100
overlap = 2

# fn = "./dmccooey/TruncatedIcosahedron.txt"
# fn = "./dmccooey/DualGeo_3_0.txt"
# fn = "./dmccooey/Rhombicosidodecahedron.txt"
fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron2.txt"
# fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron3.txt"
# fn = "./dmccooey/polyhedra/DualGeodesicIcosahedra/DualGeodesicIcosahedron4.txt"
# fn = "./dmccooey/polyhedra/Catalan/RpentagonalHexecontahedron.txt"

def spline(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False

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
  squaggles = []
  outline = []
  innerLine = []
  for i in range(len(face)):
    prev_minus_one = face[i-2]
    prev = face[i-1]
    curr = face[i]
    length = np.linalg.norm(curr-prev)
    line_1, line_2 = spline(prev_minus_one, prev), spline(prev, curr)
    p_inter = intersection(line_1, line_2)
    innerLine.append(p_inter)
    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= np.linalg.norm(v)
    a_o = v*offset_multi

    norm = rotate_via_numpy(v, pi/2)
    norm /= np.linalg.norm(norm)
    l = 3
    tab_diff = 0.03
    # tab_offset = v*tab_diff/2.0
    etch = norm*(l/tan(dihedrals[i]))
    outer = norm*(l/tan(dihedrals[i]) - overlap)
    inner = norm*(l/sin(dihedrals[i]))

    # lines.append([[prev, curr], svgwrite.rgb(10, 10, 16, '%')])
    outline.append(prev)
    out_start, out_end = prev - outer + a_o, curr - outer - a_o
    in_start, in_end = prev - inner + a_o, curr - inner - a_o
    etch_start, etch_end = prev - etch + a_o, curr - etch - a_o

    lines.append([[etch_start, etch_end], svgwrite.rgb(0, 0, 100, '%')])

    squiggles = [in_start]
    divs = 2
    length = out_start - out_end
    print(np.linalg.norm(norm))
    for i in range(divs):
      t_0 = i/divs
      t_2 = (i+1)/divs
      t_1 = (t_0 + t_2)/2+tab_diff/2.0
      squiggles.append(out_start*(1-t_0) + out_end*t_0)
      squiggles.append(out_start*(1-t_1) + out_end*t_1)
      squiggles.append(in_start*(1-t_1) + in_end*t_1)
      squiggles.append(in_start*(1-t_2) + in_end*t_2)
    
    # squiggles.append(curr - a_o)
    # squiggles.append( curr)

    # lines.append([squiggles, svgwrite.rgb(100, 10, 16, '%')])
    squaggles += squiggles
    min_x = min(*[ s[0] for s in squiggles], min_x)
    min_y = min(*[ s[1] for s in squiggles], min_y)
    max_x = max(*[ s[0] for s in squiggles], max_x)
    max_y = max(*[ s[1] for s in squiggles], max_y)
    # dwg.add(dwg.line(prev+outer, curr+outer, stroke=svgwrite.rgb(100, 10, 16, '%')))
    # dwg.add(dwg.line(prev-inner, curr-inner, stroke=svgwrite.rgb(100, 10, 16, '%')))
  squaggles.append(squaggles[0])
  outline.append(outline[0])
  # etchline.append(etchline[0])

  lines.append([outline, svgwrite.rgb(0, 0, 0, "%")])
  lines.append([squaggles, svgwrite.rgb(100, 0, 0, '%')])
  lines.append([innerLine, svgwrite.rgb(0, 100, 0, '%')])
  # lines.append([etchline, svgwrite.rgb(0,0,100,'%')])
  minp = np.array([min_x, min_y])
  # print("%imm"%max_x, "%imm"%max_y)

  size = ("%imm"%(max_x - min_x),"%imm"%(max_y- min_y))
  # size = (max_x, max_y)
  dwg = svgwrite.Drawing(filename, size=size, profile="tiny")


  # properly scaled to mms...
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
    print(self.min_length)
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
# for k, he in ti.half_edges.items():
#   print(k, he["length"]*100, 180*he["dihedral"]/pi)

