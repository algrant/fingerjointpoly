import numpy as np
from math import pi, cos, tan, sin
import numpy as np
import svgwrite

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

def svg_poly(face, filename, dihedrals, offset_multi, overlap, min_length):
  material_thickness = 3
  innerSplines = []
  innerLine = []

  # generate innerSplines
  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]

    v = (curr - prev)
    v /= np.linalg.norm(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= np.linalg.norm(norm)

    inner = norm*(material_thickness/sin(dihedrals[i]))
    in_start, in_end = prev - inner, curr - inner

    innerSplines.append(spline(in_start, in_end))

    if i > 0:
      prev_inner_spline = innerSplines[i-1]
      inner_spline = innerSplines[i]
      innerLine.append(np.array(intersection(prev_inner_spline, inner_spline)))

    if i == len(face) - 1:
      prev_inner_spline = innerSplines[0]
      inner_spline = innerSplines[i]
      innerLine.append(np.array(intersection(prev_inner_spline, inner_spline)))

  lines = []
  min_x = 1000000000
  min_y = 1000000000
  max_x = -100000000
  max_y = -100000000
  squaggles = []
  outline = []

  for i in range(len(face)):
    prev_minus_one = face[i-2]
    prev = face[i-1]
    curr = face[i]
    length = np.linalg.norm(curr-prev)
    innerLength = np.linalg.norm(innerLine[i] - innerLine[i-1])


    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= np.linalg.norm(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= np.linalg.norm(norm)

    tab_diff = 0.0003
    # tab_offset = v*tab_diff/2.0 
    etch = norm*(material_thickness/tan(dihedrals[i]))
    outer = norm*(material_thickness/tan(dihedrals[i]) - overlap)
    inner = norm*(material_thickness/sin(dihedrals[i]))


    offset_multi = (length - innerLength)/2

    a_curr = innerLine[i] - curr + inner
    a_prev = innerLine[i-1] - prev + inner

    # make both offsets the same larger length because we want tabs to remain symmetric
    if np.linalg.norm(a_curr) < np.linalg.norm(a_prev):
      a_curr = -a_prev
    else:
      a_prev = -a_curr

    outline.append(curr)
    out_start, out_end = prev - outer + a_prev, curr - outer + a_curr
    in_start, in_end = prev - inner + a_prev, curr - inner + a_curr
    etch_start, etch_end = prev - etch + a_prev, curr - etch + a_curr

    lines.append([[etch_start, etch_end], "fill:none;stroke:#0000ff"])

    squiggles = [in_start]
    length = np.linalg.norm(out_start - out_end)
    divs = int(length/6)

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
  
  # close open polygons
  innerLine.append(innerLine[0])
  squaggles.append(squaggles[0])
  outline.append(outline[0])
  # etchline.append(etchline[0])


  lines = [
    [outline, "fill:none;stroke:#000000"],
    [squaggles, "fill:none;stroke:#ff0000"],
    [innerLine, "fill:none;stroke:#00ff00"]
  ] + lines
  # lines.append([outline, "fill:none;stroke:#000000"])
  # lines.append([squaggles, "fill:none;stroke:#ff0000"])
  # lines.append([innerLine, "fill:none;stroke:#00ff00"])
  # # lines.append([etchline, svgwrite.rgb(0,0,100,'%')])
  minp = np.array([min_x, min_y])
  # print("%imm"%max_x, "%imm"%max_y)

  size = ("%imm"%(max_x - min_x),"%imm"%(max_y- min_y))
  print(size)
  # size = (max_x, max_y)
  dwg = svgwrite.Drawing(filename, size=size, profile="full")

  i = 1

  # properly scaled to mms...
  pixies_per_mm = 3.543307

  points_total = len(innerLine) - 1
  for p in range(points_total):
    point = innerLine[p]
    dwg.add(dwg.circle(center=(point - minp)*pixies_per_mm, r=15, style="fill:none;stroke:#ff00dd" ))
    # dwg.add(dwg.circle(center=(outline[p] - minp)*pixies_per_mm, r=15, stroke="none", fill=svgwrite.rgb(0,100.0*p/points_total,100.0*p/points_total, "%") ))


  for line in lines:
    dwg.add(dwg.polyline([(point - minp)*pixies_per_mm for point in line[0]], style=line[1]))
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

def get_bounding_box(vertices):
  tl = vertices[0].copy()
  br = vertices[0].copy()
  for v in vertices:
    tl[0] = min(tl[0], v[0])
    br[0] = max(tl[0], v[0])
    tl[1] = min(tl[1], v[1])
    br[1] = max(tl[1], v[1])

  return [tl, br]

class Polyhedron:
  def __init__(self, vertices=[], faces=[]):
    self.vertices = vertices
    self.faces = faces
    self.half_edges = {}
    self.faces_2d = {}

  def load_from_file(self, filename):
    pass

  def generate_half_edges(self):
    for face_id, face in enumerate(self.faces):
      face_len = len(face)
      for i in range(face_len):
        pre_m1 = face[i-2]
        prev = face[i-1]
        curr = face[i]
        curr_1 = face[(i+1)%face_len]
        self.half_edges["%i_%i"%(prev, curr)] = { "id": "%i_%i"%(prev, curr), "s": prev, "e": curr, "opp": "%i_%i"%(curr, prev), "prev":"%i_%i"%(pre_m1, prev),  "next":"%i_%i"%(curr, curr_1), "face_id": face_id }

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

  def get_2d_face(self, scale, offset_multi, overlap, index):
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
    new_face = [np.array([scale + np.dot(u, f)*scale, scale + np.dot(v, f)*scale]) for f in face]
    top_left, bottom_right = get_bounding_box(new_face)
    n_f = [p - top_left for p in new_face]
    print(n_f)
    self.min_length
    svg_poly(new_face, "./poly/num%i.svg"%index, dihedrals, offset_multi, overlap, self.min_length)

  def get_2d_faces(self, scale, offset_multi, overlap):
    for i in range(len(self.faces)):
      self.get_2d_face(scale, offset_multi, overlap, i)