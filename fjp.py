import numpy as np
from math import pi, cos, tan, sin
import numpy as np
import svgwrite
import pyclipper
import os
import tripy

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def vec_norm(vector):
  return vector/vec_mag(vector)

def vec_mag(vector):
  return np.linalg.norm(vector)

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

def get_bounding_box(vertices):
  tl = vertices[0].copy()
  br = vertices[0].copy()
  for v in vertices:
    tl[0] = min(tl[0], v[0])
    tl[1] = min(tl[1], v[1])
    br[0] = max(br[0], v[0])
    br[1] = max(br[1], v[1])

  return [tl, br]

epsilon = 0.000000001

def epsilon_eq(a,b):
  return abs(a-b) < epsilon

def epsilon_gte(a, b):
  if a < b:
    return abs(a-b) < epsilon
  return True

def epsilon_lte(a,b):
  if a > b:
    return abs(a-b) < epsilon
  return True

def point_is_between_points(p1, p2, p_test):
  tl, br = get_bounding_box([p1, p2])

  less_than_br = epsilon_lte(p_test[0], br[0]) and epsilon_lte(p_test[1], br[1])
  greater_than_tl = epsilon_gte(p_test[0], tl[0]) and epsilon_gte(p_test[1], tl[1])

  return less_than_br and greater_than_tl


def determine_min_offsets(face, dihedrals):
  material_thickness = 3
  innerSplines = []
  innerLines = []
  innerIntersections = []
  minOffsets = []

  # generate innerSplines, find intersections & add to innerLine
  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]

    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

    inner = norm*(material_thickness/sin(dihedrals[i]))
    in_start, in_end = prev - inner, curr - inner
    innerLines.append([in_start, in_end])
    innerSplines.append(spline(in_start, in_end))

    if i > 0:
      prev_inner_spline = innerSplines[i-1]
      inner_spline = innerSplines[i]
      innerIntersections.append(np.array(intersection(prev_inner_spline, inner_spline)))

    if i == len(face) - 1:
      prev_inner_spline = innerSplines[0]
      inner_spline = innerSplines[i]
      innerIntersections.append(np.array(intersection(prev_inner_spline, inner_spline)))

  # calculate offsets as a percentage
  for i in range(len(innerLines)):
    prev_intersection = innerIntersections[i-1]
    curr_intersection = innerIntersections[i]
    minOffsets.append([None, None])

    if point_is_between_points(innerLines[i][0], innerLines[i][1], prev_intersection):
      minOffsets[i][0] = vec_mag(prev_intersection - innerLines[i][0])
    else:
      minOffsets[i][0] = 0

    if point_is_between_points(innerLines[i][0], innerLines[i][1], curr_intersection):
      minOffsets[i][1] = vec_mag(curr_intersection - innerLines[i][1])
    else:
      minOffsets[i][1] = 0

  return minOffsets

def gen_offset_polyline(line, offset):
  coordinates = pyclipper.scale_to_clipper(line)
  pco = pyclipper.PyclipperOffset()
  pco.AddPath(coordinates, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
  pco.MiterLimit = 20.0
  return pyclipper.scale_from_clipper(pco.Execute(pyclipper.scale_to_clipper(offset))[0])

def get_fjpolygon(face, dihedrals, offsets, overlap, tab_width, border_width):
  material_thickness = 3
  innerSplines = []
  innerLine = []

  # generate innerSplines
  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]

    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

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

  windowLine = gen_offset_polyline(innerLine, -border_width)


  squaggles = []
  outline = []

  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]
    length = vec_mag(curr-prev)


    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

    tab_diff = 0 #0.01

    outer = norm*(material_thickness/tan(dihedrals[i]) - overlap)
    inner = norm*(material_thickness/sin(dihedrals[i]))

    a_curr = -v*offsets[i][1]
    a_prev = v*offsets[i][0]

    outline.append(curr)
    out_start, out_end = prev - outer + a_prev, curr - outer + a_curr
    in_start, in_end = prev - inner + a_prev, curr - inner + a_curr

    squiggles = [innerLine[i-1], in_start]
    length = vec_mag(out_start - out_end)

    # floating point issues, as these lengths should now be very close to a
    #   a multiple of the tab length
    divs = int(round(length/(tab_width*2)))

    for i in range(divs):
      t_0 = i/divs
      t_2 = (i+1)/divs
      t_1 = (t_0 + t_2)/2+tab_diff/2.0
      squiggles.append(out_start*(1-t_0) + out_end*t_0)
      squiggles.append(out_start*(1-t_1) + out_end*t_1)
      squiggles.append(in_start*(1-t_1) + in_end*t_1)
      squiggles.append(in_start*(1-t_2) + in_end*t_2)
    
    # squiggles.append( curr)

    # lines.append([squiggles, svgwrite.rgb(100, 10, 16, '%')])
    squaggles += squiggles

  # outset squaggles -- for lasering purposes...
  # squaggles = gen_offset_polyline(squaggles, 0.05)

  # tl, br = get_bounding_box(squaggles)

  return [squaggles, windowLine, outline]



def svg_poly(face, filename, dihedrals, offsets, overlap, tab_width, border_width):
  material_thickness = 3
  innerSplines = []
  innerLine = []

  # generate innerSplines
  for i in range(len(face)):
    prev = face[i-1]
    curr = face[i]

    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

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

  windowLine = gen_offset_polyline(innerLine, -border_width)

  lines = []

  squaggles = []
  outline = []

  for i in range(len(face)):
    prev_minus_one = face[i-2]
    prev = face[i-1]
    curr = face[i]
    length = vec_mag(curr-prev)
    innerLength = vec_mag(innerLine[i] - innerLine[i-1])


    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

    tab_diff = 0 #0.01

    etch = norm*(material_thickness/tan(dihedrals[i]))
    outer = norm*(material_thickness/tan(dihedrals[i]) - overlap)
    inner = norm*(material_thickness/sin(dihedrals[i]))


    offset_multi = (length - innerLength)/2

    a_curr = -v*offsets[i][1]
    a_prev = v*offsets[i][0]

    outline.append(curr)
    out_start, out_end = prev - outer + a_prev, curr - outer + a_curr
    in_start, in_end = prev - inner + a_prev, curr - inner + a_curr
    etch_start, etch_end = prev - etch + a_prev, curr - etch + a_curr

    # lines.append([[etch_start, etch_end], "fill:none;stroke:#0000ff"])

    squiggles = [innerLine[i-1], in_start]
    length = vec_mag(out_start - out_end)

    # floating point issues, as these lengths should now be very close to a
    #   a multiple of the tab length
    divs = int(round(length/(tab_width*2)))

    for i in range(divs):
      t_0 = i/divs
      t_2 = (i+1)/divs
      t_1 = (t_0 + t_2)/2+tab_diff/2.0
      squiggles.append(out_start*(1-t_0) + out_end*t_0)
      squiggles.append(out_start*(1-t_1) + out_end*t_1)
      squiggles.append(in_start*(1-t_1) + in_end*t_1)
      squiggles.append(in_start*(1-t_2) + in_end*t_2)
    
    # squiggles.append( curr)

    # lines.append([squiggles, svgwrite.rgb(100, 10, 16, '%')])
    squaggles += squiggles

  # outset squaggles -- for lasering purposes...
  squaggles = gen_offset_polyline(squaggles, 0.05)

  tl, br = get_bounding_box(squaggles)

  # close open polygons
  innerLine.append(innerLine[0])
  windowLine.append(windowLine[0])
  squaggles.append(squaggles[0])
  outline.append(outline[0])


  lines = [
    [outline, "fill:none;stroke:#000000"],
    [squaggles, "fill:none;stroke:#ff0000"],
    [innerLine, "fill:none;stroke:#00ff00"],
    [windowLine, "fill:none;stroke:#ff0000"]
  ] + lines

  minp = np.array(tl)

  size = ("%imm"%(br[0] - tl[0]),"%imm"%(br[1]- tl[1]))

  dwg = svgwrite.Drawing(filename, size=size, profile="full")

  i = 1

  # properly scaled to mms...
  pixies_per_mm = 3.543307

  for line in lines:
    pline = [(point - minp)*pixies_per_mm for point in line[0]]
    dwg.add(dwg.polyline(pline, style=line[1]))
  dwg.save()

""" https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python """
def new_dihedral(p0, p1, p2, p3):
  """Praxeolitic formula
  1 sqrt, 1 cross product"""
  b0 = -1.0*(p1 - p0)
  b1 = p2 - p1
  b2 = p3 - p2

  # normalize b1 so that it does not influence vec_mag of vector
  # rejections that come next
  b1 /= vec_mag(b1)

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

class FingerJointPolyhedron:
  def __init__(self, vertices=[], faces=[], scale=70, overlap=3, material_thickness=3, tab_width=3, border_width=3, poly_path="./poly"):
    self.vertices = vertices
    self.faces = faces
    self.half_edges = {}
    self.faces_2d = []
    self.scale = scale
    self.overlap = overlap
    self.material_thickness = material_thickness
    self.tab_width = tab_width
    self.border_width = border_width
    self.poly_path = poly_path


  def load_from_file(self, filename):
    pass

  def print_face_connectivity(self):
    for idx in range(len(self.faces)):
      hes = self.half_edges_for_face_id(idx)
      connected_faces = [str(self.half_edges[he["opp"]]["face_id"]) for he in hes]
      print("%i connects to %s"%(idx, ", ".join(connected_faces)))

  def generate_half_edges(self):
    for face_id, face in enumerate(self.faces):
      face_len = len(face)
      for i in range(face_len):
        pre_m1 = face[i-2]
        prev = face[i-1]
        curr = face[i]
        curr_1 = face[(i+1)%face_len]
        self.half_edges["%i_%i"%(prev, curr)] = { 
          "id": "%i_%i"%(prev, curr), 
          "s": prev, 
          "e": curr, 
          "opp": "%i_%i"%(curr, prev), 
          "prev":"%i_%i"%(pre_m1, prev),  
          "next":"%i_%i"%(curr, curr_1), 
          "face_id": face_id
        }

  def find_lengths(self):
    self.min_length = 10000000000000000
    self.max_length = -10000000000000000
    for face in self.faces:
      for i in range(len(face)):
        prev = self.vertices[face[i-1]]
        curr = self.vertices[face[i]]
        length = vec_mag(curr-prev)*self.scale
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

  def half_edges_for_face_id(self, face_idx):
    vids = self.faces[face_idx]
    return [self.half_edges["%i_%i"%(vids[i-1],vids[i])] for i in range(len(vids))]

  def determine_offsets(self):
    self.t_2d_to_3d = [0]*len(self.faces)
    # determine offsets for every half edge using their face & dihedral information
    for face_idx in range(len(self.faces)):
      hes = self.half_edges_for_face_id(face_idx)
      dihedrals = [he["dihedral"] for he in hes]
      face_2d = self.get_2d_face(face_idx)
      offsets = determine_min_offsets(face_2d, dihedrals)
      for i in range(len(hes)):
        he = hes[i]
        opp_he = self.half_edges[he["opp"]]
        hes[i]["offsets"] = offsets[i]

        if "offsets" in opp_he:
          start_offset = max(opp_he["offsets"][1], he["offsets"][0])
          end_offset = max(opp_he["offsets"][0], he["offsets"][1])
          
          ## now we want to make sure that the edge length - offsets is a multiple of tab_width*2
          current_length = he["length"] - start_offset - end_offset
          add_to_offsets = (current_length%(2*self.tab_width)) / 2

          he["offsets"][0] = start_offset + add_to_offsets
          he["offsets"][1] = end_offset + add_to_offsets
          opp_he["offsets"][1] = he["offsets"][0]
          opp_he["offsets"][0] = he["offsets"][1]

  def get_face_orientation(self, face_idx):
    verts = [self.vertices[f] for f in self.faces[face_idx]]
    o = verts[1]
    u = vec_norm(verts[0] - verts[1])
    w = vec_norm(np.cross(u, verts[2] - verts[1]))
    v = vec_norm(np.cross(u, w))
    return np.array([u, w, v, o, [0,0,0,1]])

  def get_2d_face(self, face_idx):
    # polygon in 3D
    face_vert_ids = self.faces[face_idx]

    face = [self.vertices[f] for f in face_vert_ids]


    # this is generating a transformation matrix...
    # I should be saving this and applying rather than the opposite...


    # grab any three ordered vertices on the face 
    p0, p1, p2 = face[0], face[1], face[2]

    # calc a & b as directional vectors
    a = p0 - p1
    b = p2 - p1

    #            a 
    #    p0 <--------- p1
    #                     \  
    #                      \  b
    #                       v
    #                       p2


    # calculate normal vector for a & b (and normalize)
    # (given p0, p1, p2 lie on a plane must be orthogonal)
    norm = np.cross(a, b)
    norm /= vec_mag(norm)

    # u can be either a or b, but also must be normalized
    u = a/vec_mag(a)

    # v is orthogonal to both u & norm
    v = np.cross(u, norm)
    v /= vec_mag(v)

    # Create 2D to 3D Transform 
    # (https://cs184.eecs.berkeley.edu/uploads/lectures/04_transforms-1/04_transforms-1_slides.pdf)

    t_2d_3d = np.zeros((4,4))
    t_2d_3d[0:3, 0] = u
    t_2d_3d[0:3, 1] = v
    t_2d_3d[0:3, 2] = norm
    t_2d_3d[0:3, 3] = p1
    t_2d_3d[3,3] = 1
    
    self.t_2d_to_3d[face_idx] = t_2d_3d

    t_3d_2d = np.linalg.inv(t_2d_3d)

    face_points = np.zeros((4, len(face)))
    face_points[3,:] =1

    for i,vert in enumerate(face):
      face_points[0:3,i] = vert

    face_2d = t_3d_2d.dot(face_points)[0:2,:]*self.scale

    return face_2d.transpose()

  def gen_2d_faces(self):
    for face_idx in range(len(self.faces)):
      self.faces_2d.append(self.get_2d_face(face_idx))

      # dihedrals = [he["dihedral"] for he in self.half_edges_for_face_id(face_idx)]
      # offsets = [he["offsets"] for he in self.half_edges_for_face_id(face_idx)]
      # svg_poly(face_2d, "%s/num%i.svg"%(self.poly_path, face_idx), dihedrals, offsets, self.overlap, self.tab_width, self.border_width)

  def get_2d_faces(self):
    ensure_dir(self.poly_path)
    for face_idx in range(len(self.faces)):
      face_2d = self.get_2d_face(face_idx)
      dihedrals = [he["dihedral"] for he in self.half_edges_for_face_id(face_idx)]
      offsets = [he["offsets"] for he in self.half_edges_for_face_id(face_idx)]
      svg_poly(face_2d, "%s/num%i.svg"%(self.poly_path, face_idx), dihedrals, offsets, self.overlap, self.tab_width, self.border_width)

  def save_3d_orig(self):
    with open("test2.obj", 'w') as f:
      f.write("# OBJ file\n")
      for v in self.vertices:
        f.write("v %s\n" % " ".join(["%.4f"%vv for vv in v]))
      for p in self.faces:
          f.write("f")
          for i in p:
              f.write(" %d" % (i + 1))
          f.write("\n")

  def save_3d_fjp(self):
    new_verts = []
    new_faces = []
    self.t_2d_to_3d = [0]*len(self.faces)

    for face_idx in range(2): #len(self.faces)):
      face_2d = self.get_2d_face(face_idx)
      dihedrals = [he["dihedral"] for he in self.half_edges_for_face_id(face_idx)]
      offsets = [he["offsets"] for he in self.half_edges_for_face_id(face_idx)]
      new_poly_2d, window, orig = get_fjpolygon(face_2d, dihedrals, offsets, self.overlap, self.tab_width, self.border_width)

      t_3d_2d = self.t_2d_to_3d[face_idx]

      # for tri in tripy.earclip(new_poly_2d):
      #   face_points = np.zeros((4, len(tri)))
      #   face_points[3,:] =1
      #   for i, vert in enumerate(tri):
      #     face_points[0:2,i] = vert

      #   new_poly_3d = t_3d_2d.dot(face_points/self.scale)[0:3,:]
      #   np_3d = new_poly_3d.transpose()

      #   new_poly_list = [v for v in np_3d]
      #   # print("\n".join([str(i) for i in new_poly_list]))
      #   v_off = len(new_verts)
      #   new_verts += new_poly_list
      #   new_faces.append([v_off + i for i in range(len(new_poly_list))])

      
      for tri in tripy.earclip(orig):
        face_points = np.zeros((4, len(tri)))
        face_points[3,:] =1
        for i, vert in enumerate(tri):
          face_points[0:2,i] = vert

        new_poly_3d = t_3d_2d.dot(face_points/self.scale)[0:3,:]
        np_3d = new_poly_3d.transpose()

        new_poly_list = [v for v in np_3d]
        # print("\n".join([str(i) for i in new_poly_list]))
        v_off = len(new_verts)
        new_verts += new_poly_list
        new_faces.append([v_off + i for i in range(len(new_poly_list))])

      face_vert_ids = self.faces[face_idx]

      actual_face = [self.vertices[f] for f in face_vert_ids]
      # actual = this.faces
      new_poly_list = [v for v in actual_face]
      v_off = len(new_verts)
      new_verts += new_poly_list
      new_faces.append([v_off + i for i in range(len(new_poly_list))])
      
    # print(new_verts)
    with open("test2.obj", 'w') as f:
      f.write("# OBJ file\n")
      for v in new_verts:
        f.write("v %s\n" % " ".join(["%.4f"%vv for vv in v]))
      for p in new_faces:
          f.write("f")
          for i in p:
              f.write(" %d" % (i + 1))
          f.write("\n")