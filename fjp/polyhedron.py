import numpy as np
from collections import namedtuple
import pyclipper

# not sure what the right way to do this is...
try:
  # for running locally
  import tripy
except:
  #  for running via www/api/fjp_crud
  import fjp.tripy as tripy

import os

from math import pi, sin, tan, cos, sqrt

import svgwrite


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


""" https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python """
def get_dihedral(p0, p1, p2, p3):
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

# this feels like it should be easier...
def apply_transform_to_points(transform, points):
  if (points.shape[1] != 3):
    print("warning... points matrix must have an (n, 3) shape")

  # add 1s to homogenize...
  fancy_points = np.ones((points.shape[0], 4))
  fancy_points[:, 0:3] = points

  transformed_points = transform.dot(fancy_points.transpose()).transpose()
  return transformed_points[:, 0:3]

# polygon3D is at least 3 vertices in a numpy array
def find_flat_transform(polygon3D):
    # calc a & b as directional vectors
    a = polygon3D[0] - polygon3D[1]
    b = polygon3D[2] - polygon3D[1]

    #            a 
    #    p0 <--------- p1
    #                     \  
    #                      \  b
    #                       v
    #                       p2

    # calculate normal vector for a & b (and normalize)
    # (given p0, p1, p2 lie on a plane must be orthogonal)
    normal_vector = np.cross(a, b)
    normal_vector = normal_vector/np.linalg.norm(normal_vector)


    # # u can be either a or b, but also must be normalized
    u = a/np.linalg.norm(a)

    # # v is orthogonal to both u & norm
    v = np.cross(u, normal_vector)
    v /= np.linalg.norm(v)

    # # Create 2D to 3D Transform 
    # # (https://cs184.eecs.berkeley.edu/uploads/lectures/04_transforms-1/04_transforms-1_slides.pdf)

    t_2d_3d = np.zeros((4,4))
    t_2d_3d[0:3, 0] = u
    t_2d_3d[0:3, 1] = v
    t_2d_3d[0:3, 2] = normal_vector
    t_2d_3d[0:3, 3] = polygon3D[1]
    t_2d_3d[3,3] = 1

    t_3d_2d = np.linalg.inv(t_2d_3d)

    return t_3d_2d, t_2d_3d

# convert 2d polygon into a set of triangles
def get_triangle_array(polygon2D):
  # polygon2D points in format
  # np.array([[x0,y0,0], [x1,y1,0], [x2,y2,0], [x3,y3,0], ..., [xn, yn, 0]])
  triangles = tripy.earclip(polygon2D[:,0:2])

  # reshape into list of 2D points
  triArray = np.array(triangles).reshape((-1,2))
  
  # create a column of zeros
  zeros = np.zeros((triArray.shape[0],1))
  
  # append column of zeros
  triArray3D = np.append(triArray, zeros, axis=1)
  
  return triArray3D


def determine_min_offsets(face_2d, dihedrals, material_thickness):
  innerSplines = []
  innerLines = []
  innerIntersections = []
  minOffsets = []

  # generate innerSplines, find intersections & add to innerLine
  for i in range(len(face_2d)):
    prev = face_2d[i-1]
    curr = face_2d[i]

    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)
    outset, inset = outsetInset(material_thickness, dihedrals[i], 0)
    inner = norm*inset
    in_start, in_end = prev - inner, curr - inner
    innerLines.append([in_start, in_end])
    innerSplines.append(spline(in_start, in_end))

    if i > 0:
      prev_inner_spline = innerSplines[i-1]
      inner_spline = innerSplines[i]
      innerIntersections.append(np.array(intersection(prev_inner_spline, inner_spline)))

    if i == len(face_2d) - 1:
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
  out = pyclipper.scale_from_clipper(pco.Execute(pyclipper.scale_to_clipper(offset))[0])
  return [np.array(p) for p in out]

def outsetInset(material_thickness, dihedral, overlap, style="simple"):
  if not (pi/2 < dihedral < pi):
    # this calc is "full" coverage
    # i.e. fingers will have no gaps after sanding to plane
    # fingers get infinitely long as we approach 0 / 180 degrees
    if style == "full":
      outset = -material_thickness/tan(dihedral) + overlap
      inset = material_thickness/sin(dihedral)
      return [-outset, inset]
    
    # this metric is defined such that finger A overlaps
    # finger B by material thickness
    # In essence is a smooth transition from 90 (outset of 0 inset r) 
    # to 180 (outset r inset r) 
    rtd = material_thickness/tan(dihedral/2)
    outset2 = material_thickness - rtd
    inset2 = material_thickness*cos(pi - dihedral) + rtd

    return [-outset2, inset2]

  print(f"warning: dihedral of ({dihedral}). currently only works for dihedrals between 90 & 180 degrees")
  # to do implement other angles... (see overlap_calc_diagrams.svg)
  return [0.00000001, 0.000000001]

def get_fjpolygon(face_2d, dihedrals, offsets, overlap, tab_width, border_width):
  material_thickness = 3
  innerSplines = []
  innerLine = []

  # generate innerSplines
  for i in range(len(face_2d)):
    prev = face_2d[i-1]
    curr = face_2d[i]

    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)
    
    outset, inset = outsetInset(material_thickness, dihedrals[i], 0)

    inner = norm*inset
    in_start, in_end = prev - inner, curr - inner

    innerSplines.append(spline(in_start, in_end))

    if i > 0:
      prev_inner_spline = innerSplines[i-1]
      inner_spline = innerSplines[i]
      innerLine.append(np.array(intersection(prev_inner_spline, inner_spline)))

    if i == len(face_2d) - 1:
      prev_inner_spline = innerSplines[0]
      inner_spline = innerSplines[i]
      innerLine.append(np.array(intersection(prev_inner_spline, inner_spline)))

  try:
    window_line = gen_offset_polyline(innerLine, -border_width)
  except:
    window_line = []

  fjp_cut_line = []
  outline = []

  for i in range(len(face_2d)):
    prev = face_2d[i-1]
    curr = face_2d[i]
    length = vec_mag(curr-prev)


    # dwg.add(dwg.line(prev, curr, stroke=svgwrite.rgb(10, 10, 16, '%')))
    v = (curr - prev)
    v /= vec_mag(v)

    norm = rotate_via_numpy(v, pi/2)
    norm /= vec_mag(norm)

    tab_diff = 0 #0.01

    # v1
    # outer = norm*(material_thickness/tan(dihedrals[i]) - overlap)
    # inner = norm*(material_thickness/sin(dihedrals[i]))

    outset, inset = outsetInset(material_thickness, dihedrals[i], overlap)
    outer = norm*outset
    inner = norm*inset


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


    # lines.append([squiggles, svgwrite.rgb(100, 10, 16, '%')])
    fjp_cut_line += squiggles

  return [gen_offset_polyline(fjp_cut_line, 0.05), window_line, outline]

def svg_poly(face_2d, filename, dihedrals, offsets, overlap, tab_width, border_width):
  fjp_cut_line, window_line, outline = get_fjpolygon(face_2d, dihedrals, offsets, overlap, tab_width, border_width)
  
  # close open polygons
  fjp_cut_line.append(fjp_cut_line[0])
  outline.append(outline[0])

  if len(window_line) > 0:
    window_line.append(window_line[0])

  lines = [
    [outline, "fill:none;stroke:#000000"],
    [fjp_cut_line, "fill:none;stroke:#ff0000"],
  ]

  if len(window_line) > 0:
    lines.append([window_line, "fill:none;stroke:#ff0000"])

  tl, br = get_bounding_box(fjp_cut_line+outline)
  minp = np.array(tl)

  size = ("%imm"%(br[0] - tl[0]),"%imm"%(br[1]- tl[1]))

  dwg = svgwrite.Drawing(filename, size=size, profile="full")

  # properly scaled to mms...
  pixies_per_mm = 3.543307

  for line in lines:
    pline = [(point - minp)*pixies_per_mm for point in line[0]]
    dwg.add(dwg.polyline(pline, style=line[1]))
  dwg.save()

class Shape:
  # https://svgwrite.readthedocs.io/en/latest/classes/path.html
  def __init__(self, polys):
    self.polys = polys

  def points(self):
    points = []
    for poly in self.polys:
      points += poly
    return points

  def bounding_box(self):
    return get_bounding_box(self.points())

  def translate(self, v):
    self.polys = [[p + v for p in poly] for poly in self.polys]
  
  def normalize(self):
    self.translate(-self.bounding_box()[0])


def svg_from_polys(shapes, filename):
  # in mms
  max_width = 200

  curr = np.array([0, 0])
  curr_bottom = 0

  lines = []
  for shape in shapes:
    shape.normalize()
    bb = shape.bounding_box()[1]
    margin = 2
    if curr[0] + bb[0] + margin > max_width:
      curr = np.array([0, curr[1] + curr_bottom + margin])
      curr_bottom = 0
    
    shape.translate(curr)
    curr[0] = curr[0] + bb[0] + margin

    curr_bottom = max(curr_bottom, bb[1])


  max_height = curr_bottom + curr[1]

  dwg = svgwrite.Drawing(filename, size=(f"{max_width}mm", f"{max_height}mm"), profile="full")

  # properly scaled to mms...
  pixies_per_mm = 3.543307

  for shape in shapes:
    path = dwg.path(fill="none", stroke="#f00")
    for line in shape.polys:
      pline = [point*pixies_per_mm for point in line]
      x = f"M {pline[-1][0]} {pline[-1][1]} L " + " ".join([f"{p[0]} {p[1]}" for p in pline])
      path.push(x)

    dwg.add(path)

  dwg.save()



class HalfEdge:
  def __init__(self, id, s, e, opp, prev, next, face_id, length, dihedral ):
    self.id = id             # s_e
    self.s = s               # start vertex id
    self.e = e               # end vertex id
    self.opp = opp           # e_s (opposite half edge)
    self.prev = prev         # previous halfedge around face  
    self.next = next         # next halfedge around face
    self.face_id = face_id   # face
    self.length = length     # distance between v_s & v_e,
    self.dihedral = None     # dihedral angle between faces

    # FJP stuff
    self.offsets = None      # [s, e] how far from corners is reasonable to interlace

  def __repr__(self):
    di_str = "??" if self.dihedral == None else "%04.2f"%self.dihedral
    off_str = "??" if self.offsets == None else "(%01.2f, %01.2f)"%(self.offsets[0], self.offsets[1])
    return "(%s -> p: %s, n: %s, f: %s, l: %04.2f, d: %s, o:%s)"%(
      self.id,
      self.prev,
      self.next,
      self.face_id,
      self.length,
      di_str,
      off_str
    )
class Polyhedron:
  def __init__(self, vertices, faces):
    self.vertices = np.array(vertices)
    self.faces = faces

    self.face_polygons = [
      self.vertices[f] for f in faces
    ]

    self.face_transforms = [find_flat_transform(f) for f in self.face_polygons]

    # FJP info
    self.overlap = 2
    self.material_thickness = 3 
    self.tab_width = 2
    self.border_width = 3

    self.generate_half_edges()
    self.find_dihedrals()


  def half_edges_for_face_id(self, face_idx):
    vids = self.faces[face_idx]
    return [self.half_edges["%i_%i"%(vids[i-1],vids[i])] for i in range(len(vids))]

  def as_triangles(self):
    triangles = False
  
    for id, face_polygon in enumerate(self.face_polygons):
      t_3d_2d, t_2d_3d = self.face_transforms[id]
      
      flat_poly = apply_transform_to_points(t_3d_2d, face_polygon)
      poly_triangles = get_triangle_array(flat_poly)
      poly3d_triangles = apply_transform_to_points(t_2d_3d, poly_triangles)

      if id == 0:
        triangles = poly3d_triangles
      else:
        triangles = np.append(triangles, poly3d_triangles, axis=0)

    return triangles

  def generate_half_edges(self):
    self.half_edges = {}
    self.min_length = 10000000000000000
    self.max_length = -10000000000000000

    for face_id, face in enumerate(self.faces):
      face_len = len(face)
      for i in range(face_len):
        pre_m1 = face[i-2]
        prev = face[i-1]
        curr = face[i]
        curr_1 = face[(i+1)%face_len]

        prev_v = self.vertices[face[i-1]]
        curr_v = self.vertices[face[i]]
        length = vec_mag(curr_v-prev_v)
        self.min_length = min(length, self.min_length)
        self.max_length = max(length, self.max_length)

        self.half_edges["%i_%i"%(prev, curr)] = HalfEdge( 
           "%i_%i"%(prev, curr), 
          prev, 
          curr, 
          "%i_%i"%(curr, prev), 
          "%i_%i"%(pre_m1, prev),  
         "%i_%i"%(curr, curr_1), 
          face_id,
          length,
          None
        )

  def find_dihedrals(self):
    for key, he in self.half_edges.items():
      try:
        if he.dihedral == None:
          opp_he = self.half_edges[he.opp]
          p0_id = self.half_edges[opp_he.next].e
          p1_id = he.s
          p2_id = he.e
          p3_id = self.half_edges[he.next].e

          p0 = self.vertices[p0_id]
          p1 = self.vertices[p1_id]
          p2 = self.vertices[p2_id]
          p3 = self.vertices[p3_id]
          he.dihedral = get_dihedral(p0, p1, p2, p3)
          opp_he.dihedral = he.dihedral
      except:
        print("failed on dihedral", key)

  def get_2d_face(self, face_id):
      t_3d_2d, t_2d_3d = self.face_transforms[face_id]
      return apply_transform_to_points(t_3d_2d, self.face_polygons[face_id])

  def determine_offsets(self, scale):
    # determine offsets for every half edge using their face & dihedral information
    #   Basically trying to find space where finger joints won't overlap neighbouring faces
    #   in retrospect - this is only "valid" if the vertices have 3 faces, otherwise this will no longer
    #   work...
    #       o[0]              o[1]
    #      *--|--------------|--*  
    #    ./                     |  o[0]
    #    /    *--------------*  +  
    #  ./    /               |  |
    #  /    *                |  |

    for face_idx in range(len(self.faces)):
      hes = self.half_edges_for_face_id(face_idx)
      dihedrals = [he.dihedral for he in hes]
      face_2d = self.get_2d_face(face_idx)[:,0:2]*scale

      offsets = determine_min_offsets(face_2d, dihedrals, self.material_thickness)
      for i in range(len(hes)):
        he = hes[i]
        opp_he = self.half_edges[he.opp]
        hes[i].offsets = offsets[i]

        if opp_he.offsets != None:
          start_offset = max(opp_he.offsets[1], he.offsets[0])
          end_offset = max(opp_he.offsets[0], he.offsets[1])
          
          ## now we want to make sure that the edge length - offsets is a multiple of tab_width*2
          current_length = he.length - start_offset - end_offset
          add_to_offsets = (current_length%(2*self.tab_width)) / 2

          he.offsets[0] = start_offset + add_to_offsets
          he.offsets[1] = end_offset + add_to_offsets
          opp_he.offsets[1] = he.offsets[0]
          opp_he.offsets[0] = he.offsets[1]

  def save_3d_fjp(self, scale=40):
    self.determine_offsets(scale)
    new_verts = []
    new_faces = []
    triangles = []
    for face_idx in range(len(self.faces)):
      face_2d = self.get_2d_face(face_idx)[:,0:2]*scale
      dihedrals = [he.dihedral for he in self.half_edges_for_face_id(face_idx)]
      offsets = [he.offsets for he in self.half_edges_for_face_id(face_idx)]
      new_poly_2d, window, orig = get_fjpolygon(face_2d, dihedrals, offsets, self.overlap, self.tab_width, self.border_width)
      t_3d_2d, t_2d_3d = self.face_transforms[face_idx]
      

      if len(window) > 0:
        last = new_poly_2d[-1]
        window.reverse()
        new_poly_2d.append(window[-1])
        new_poly_2d += window
        new_poly_2d.append(last)

      np_poly_2d = np.array(new_poly_2d)/scale
      # create a column of zeros
      zeros = np.zeros((np_poly_2d.shape[0],1))

      # append column of zeros
      np2_poly_2d = np.append(np_poly_2d, zeros, axis=1)

      poly_triangles = get_triangle_array(np2_poly_2d)
      
      poly_triangles_bottom_face = np.flip(poly_triangles.copy(), 0)
      poly_triangles_bottom_face[:,2] = 1.0*self.material_thickness/scale

      top_and_bottom = np.append(poly_triangles, poly_triangles_bottom_face, axis=0)
      poly3d_triangles = apply_transform_to_points(t_2d_3d, top_and_bottom)

      if face_idx == 0:
        triangles = poly3d_triangles
      else:
        triangles = np.append(triangles, poly3d_triangles, axis=0)
    return triangles

  def gen_svg_faces(self, scale, path):
    self.determine_offsets(scale)

    ensure_dir(path)
    for face_idx in range(len(self.faces)):
      face_2d = self.get_2d_face(face_idx)[:,0:2]*scale
      dihedrals = [he.dihedral for he in self.half_edges_for_face_id(face_idx)]
      offsets = [he.offsets for he in self.half_edges_for_face_id(face_idx)]
      svg_poly(face_2d, "%s/num%i.svg"%(path, face_idx), dihedrals, offsets, self.overlap, self.tab_width, self.border_width)

  def gen_fjp_svg(self, scale, path):
    self.determine_offsets(scale)

    shapes = []
    for face_idx in range(len(self.faces)):
      face_2d = self.get_2d_face(face_idx)[:,0:2]*scale
      dihedrals = [he.dihedral for he in self.half_edges_for_face_id(face_idx)]
      offsets = [he.offsets for he in self.half_edges_for_face_id(face_idx)]
      new_poly_2d, window, orig = get_fjpolygon(face_2d, dihedrals, offsets, self.overlap, self.tab_width, self.border_width)
      window.reverse()
      shapes.append(Shape([new_poly_2d, window]))

    svg_from_polys(shapes, path)




if __name__ == "__main__":
  data = {'vertices': [(-0.3882212544999999, 0.22575205180000002, 0.10882729580000006), (-0.49441745857999986, 0.3996130939000002, 0.09102012770000001), (-0.49441745857999986, 0.3996130939000002, -0.09102012769999979), (-0.3882212544999999, 0.22575205180000002, -0.10882729579999983), (-0.31141054175999994, 0.10000000000000009, -0.06544614629999979), (-0.31141054175999994, 0.10000000000000009, 0.06544614630000001), (0.34071611720000017, 1.2, 0.12434754600000009), (0.4515207295000001, 1.1573828414000003, 0.049904329900000155), (0.4515207295000001, 1.1573828414000003, -0.04990432989999993), (0.34071611720000017, 1.2, -0.12434754599999986), (0.11521419820000012, 1.2, 0.34391201390000004), (0.24250179400000005, 1.1573828414000003, 0.3841279621000002), (0.32053534640000003, 1.1573828414000003, 0.32189828060000014), (0.3096518507000001, 1.2, 0.1888531602000001), (-0.1970463619999998, 1.2, 0.3045037207000001), (-0.1491259385999999, 1.1573828414000003, 0.42909540410000013), (-0.05181969029999989, 1.1573828414000003, 0.4513049202000001), (0.04541342480000021, 1.2, 0.3598435849000001), (-0.36092699262999994, 1.2, 0.035797915200000086), (-0.42845879775999984, 1.1573828414000003, 0.1509452548000001), (-0.3851534432699999, 1.1573828414000003, 0.2408697500000001), (-0.25302223619999986, 1.2, 0.2598644507000001), (-0.25302223619999986, 1.2, -0.25986445067999986), (-0.3851534432699999, 1.1573828414000003, -0.24086974999999988), (-0.42845879775999984, 1.1573828414000003, -0.15094525479999987), (-0.36092699262999994, 1.2, -0.035797915199999863), (0.04541342480000021, 1.2, -0.3598435848899999), (-0.05181969029999989, 1.1573828414000003, -0.4513049202399999), (-0.1491259385999999, 1.1573828414000003, -0.4290954040499999), (-0.1970463619999998, 1.2, -0.3045037207399999), (0.3096518507000001, 1.2, -0.18885316019999987), (0.32053534640000003, 1.1573828414000003, -0.3218982805899999), (0.24250179400000005, 1.1573828414000003, -0.3841279620699999), (0.11521419820000012, 1.2, -0.3439120138599999), (0.4441978737000001, 1.0924810159, 0.3021426290000002), (0.5131774638000002, 1.0924810159, 0.15890503420000002), (0.04072822470000004, 1.0924810159, 0.5356707300000001), (0.1957239567000002, 1.0924810159, 0.5002939656000001), (-0.39341060816999984, 1.0924810159, 0.36582784560000015), (-0.26911368178999984, 1.0924810159, 0.4649513367000002), (-0.5313032289999999, 1.0924810159, -0.07949086799999994), (-0.5313032289999999, 1.0924810159, 0.07949086800000016), (-0.26911368178999984, 1.0924810159, -0.4649513367299999), (-0.39341060816999984, 1.0924810159, -0.3658278456199999), (0.1957239567000002, 1.0924810159, -0.50029396559), (0.04072822470000004, 1.0924810159, -0.5356707299799999), (0.4441978737000001, 1.0924810159, -0.3021426289799999), (0.5131774638000002, 1.0924810159, -0.1589050341999998), (0.5858032385, 1.0160328320000003, 0.09897563120000008), (0.5858032385, 1.0160328320000003, -0.09897563119999986), (0.2878600806, 1.0160328320000003, 0.5197097111000002), (0.44262460960000016, 1.0160328320000003, 0.3962891177000001), (-0.22684758919999992, 1.0160328320000003, 0.5490917783), (-0.033859378199999846, 1.0160328320000003, 0.5931400781), (-0.5707343975199999, 1.0160328320000003, 0.16499653710000017), (-0.4848465636199999, 1.0160328320000003, 0.3433444619000001), (-0.4848465636199999, 1.0160328320000003, -0.3433444618899999), (-0.5707343975199999, 1.0160328320000003, -0.16499653709999995), (-0.033859378199999846, 1.0160328320000003, -0.5931400781199999), (-0.22684758919999992, 1.0160328320000003, -0.5490917783299999), (0.44262460960000016, 1.0160328320000003, -0.39628911766999986), (0.2878600806, 1.0160328320000003, -0.5197097110499999), (0.5080701871000002, 0.9263101329000001, 0.36704993910000017), (0.6037477783, 0.9263101329000001, 0.16837337380000017), (0.029805382300000183, 0.9263101329000001, 0.6260771614), (0.24479097820000018, 0.9263101329000001, 0.5770081020000002), (-0.47090348330999987, 0.9263101329000001, 0.4136555115), (-0.2984984212899999, 0.9263101329000001, 0.5511439606000001), (-0.6170124212999999, 0.9263101329000001, 0.11025717550000014), (-0.6170124212999999, 0.9263101329000001, -0.11025717549999992), (-0.2984984212899999, 0.9263101329000001, -0.5511439605799999), (-0.47090348330999987, 0.9263101329000001, -0.4136555115099999), (0.24479097820000018, 0.9263101329000001, -0.5770081020099999), (0.029805382300000183, 0.9263101329000001, -0.6260771613699999), (0.5080701871000002, 0.9263101329000001, -0.36704993906999994), (0.6037477783, 0.9263101329000001, -0.16837337379999995), (0.6248701318000001, 0.8206983652, 0.11477594300000016), (0.6248701318000001, 0.8206983652, -0.11477594299999994), (0.2998647090000002, 0.8206983652, 0.5601047715), (0.4793356004, 0.8206983652, 0.4169815115000002), (-0.2509449558599999, 0.8206983652, 0.5836632830000001), (-0.027148414799999854, 0.8206983652, 0.6347433831), (-0.6127879505899999, 0.8206983652, 0.1677114378000002), (-0.5131891199299999, 0.8206983652, 0.37453054070000014), (-0.5131891199299999, 0.8206983652, -0.3745305407299999), (-0.6127879505899999, 0.8206983652, -0.16771143779999997), (-0.027148414799999854, 0.8206983652, -0.6347433830599999), (-0.2509449558599999, 0.8206983652, -0.5836632829699999), (0.4793356004, 0.8206983652, -0.41698151152999985), (0.2998647090000002, 0.8206983652, -0.5601047715099999), (0.4985908871000002, 0.6976566087, 0.36565420400000015), (0.5967463018000001, 0.6976566087, 0.1618323852000001), (0.024986365000000177, 0.6976566087, 0.6177957196000001), (0.24553957980000019, 0.6976566087, 0.5674558876000002), (-0.46743339954999985, 0.6976566087, 0.4047244576000002), (-0.29056345381999993, 0.6976566087, 0.5457735326000002), (-0.6078662803299999, 0.6976566087, -0.11311257589999979), (-0.6078662803299999, 0.6976566087, 0.11311257590000001), (-0.29056345381999993, 0.6976566087, -0.5457735325999999), (-0.46743339954999985, 0.6976566087, -0.40472445758999986), (0.24553957980000019, 0.6976566087, -0.5674558875499999), (0.024986365000000177, 0.6976566087, -0.6177957195899999), (0.5967463018000001, 0.6976566087, -0.16183238519999987), (0.4985908871000002, 0.6976566087, -0.36565420400999993), (0.5646210049, 0.5571084349000002, 0.10523867650000018), (0.5646210049, 0.5571084349000002, -0.10523867649999996), (0.26975652800000005, 0.5571084349000002, 0.5070537188000002), (0.4343143489000001, 0.5571084349000002, 0.3758232357000002), (-0.22824011649999987, 0.5571084349000002, 0.5270469689000001), (-0.023039870199999868, 0.5571084349000002, 0.5738825861000001), (-0.5543672980399998, 0.5571084349000002, 0.15016310160000002), (-0.4630445971299999, 0.5571084349000002, 0.3397966440000002), (-0.4630445971299999, 0.5571084349000002, -0.33979664399999987), (-0.5543672980399998, 0.5571084349000002, -0.1501631015999998), (-0.023039870199999868, 0.5571084349000002, -0.5738825860699999), (-0.22824011649999987, 0.5571084349000002, -0.5270469689099999), (0.4343143489000001, 0.5571084349000002, -0.3758232357499999), (0.26975652800000005, 0.5571084349000002, -0.5070537188499998), (0.4059625846000001, 0.3996130939000002, 0.29652599700000004), (0.4849468912000001, 0.3996130939000002, 0.13251339420000008), (0.021280171600000175, 0.3996130939000002, 0.5022752645000002), (0.19875629770000014, 0.3996130939000002, 0.46176749680000007), (-0.37942664461999986, 0.3996130939000002, 0.3298010132000002), (-0.23710184193999984, 0.3996130939000002, 0.44330125600000003), (-0.23710184193999984, 0.3996130939000002, -0.4433012559499999), (-0.37942664461999986, 0.3996130939000002, -0.3298010132399999), (0.19875629770000014, 0.3996130939000002, -0.46176749681999985), (0.021280171600000175, 0.3996130939000002, -0.5022752644499999), (0.4849468912000001, 0.3996130939000002, -0.13251339419999986), (0.4059625846000001, 0.3996130939000002, -0.29652599697999993), (0.3969936582000002, 0.22575205180000002, 0.0703928840000001), (0.3969936582000002, 0.22575205180000002, -0.07039288399999988), (0.19248612450000002, 0.22575205180000002, 0.35427138560000015), (0.3025568701000001, 0.22575205180000002, 0.26649289500000006), (-0.1569673869999999, 0.22575205180000002, 0.3713763080000001), (-0.019711412199999945, 0.22575205180000002, 0.4027040886000002), (-0.3271365990799999, 0.22575205180000002, 0.23567088980000017), (-0.3271365990799999, 0.22575205180000002, -0.23567088980999984), (-0.019711412199999945, 0.22575205180000002, -0.40270408861999984), (-0.1569673869999999, 0.22575205180000002, -0.3713763080499999), (0.3025568701000001, 0.22575205180000002, -0.26649289502999984), (0.19248612450000002, 0.22575205180000002, -0.35427138559999993), (0.2521751846000002, 0.10000000000000009, 0.1940809106000001), (0.30896722190000014, 0.10000000000000009, 0.07615102990000011), (0.13310063930000005, 0.10000000000000009, 0.2890396917000002), (0.0054900898000000975, 0.10000000000000009, 0.3181659669000001), (-0.14299343939999987, 0.10000000000000009, 0.28427557030000017), (-0.24532915453999993, 0.10000000000000009, 0.20266556070000008), (-0.24532915453999993, 0.10000000000000009, -0.20266556069999986), (-0.14299343939999987, 0.10000000000000009, -0.28427557029999984), (0.0054900898000000975, 0.10000000000000009, -0.3181659669099999), (0.13310063930000005, 0.10000000000000009, -0.2890396917199999), (0.2521751846000002, 0.10000000000000009, -0.19408091059999988), (0.30896722190000014, 0.10000000000000009, -0.07615102989999989)], 'faces': [[5, 4, 3, 2, 1, 0], [9, 8, 7, 6], [13, 12, 11, 10], [17, 16, 15, 14], [21, 20, 19, 18], [25, 24, 23, 22], [29, 28, 27, 26], [33, 32, 31, 30], [13, 6, 7, 35, 34, 12], [17, 10, 11, 37, 36, 16], [21, 14, 15, 39, 38, 20], [25, 18, 19, 41, 40, 24], [29, 22, 23, 43, 42, 28], [33, 26, 27, 45, 44, 32], [47, 8, 9, 30, 31, 46], [8, 47, 49, 48, 35, 7], [34, 51, 50, 37, 11, 12], [36, 53, 52, 39, 15, 16], [38, 55, 54, 41, 19, 20], [40, 57, 56, 43, 23, 24], [42, 59, 58, 45, 27, 28], [44, 61, 60, 46, 31, 32], [48, 63, 62, 51, 34, 35], [50, 65, 64, 53, 36, 37], [52, 67, 66, 55, 38, 39], [69, 57, 40, 41, 54, 68], [56, 71, 70, 59, 42, 43], [58, 73, 72, 61, 44, 45], [75, 49, 47, 46, 60, 74], [49, 75, 77, 76, 63, 48], [62, 79, 78, 65, 50, 51], [64, 81, 80, 67, 52, 53], [66, 83, 82, 68, 54, 55], [69, 85, 84, 71, 56, 57], [70, 87, 86, 73, 58, 59], [72, 89, 88, 74, 60, 61], [76, 91, 90, 79, 62, 63], [78, 93, 92, 81, 64, 65], [80, 95, 94, 83, 66, 67], [82, 97, 96, 85, 69, 68], [84, 99, 98, 87, 70, 71], [86, 101, 100, 89, 72, 73], [88, 103, 102, 77, 75, 74], [102, 105, 104, 91, 76, 77], [90, 107, 106, 93, 78, 79], [92, 109, 108, 95, 80, 81], [94, 111, 110, 97, 82, 83], [96, 113, 112, 99, 84, 85], [98, 115, 114, 101, 86, 87], [100, 117, 116, 103, 88, 89], [104, 119, 118, 107, 90, 91], [106, 121, 120, 109, 92, 93], [108, 123, 122, 111, 94, 95], [110, 1, 2, 113, 96, 97], [112, 125, 124, 115, 98, 99], [114, 127, 126, 117, 100, 101], [116, 129, 128, 105, 102, 103], [128, 131, 130, 119, 104, 105], [118, 133, 132, 121, 106, 107], [120, 135, 134, 123, 108, 109], [122, 136, 0, 1, 110, 111], [2, 3, 137, 125, 112, 113], [124, 139, 138, 127, 114, 115], [126, 141, 140, 129, 116, 117], [130, 143, 142, 133, 118, 119], [142, 144, 132, 133], [144, 145, 135, 120, 121, 132], [145, 146, 134, 135], [146, 147, 136, 122, 123, 134], [147, 5, 0, 136], [4, 148, 137, 3], [148, 149, 139, 124, 125, 137], [149, 150, 138, 139], [150, 151, 141, 126, 127, 138], [151, 152, 140, 141], [152, 153, 131, 128, 129, 140], [153, 143, 130, 131], [143, 153, 152, 151, 150, 149, 148, 4, 5, 147, 146, 145, 144, 142], [10, 17, 14, 21, 18, 25, 22, 29, 26, 33, 30, 9, 6, 13]]}

  v = data['vertices']

  f = data['faces']

  p = Polyhedron(v, f)

  p.gen_fjp_svg(121, "./svg_out/test.svg")