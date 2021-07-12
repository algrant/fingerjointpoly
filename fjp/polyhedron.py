import numpy as np
import fjp.tripy as tripy
from collections import namedtuple

from math import pi, sin

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

    inner = norm*(material_thickness/sin(dihedrals[i]))
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
    self.faces = np.array(faces)

    self.face_polygons = np.array([
      self.vertices[f] for f in faces
    ])

    self.face_transforms = [find_flat_transform(f) for f in self.face_polygons]

    # FJP info
    self.material_thickness = 3 
    self.tab_width = 2
    self.border_width = 5

    self.generate_half_edges()
    self.find_dihedrals()
    self.determine_offsets()

    for k, v in self.half_edges.items():
      print(v)

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

  def get_2d_face(self, face_id):
      t_3d_2d, t_2d_3d = self.face_transforms[face_id]
      return apply_transform_to_points(t_3d_2d, self.face_polygons[face_id])

  def determine_offsets(self):
    # determine offsets for every half edge using their face & dihedral information
    #   Basically trying to find space where finger joints won't overlap neighbours.
    #       o[0]              o[1]
    #      *--|--------------|--*  
    #    ./                     |  o[0]
    #    /    *--------------*  +  
    #  ./    /               |  |
    #  /    *                |  |

    for face_idx in range(len(self.faces)):
      hes = self.half_edges_for_face_id(face_idx)
      dihedrals = [he.dihedral for he in hes]
      face_2d = self.get_2d_face(face_idx)[:,0:2]

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

if __name__ == "__main__":
  v = [
    [0,0,0],
    [0,2,0],
    [0,2,1],
    [0,0,1],
    [1,0,0],
    [1,2,0],
    [1,2,1],
    [1,0,1],  
  ]

  f = [
    [0, 1, 2, 3],
    [7, 6, 5, 4]
  ]

  p = Polyhedron(v, f)

  print(p.as_triangles())
