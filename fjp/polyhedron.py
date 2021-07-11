import numpy as np
import tripy

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

class Polyhedron:
  def __init__(self, vertices, faces):
    self.vertices = np.array(vertices)
    self.faces = np.array(faces)

    self.face_polygons = np.array([
      self.vertices[f] for f in faces
    ])

    self.face_transforms = [find_flat_transform(f) for f in self.face_polygons]

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
