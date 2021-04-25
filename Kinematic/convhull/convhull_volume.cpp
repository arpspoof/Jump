#include "convhull_volume.h"
#define CONVHULL_3D_ENABLE
#include "convhull_3d.h"
#include <Eigen/Dense>

double convexHullVolume(const Eigen::MatrixXd& verts)
{
  // convert data
  int nVerts = verts.cols();
  ch_vertex* vertices;
  vertices = (ch_vertex*)malloc(nVerts*sizeof(ch_vertex));
  for (int i = 0; i < nVerts; i++) {
    vertices[i].x = verts(0, i);
    vertices[i].y = verts(1, i);
    vertices[i].z = verts(2, i);
  }
  int* faceIndices = NULL;
  int nFaces;
  convhull_3d_build(vertices, nVerts, &faceIndices, &nFaces);
  Eigen::Map<Eigen::MatrixXi> faces(faceIndices, 3, nFaces);
  double vol = polygonVolume(verts, faces);
  free(vertices);
  free(faceIndices);
  return vol;
}

double polygonVolume(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
{
  double vol = 0;
  Eigen::Vector3d com = vertices.rowwise().mean();
  for (int i=0; i < faces.cols(); i++)
  {
    Eigen::Vector3i f = faces.col(i);
    Eigen::Vector3d a = vertices.col(f(0)) - com;
    Eigen::Vector3d b = vertices.col(f(1)) - com;
    Eigen::Vector3d c = vertices.col(f(2)) - com;

    double e_vol = a.cross(b).dot(c);
    vol += e_vol;
  }
  vol /= 6.0;
  return vol;
}
