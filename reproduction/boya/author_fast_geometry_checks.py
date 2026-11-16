"""Full-source geometry checks without per-ray/per-tick temporary arrays.

No vertex/triangle sampling, mesh approximation, fast-math or physics changes.
These are CPU loops over every original vertex or triangle.
"""
from contextlib import contextmanager
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation


@njit(cache=True,fastmath=False)
def _source_z(vertices,row,z):
    lo=np.inf;hi=-np.inf
    for i in range(len(vertices)):
        x,y,v=vertices[i,0],vertices[i,1],vertices[i,2]
        if not(np.isfinite(x)and np.isfinite(y)and np.isfinite(v)):
            return np.nan,np.nan
        value=(x*row[0]+y*row[1])+v*row[2]+z
        lo=min(lo,value);hi=max(hi,value)
    return lo,hi


def source_metrics(vertices,pose_xyzw):
    v,p=np.asarray(vertices,dtype=float),np.asarray(pose_xyzw,dtype=float)
    if(v.ndim!=2 or v.shape[1]!=3 or not len(v)or p.shape!=(7,)
            or not np.isfinite(p).all()or abs(np.linalg.norm(p[3:])-1)>1e-4):
        raise ValueError('Finite source vertices and native XYZW pose required')
    lo,hi=_source_z(v,Rotation.from_quat(p[3:]).as_matrix()[2],p[2])
    if not(np.isfinite(lo)and np.isfinite(hi)):
        raise ValueError('Finite source vertices and native XYZW pose required')
    return dict(source_bottom_m=float(lo),source_top_m=float(hi))


@njit(cache=True,fastmath=False)
def _first_depth(triangles,origin,direction):
    mesh=np.inf
    for i in range(len(triangles)):
        ax,ay,az=triangles[i,0,0],triangles[i,0,1],triangles[i,0,2]
        ex,ey,ez=triangles[i,1,0]-ax,triangles[i,1,1]-ay,triangles[i,1,2]-az
        fx,fy,fz=triangles[i,2,0]-ax,triangles[i,2,1]-ay,triangles[i,2,2]-az
        dx,dy,dz=direction[0],direction[1],direction[2]
        hx,hy,hz=dy*fz-dz*fy,dz*fx-dx*fz,dx*fy-dy*fx
        det=(ex*hx+ey*hy)+ez*hz
        if abs(det)<=1e-14:continue
        inv=1./det;sx,sy,sz=origin[0]-ax,origin[1]-ay,origin[2]-az
        u=((sx*hx+sy*hy)+sz*hz)*inv
        qx,qy,qz=sy*ez-sz*ey,sz*ex-sx*ez,sx*ey-sy*ex
        v=((qx*dx+qy*dy)+qz*dz)*inv
        t=((fx*qx+fy*qy)+fz*qz)*inv
        if u>=-1e-10 and v>=-1e-10 and u+v<=1+1e-10 and t>1e-10:
            mesh=min(mesh,t)
    ground=-origin[2]/direction[2] if direction[2]<-1e-9 else np.inf
    return min(mesh,ground),mesh<ground,np.isfinite(mesh)and abs(mesh-ground)<=1e-9


def first_depth(triangles,origin,direction):
    # Inputs are independently validated by the unchanged capture auditor.
    a,o,d=np.asarray(triangles,dtype=float),np.asarray(origin,dtype=float),np.asarray(direction,dtype=float)
    if a.ndim!=3 or a.shape[1:]!=(3,3)or o.shape!=(3,)or d.shape!=(3,):
        raise ValueError('Full triangles and 3D ray required')
    t,hit,tie=_first_depth(a,o,d)
    return float(t),bool(hit),bool(tie)


@contextmanager
def installed_metrics(lift):
    original=lift.source_metrics;lift.source_metrics=source_metrics
    try:yield
    finally:lift.source_metrics=original


def precompile():
    source_metrics(np.zeros((1,3)),[0.,0.,0.,0.,0.,0.,1.])
    first_depth(np.zeros((1,3,3)),np.array([0.,0.,1.]),np.array([0.,0.,-1.]))
