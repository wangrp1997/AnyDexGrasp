"""Bounded geometric contact-width alignment for Boya g1, diagnostic only.

One shared five-finger closure parameter, four thumb/opponent branches. A
precomputed width table initializes two fixed secant corrections with batched
exact URDF FK; rigid wrist alignment has a closed form. No learned model,
object mesh, collision search, IK, physics or production-label writer.

Targets are the original CGR width, never its 1.7x preopening request. Matching
computed pads to this reference does not certify real contact or force bearing.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from author_action_semantics import ActionContract, CGR_R_REFERENCE

GROUPS=('th','ff','mf','rf','lf')
OPPONENTS=GROUPS[1:]
GRID_SIZE=513
REFINEMENTS=2
WIDTH_TOLERANCE_M=1e-6


def unit(rows):
    norm=np.linalg.norm(rows,axis=-1,keepdims=True)
    if np.any(norm<1e-12):raise ValueError('Degenerate geometric reference')
    return rows/norm


def skew(axis):
    x,y,z=axis
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


@dataclass
class Alignment:
    valid: np.ndarray
    progress: np.ndarray
    world_T_palm: np.ndarray
    q_contact: np.ndarray
    width_m: np.ndarray
    target_world_reference: np.ndarray
    palm_T_contact_reference: np.ndarray
    pad_points_palm: np.ndarray
    pad_normals_palm: np.ndarray


class FiveFingerWidthMapper:
    def __init__(self,contract=None):
        self.contract=contract or ActionContract()
        self.slot=self.contract.slots['g1:distal_000:ff']
        if set(self.contract.active[1])!=set(GROUPS):raise ValueError('Original five-finger wrap required')
        self.q_open=np.asarray(self.slot['independent_q'],dtype=float)
        self.q_close=np.asarray(self.slot['close_q'],dtype=float)
        self.delta=self.q_close-self.q_open
        endpoints=self.contract._expand(np.stack([self.q_open,self.q_close]))
        self.contract._check_joint_limits(endpoints)
        full_names=self.contract.hand_names
        self.joint_start=dict(zip(full_names,endpoints[0]))
        self.joint_delta=dict(zip(full_names,endpoints[1]-endpoints[0]))
        self.joint_start.update(self.slot['wrist'])
        self.joint_delta.update({k:0. for k in self.slot['wrist']})
        # Compile only descendants of palm. Wrist coordinates are already part
        # of the arm-to-palm interface and do not change palm-local hand FK.
        self.tree=[];reached={'palm'};remaining=dict(self.contract.fk.joints)
        while True:
            found=[]
            for name,j in remaining.items():
                if j['parent']in reached:
                    if j['kind']not in ('fixed','revolute','continuous'):raise ValueError('Unsupported hand joint')
                    k=skew(j['axis']);self.tree.append((name,j,k,k@k));reached.add(j['child']);found.append(name)
            for name in found:del remaining[name]
            if not found:break
        if any(self.contract.pads[g]['collision_link']not in reached for g in GROUPS):raise ValueError('Pad outside palm tree')
        self.progress_grid=np.linspace(0.,1.,GRID_SIZE)
        points,_=self.pad_fk(self.progress_grid)
        self.width_grid=np.linalg.norm(points[:,1:]-points[:,:1],axis=-1).T
        self.segments=[]
        for widths in self.width_grid:
            signs=np.sign(np.diff(widths));signs[abs(np.diff(widths))<1e-14]=0
            for i in range(1,len(signs)):
                if signs[i]==0:signs[i]=signs[i-1]
            cuts=np.r_[0,np.flatnonzero(signs[1:]*signs[:-1]<0)+1,GRID_SIZE-1]
            self.segments.append([(int(a),int(b))for a,b in zip(cuts[:-1],cuts[1:])])

    def frames(self,progress):
        s=np.asarray(progress,dtype=float).reshape(-1)
        if not np.isfinite(s).all()or np.any((s<0)|(s>1)):raise ValueError('Progress must stay inside original closure')
        rotations={'palm':np.broadcast_to(np.eye(3),(len(s),3,3))}
        positions={'palm':np.zeros((len(s),3))}
        for name,j,k,k2 in self.tree:
            parent,child=j['parent'],j['child'];origin=j['origin']
            pre=rotations[parent]@origin[:3,:3]
            positions[child]=positions[parent]+np.einsum('nij,j->ni',rotations[parent],origin[:3,3])
            if j['kind']=='fixed':rotations[child]=pre
            else:
                angle=self.joint_start[name]+s*self.joint_delta[name]
                motion=np.eye(3)+np.sin(angle)[:,None,None]*k+(1-np.cos(angle))[:,None,None]*k2
                rotations[child]=pre@motion
        return rotations,positions

    def pad_fk(self,progress):
        rotations,positions=self.frames(progress);points=[];normals=[]
        for group in GROUPS:
            pad=self.contract.pads[group];link=pad['collision_link'];r=rotations[link]
            points.append(positions[link]+np.einsum('nij,j->ni',r,pad['point_link_m']))
            normals.append(np.einsum('nij,j->ni',r,pad['normal_link']))
        return np.stack(points,axis=1),unit(np.stack(normals,axis=1))

    def lookup(self,target):
        target=np.asarray(target,dtype=float).reshape(-1);n=len(target)
        progress=np.full((n,4),np.inf);bracket=np.zeros((n,4),dtype=int)
        for pair,widths in enumerate(self.width_grid):
            for a,b in self.segments[pair]:
                index=np.arange(a,b+1)
                if widths[a]>widths[b]:index=index[::-1]
                ascending=widths[index]
                valid=(target>=ascending[0])&(target<=ascending[-1])
                j=np.clip(np.searchsorted(ascending,target),1,len(index)-1)
                left=np.minimum(index[j-1],index[j]);right=left+1
                denominator=widths[right]-widths[left]
                fraction=np.divide(target-widths[left],denominator,out=np.zeros(n),where=abs(denominator)>1e-14)
                s=self.progress_grid[left]+np.clip(fraction,0,1)/(GRID_SIZE-1)
                choose=valid&(s<progress[:,pair]);progress[choose,pair]=s[choose];bracket[choose,pair]=left[choose]
        valid=np.isfinite(progress);progress[~valid]=0.
        return progress,bracket,valid

    def align(self,rows,world_T_camera,extra_depth=0.):
        rows=np.asarray(rows,dtype=float)
        if rows.ndim!=2 or rows.shape[1]!=15 or not len(rows)or not np.isfinite(rows).all():raise ValueError('Finite Nx15 actual post-flip CGR rows required')
        if np.any(rows[:,1]<=0)or np.any(rows[:,2]<0):raise ValueError('Positive widths/nonnegative depths required')
        world=np.asarray(world_T_camera,dtype=float)
        if world.shape!=(4,4)or not np.isfinite(world).all()or not np.allclose(world[3],[0,0,0,1]):raise ValueError('World camera rigid transform required')
        if np.max(abs(world[:3,:3].T@world[:3,:3]-np.eye(3)))>2e-6 or np.linalg.det(world[:3,:3])<0:raise ValueError('Proper camera rotation required')
        rotation=rows[:,3:12].reshape(-1,3,3)
        if np.max(abs(rotation.transpose(0,2,1)@rotation-np.eye(3)))>2e-6 or np.any(np.linalg.det(rotation)<=0):raise ValueError('Proper original CGR rotations required')
        u,_,vt=np.linalg.svd(rotation);rotation=u@vt
        extra=np.broadcast_to(np.asarray(extra_depth,dtype=float),(len(rows),))
        if not np.isfinite(extra).all()or np.any(extra<0):raise ValueError('Finite nonnegative frozen extra depth required')
        rotation=world[:3,:3]@rotation
        reference=np.broadcast_to(np.eye(4),(len(rows),4,4)).copy()
        reference[:,:3,:3]=rotation@CGR_R_REFERENCE
        reference[:,:3,3]=rows[:,12:15]@world[:3,:3].T+world[:3,3]+rotation[:,:,0]*(rows[:,2]+extra)[:,None]
        s,bracket,valid=self.lookup(rows[:,1]);flat_pair=np.tile(np.arange(4),len(rows))
        target=np.repeat(rows[:,1],4);gridpair=flat_pair
        low=bracket.ravel()/(GRID_SIZE-1);high=(bracket.ravel()+1)/(GRID_SIZE-1)
        slope=(self.width_grid[gridpair,bracket.ravel()+1]-self.width_grid[gridpair,bracket.ravel()])*(GRID_SIZE-1)
        flat=s.ravel()
        # Fixed work, not an optimizer loop. No retries or convergence-dependent
        # runtime; candidates still outside tolerance are returned invalid.
        for _ in range(REFINEMENTS):
            points,_=self.pad_fk(flat);delta=points[np.arange(len(flat)),flat_pair+1]-points[:,0]
            width=np.linalg.norm(delta,axis=1);correction=np.divide(width-target,slope,out=np.zeros_like(flat),where=abs(slope)>1e-12)
            flat=np.clip(flat-correction,low,high)
        points,normals=self.pad_fk(flat)
        delta=points[np.arange(len(flat)),flat_pair+1]-points[:,0];width=np.linalg.norm(delta,axis=1)
        valid&=(abs(width.reshape(-1,4)-rows[:,1,None])<=WIDTH_TOLERANCE_M)
        x=unit(delta);thumb_normal=normals[:,0];tangent=np.broadcast_to([0.,1.,0.],thumb_normal.shape).copy()
        tangent-=(tangent*thumb_normal).sum(-1)[:,None]*thumb_normal
        bad=np.linalg.norm(tangent,axis=-1)<1e-6
        if bad.any():
            fallback=np.eye(3)[np.argmin(abs(thumb_normal[bad]),axis=1)]
            tangent[bad]=fallback-(fallback*thumb_normal[bad]).sum(-1)[:,None]*thumb_normal[bad]
        z=unit(tangent-(tangent*x).sum(-1)[:,None]*x);y=np.cross(z,x)
        local=np.broadcast_to(np.eye(4),(len(flat),4,4)).copy()
        local[:,:3,:3]=np.stack([x,y,z],axis=-1)
        local[:,:3,3]=(points[:,0]+points[np.arange(len(flat)),flat_pair+1])/2
        target_ref=np.repeat(reference,4,axis=0)
        palm=np.broadcast_to(np.eye(4),local.shape).copy()
        palm[:,:3,:3]=target_ref[:,:3,:3]@local[:,:3,:3].transpose(0,2,1)
        palm[:,:3,3]=target_ref[:,:3,3]-np.einsum('nij,nj->ni',palm[:,:3,:3],local[:,:3,3])
        q=self.q_open+flat[:,None]*self.delta
        result=Alignment(valid,flat.reshape(-1,4),palm.reshape(-1,4,4,4),q.reshape(-1,4,16),width.reshape(-1,4),
            reference,local.reshape(-1,4,4,4),points.reshape(-1,4,5,3),normals.reshape(-1,4,5,3))
        for array in (result.progress,result.world_T_palm,result.q_contact,result.width_m,result.palm_T_contact_reference,
                      result.pad_points_palm,result.pad_normals_palm):array[~valid]=np.nan
        return result
