"""CPU-only prototype: conservative axis-yaw swept triangle envelopes.

Nothing imports this module in the runtime. A retained cavity is supported by
an axis ray to the exterior, in addition to a radial surface exclusion. This
prevents a closed solid sphere/cylinder from acquiring a false centre hole.
Closed native material components and open collision-only surfaces must be
audited separately before a solid-material interpretation is permitted.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

SCHEMA='r10_TRAIN_bin_axis_yaw_swept_PROTOTYPE_v1'


def _cross(a,b):return float(a[0]*b[1]-a[1]*b[0])


def _hull(points):
    values=sorted(set(map(tuple,np.asarray(points,float))))
    if len(values)<=2:return np.asarray(values,float)
    low=[]
    for point in values:
        while len(low)>=2 and _cross(np.subtract(low[-1],low[-2]),np.subtract(point,low[-1]))<=0:low.pop()
        low.append(point)
    high=[]
    for point in reversed(values):
        while len(high)>=2 and _cross(np.subtract(high[-1],high[-2]),np.subtract(point,high[-1]))<=0:high.pop()
        high.append(point)
    return np.asarray(low[:-1]+high[:-1],float)


def _segment_distance(first,second):
    delta=second-first;den=float(delta@delta)
    if den==0:return float(np.linalg.norm(first))
    parameter=float(np.clip(-first@delta/den,0.,1.))
    return float(np.linalg.norm(first+parameter*delta))


def projected_radial_bounds(points,*,margin):
    """Origin distance to the projected convex polygon, not vertex minimum."""
    values=np.asarray(points,float);polygon=_hull(values)
    maximum=float(np.linalg.norm(values,axis=1).max())
    if len(polygon)==1:minimum=float(np.linalg.norm(polygon[0]))
    elif len(polygon)==2:minimum=_segment_distance(*polygon)
    else:
        cross=[_cross(polygon[(i+1)%len(polygon)]-point,-point) for i,point in enumerate(polygon)]
        if min(cross)>=-margin*max(1.,maximum):minimum=0.
        else:minimum=min(_segment_distance(point,polygon[(i+1)%len(polygon)]) for i,point in enumerate(polygon))
    return max(0.,minimum-margin),maximum+margin


def clip_triangle_to_slab(triangle,low,high):
    polygon=np.asarray(triangle,float)
    for bound,lower in ((low,True),(high,False)):
        if not len(polygon):break
        output=[];previous=polygon[-1];previous_inside=bool(previous[0]>=bound if lower else previous[0]<=bound)
        for current in polygon:
            inside=bool(current[0]>=bound if lower else current[0]<=bound)
            if inside!=previous_inside:
                parameter=(bound-previous[0])/(current[0]-previous[0])
                point=previous+parameter*(current-previous);point[0]=bound;output.append(point)
            if inside:output.append(current.copy())
            previous,previous_inside=current,inside
        polygon=np.asarray(output,float).reshape(-1,3)
    return polygon


def _outward_float32(value,direction,steps=4):
    result=np.float32(value)
    target=np.float32(np.inf if direction>0 else -np.inf)
    for _ in range(steps):result=np.nextafter(result,target)
    if not np.isfinite(result):raise ValueError('prototype mesh float32 bound overflow')
    return float(result)


def _exact_disk_inside_polygon(polygon,radius):
    """Exact rational certificate for stored float32 vertices/radius."""
    rr=Fraction.from_float(float(radius))**2
    for i,point in enumerate(polygon):
        other=polygon[(i+1)%len(polygon)]
        x,y=map(lambda v:Fraction.from_float(float(v)),point)
        u,v=map(lambda z:Fraction.from_float(float(z)),other)
        cross=x*v-y*u;edge2=(u-x)**2+(v-y)**2
        if cross<=0 or cross*cross<rr*edge2:return False
    return True


def _exact_polygon_inside_disk(polygon,radius):
    rr=Fraction.from_float(float(radius))**2
    return all(sum(Fraction.from_float(float(v))**2 for v in point)<=rr for point in polygon)


def _radial_polygons(minimum,maximum,segments):
    angles=np.arange(segments)*2*np.pi/segments
    directions=np.c_[np.cos(angles),np.sin(angles)]
    outer_radius=_outward_float32(maximum/math.cos(np.pi/segments),1)
    for _ in range(16):
        outer=np.asarray(directions*outer_radius,np.float32)
        if _exact_disk_inside_polygon(outer,maximum):break
        outer_radius=_outward_float32(outer_radius,1)
    else:raise ValueError('failed outward float32 polygon certificate')
    inner=None;inner_radius=0.
    if minimum>0:
        inner_radius=max(0.,_outward_float32(minimum,-1))
        for _ in range(16):
            inner=np.asarray(directions*inner_radius,np.float32)
            if _exact_polygon_inside_disk(inner,minimum):break
            inner_radius=max(0.,_outward_float32(inner_radius,-1))
        else:raise ValueError('failed inward float32 polygon certificate')
    return outer,inner,outer_radius,inner_radius


def _prism(low,high,outer,inner):
    n=len(outer);vertices=[];triangles=[]
    for x in (low,high):vertices.extend([[x,*point] for point in outer])
    for i in range(n):
        j=(i+1)%n;triangles.extend(((i,j,n+j),(i,n+j,n+i)))
    if inner is None:
        ca,cb=2*n,2*n+1;vertices.extend(([low,0.,0.],[high,0.,0.]))
        for i in range(n):
            j=(i+1)%n;triangles.extend(((ca,j,i),(cb,n+i,n+j)))
    else:
        for x in (low,high):vertices.extend([[x,*point] for point in inner])
        for i in range(n):
            j=(i+1)%n;ai,aj=2*n+i,2*n+j;bi,bj=3*n+i,3*n+j
            triangles.extend(((ai,bi,bj),(ai,bj,aj),
                (bi,n+i,n+j),(bi,n+j,bj),(ai,j,i),(ai,aj,j)))
    return np.asarray(vertices,np.float32),np.asarray(triangles,np.int32)


@dataclass(frozen=True)
class SweptPrototype:
    vertices: np.ndarray
    triangles: np.ndarray
    slabs: tuple[dict,...]
    receipt: dict


def partial_collision_axis_ready(evidence,*,axis_local):
    """Pure proposed readiness gate; never infer polarity from a gauge pose."""
    axis=np.asarray(axis_local,float)
    if axis.shape!=(3,) or not np.array_equal(axis,np.array([1.,0.,0.])):return False
    if not isinstance(evidence,dict) or evidence.get('axis_observable') is not True or evidence.get('axis_polarity')!='directed':return False
    world=np.asarray(evidence.get('axis_world'),float)
    return bool(world.shape==(3,) and np.isfinite(world).all() and abs(float(np.linalg.norm(world))-1.)<=1e-6
        and isinstance(evidence.get('evidence'),str) and evidence['evidence'])


def axis_yaw_swept_prototype(vertices,triangles,*,slab_count=32,angular_segments=64):
    """Conservative local-X sweep of all source triangle surfaces.

    For closed bounded material components, an axis escape ray proves that the
    retained central cylinder is exterior. For open collision-only components
    the explicit guarantee is triangle-surface containment. Input provenance,
    material topology and current directed axis must be audited by a caller.
    """
    vertices,triangles=np.asarray(vertices,float),np.asarray(triangles)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all()
            or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.dtype.kind not in 'iu'
            or triangles.min()<0 or triangles.max()>=len(vertices)
            or type(slab_count) is not int or not 2<=slab_count<=128
            or type(angular_segments) is not int or not 8<=angular_segments<=256):
        raise ValueError('explicit finite TRAIN triangles and preregistered prototype discretization required')
    extent=float(np.ptp(vertices[:,0]));scale=float(np.linalg.norm(vertices,axis=1).max())
    if extent<=0 or scale<=0:raise ValueError('prototype requires finite nondegenerate local-X extent')
    margin=128*np.finfo(float).eps*max(1.,scale)
    faces=vertices[triangles];edges=np.linspace(float(vertices[:,0].min()),float(vertices[:,0].max()),slab_count+1)
    axis_hits=[]
    for face in faces:
        rmin,_=projected_radial_bounds(face[:,1:],margin=margin)
        if rmin<=margin:axis_hits.append((float(face[:,0].min()),float(face[:,0].max())))
    slabs=[];mesh_vertices=[];mesh_triangles=[];offset=0
    for index,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        radial=[];count=0
        for face in faces:
            if face[:,0].max()<low or face[:,0].min()>high:continue
            polygon=clip_triangle_to_slab(face,low,high)
            if not len(polygon):continue
            radial.append(projected_radial_bounds(polygon[:,1:],margin=margin));count+=1
        if not radial:continue
        minimum=min(row[0] for row in radial);maximum=max(row[1] for row in radial)
        plus=not any(last>=low-margin for first,last in axis_hits)
        minus=not any(first<=high+margin for first,last in axis_hits)
        if not (plus or minus):minimum=0.
        minimum=max(0.,_outward_float32(minimum,-1)) if minimum>0 else 0.
        maximum=_outward_float32(maximum,1)
        outer,inner,rout,rin=_radial_polygons(minimum,maximum,angular_segments)
        stored_low,stored_high=_outward_float32(low-margin,-1),_outward_float32(high+margin,1)
        v,t=_prism(stored_low,stored_high,outer,inner)
        mesh_vertices.append(v);mesh_triangles.append(t+offset);offset+=len(v)
        slabs.append(dict(index=index,source_axial_interval=[float(low),float(high)],
            stored_axial_interval=[stored_low,stored_high],source_intersecting_triangles=count,
            radial_lower_bound=minimum,radial_upper_bound=maximum,
            outer_vertex_radius=rout,inner_vertex_radius=rin,
            axis_escape_positive_x=plus,axis_escape_negative_x=minus,
            retained_cavity=inner is not None,outer_polygon_yz=outer.tolist(),
            inner_polygon_yz=inner.tolist() if inner is not None else None,
            exact_float32_polygon_containment_certificates=True))
    if not slabs:raise ValueError('empty swept prototype')
    receipt=dict(schema=SCHEMA,runtime_installed=False,axis_local=[1.,0.,0.],
        requires_current_axis_observable=True,requires_current_axis_polarity='directed',
        axis_gauge_or_native_GT_is_not_permission=True,slab_count=slab_count,angular_segments=angular_segments,
        numeric_coordinate_margin=margin,retained_cavity_slabs=sum(s['retained_cavity'] for s in slabs),
        exact_CAD_claimed=False,all_axis_yaw_source_triangles_enclosed=True,
        construction='triangle/slab clipping; projected convex-polygon origin distance; min/max radial union; '
                     'circumscribed outer N-gon and inscribed inner N-gon; axial float32 outward rounding',
        cavity_certificate='radial exclusion plus axis ray to bounded-object exterior without any source surface crossing',
        material_precondition='closed bounded respondable native components audited; open collidable-only components interpreted as surfaces',
        default_cube_changed=False,native_pose_or_future_state_used=False)
    return SweptPrototype(np.concatenate(mesh_vertices),np.concatenate(mesh_triangles),tuple(slabs),receipt)


def points_inside_prototype(points,prototype):
    """CPU diagnostic membership in the union of constructed polygon prisms."""
    points=np.asarray(points,float);result=np.zeros(len(points),bool)
    for slab in prototype.slabs:
        indices=np.flatnonzero((points[:,0]>=slab['stored_axial_interval'][0])&(points[:,0]<=slab['stored_axial_interval'][1])&~result)
        if not len(indices):continue
        yz=points[indices,1:]
        def within(polygon,strict=False):
            polygon=np.asarray(polygon,float);edge=np.roll(polygon,-1,axis=0)-polygon
            delta=yz[:,None,:]-polygon[None,:,:]
            cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
            return np.all(cross>1e-14 if strict else cross>=-1e-14,axis=1)
        material=within(slab['outer_polygon_yz'])
        if slab['inner_polygon_yz'] is not None:material&=~within(slab['inner_polygon_yz'],strict=True)
        result[indices[material]]=True
    return result


def verify_stored_sweep_exact(vertices,triangles,prototype,*,inflation_m=0.):
    """Slow diagnostic certificate for the *stored* finite input and envelope.

    Fraction arithmetic removes clipping/closest-point rounding from this
    post-check. The angular sweep itself is proved by radial invariance. This
    does not establish manifold topology, correct axis perception or native
    collision-engine semantics, and is deliberately outside the runtime.
    """
    if not np.isfinite(inflation_m) or inflation_m<0:raise ValueError('finite nonnegative explicit inflation required')
    fraction=lambda value:Fraction.from_float(float(value))
    inflation=fraction(inflation_m)
    faces=[[[fraction(v) for v in p] for p in face]
           for face in np.asarray(vertices,float)[np.asarray(triangles,int)]]
    def clip(face,low,high):
        polygon=face
        for bound,lower in ((low,True),(high,False)):
            if not polygon:break
            output=[];previous=polygon[-1]
            previous_inside=previous[0]>=bound if lower else previous[0]<=bound
            for current in polygon:
                inside=current[0]>=bound if lower else current[0]<=bound
                if inside!=previous_inside:
                    parameter=(bound-previous[0])/(current[0]-previous[0])
                    output.append([a+parameter*(b-a) for a,b in zip(previous,current)])
                if inside:output.append(current)
                previous,previous_inside=current,inside
            polygon=output
        return polygon
    def radial2(polygon):
        yz=[p[-2:] for p in polygon]
        norm2=lambda p:sum(v*v for v in p)
        maximum=max(map(norm2,yz));crosses=[];distances=[]
        for first,second in zip(yz,yz[1:]+yz[:1]):
            a,b=first;u,v=second;dx,dy=u-a,v-b
            crosses.append(a*v-b*u);length=dx*dx+dy*dy;dot=a*dx+b*dy
            if length and -length<dot<0:distance=norm2(first)-dot*dot/length
            else:distance=min(norm2(first),norm2(second))
            distances.append(distance)
        nonzero=any(crosses)
        inside=nonzero and (min(crosses)>=0 or max(crosses)<=0)
        return (Fraction(0) if inside else min(distances)),maximum
    axis_hits=[(min(p[0] for p in face),max(p[0] for p in face))
               for face in faces if radial2(face)[0]==0]
    checked=0
    for slab in prototype.slabs:
        low,high=map(fraction,slab['source_axial_interval']);bounds=[]
        for face in faces:
            if max(p[0] for p in face)<low or min(p[0] for p in face)>high:continue
            polygon=clip(face,low,high)
            if polygon:bounds.append(radial2(polygon));checked+=1
        if not bounds:raise ValueError('stored slab has no exact source intersections')
        maximum=max(pair[1] for pair in bounds);minimum=min(pair[0] for pair in bounds)
        upper=fraction(slab['radial_upper_bound']);lower=fraction(slab['radial_lower_bound'])
        if upper<inflation or (upper-inflation)**2<maximum:
            raise ValueError('stored radial upper bound omits exact source radius plus inflation')
        if lower>0 and (lower+inflation)**2>minimum:
            raise ValueError('stored radial lower bound omits source radial exclusion minus inflation')
        outer=[[fraction(v) for v in p] for p in slab['outer_polygon_yz']]
        for (a,b),(u,v) in zip(outer,outer[1:]+outer[:1]):
            cross=a*v-b*u;edge2=(u-a)**2+(v-b)**2
            if cross<=0 or cross*cross<upper*upper*edge2:
                raise ValueError('exact source radial maximum escapes stored outer polygon')
        inner=slab['inner_polygon_yz']
        if inner is not None:
            if any(sum(fraction(v)**2 for v in p)>lower*lower for p in inner):
                raise ValueError('stored cavity exceeds exact source radial minimum')
            plus=not any(last>=low for first,last in axis_hits)
            minus=not any(first<=high for first,last in axis_hits)
            if not (plus or minus):raise ValueError('retained cavity lacks exact axis escape certificate')
        stored_low,stored_high=map(fraction,slab['stored_axial_interval'])
        if stored_low>low-inflation or stored_high<high+inflation:raise ValueError('stored axial interval is not outward')
    return dict(exact_rational_source_triangle_slab_clips=True,
        exact_rational_polygon_radial_extrema=True,exact_stored_float32_outer_inner_and_axial_enclosure=True,
        exact_axis_escape_for_every_retained_cavity=True,checked_triangle_slab_intersections=checked,
        axial_radial_inflation_m=float(inflation_m),
        arithmetic='Fraction of exact IEEE float input/output values; no tolerance in certificate',
        topology_or_perception_certified=False)
