"""Synthetic algebra only; never reads real pixels, GT, calibration or fits."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/framework/near_goal_segmentation_20261103/full32_FIT_selection_renderer_C_diagnostic_v1/analyze.py'
SPEC=importlib.util.spec_from_file_location('independent_measurement_C',SOURCE)
C=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(C)


def plane():
    y,x=np.indices((9,9));a,b,c=.002,-.0013,1.
    q=c+a*(x-4)+b*(y-4)
    return q,1/(q-.5*max(abs(a),abs(b)))


class RendererCIndependentTests(unittest.TestCase):
    def test_source_half_max_rule_exact_on_known_affine_inverse_depth(self):
        q,z=plane();pixels=np.array([[4,4],[3,3]],np.int64)
        result=C.correct_classic_stencils(z,np.ones_like(z,bool),pixels)
        self.assertTrue(result['accepted'].all())
        np.testing.assert_allclose(result['depth_corrected'],1/q[pixels[:,1],pixels[:,0]],rtol=0,atol=1e-12)
        self.assertTrue(np.all(result['metric_residual_max']<=.00025))
        flat=np.full((9,9),1.3)
        unchanged=C.correct_classic_stencils(flat,np.ones_like(flat,bool),pixels)
        self.assertTrue(unchanged['accepted'].all())
        np.testing.assert_allclose(unchanged['depth_corrected'],1.3,rtol=0,atol=1e-12)

    def test_original_centre_depth_is_not_replaced_by_LS_intercept(self):
        _,z=plane();z[4,4]+=.0001
        r=C.correct_classic_stencils(z,np.ones_like(z,bool),np.array([[4,4]]))
        self.assertTrue(r['accepted'][0])
        a,b,c=r['coefficients'][0]
        expected=1/(1/z[4,4]+.5*max(abs(a),abs(b)))
        wrong=1/(c+.5*max(abs(a),abs(b)))
        self.assertAlmostEqual(r['depth_corrected'][0],expected,places=14)
        self.assertGreater(abs(r['depth_corrected'][0]-wrong),.00005)

    def test_crease_bad_raw_support_and_image_edge_stay_explicit_unknown(self):
        y,x=np.indices((9,9));crease=1/(1+.01*np.abs(x-4))
        pixels=np.array([[4,4],[0,0],[8,8]],np.int64)
        result=C.correct_classic_stencils(crease,np.ones_like(crease,bool),pixels)
        self.assertEqual(result['accepted'].tolist(),[False]*3)
        self.assertTrue(np.isnan(result['depth_corrected']).all())
        self.assertEqual(len(result['reasons']),3)
        self.assertIn('nonplanar',result['reasons'][0])
        _,z=plane();raw=np.ones_like(z,bool);raw[3,3]=False
        self.assertFalse(C.correct_classic_stencils(z,raw,pixels[:1])['accepted'][0])
        raw[:]=True;z[3,3]=np.nan
        self.assertFalse(C.correct_classic_stencils(z,raw,pixels[:1])['accepted'][0])
        empty=C.correct_classic_stencils(z,raw,np.empty((0,2),np.int64))
        self.assertEqual(empty['accepted'].shape,(0,))

    def test_front_halfpixel_only_and_single_erosion_keep_point_ledger(self):
        depth=np.ones((4,256,256),np.float32)
        depth[0,127,127]=0  # front has no Classic3x3 requirement
        yy,xx=np.indices((3,3));depth[1,127:130,127:130]=1/(1+.01*np.abs(xx-1))
        inputs=dict(depth_m=depth,K=np.repeat(np.array([[-100.,0,128],[0,-100.,128],[0,0,1.]])[None],4,0),
            T=np.repeat(np.eye(4)[None],4,0))
        crops=dict(valid_depth=np.ones((4,64,64),bool),depth_m=depth[:,96:160,96:160],
                   origins_xy=np.full((4,2),96))
        raw_crop=np.zeros((4,64,64),bool);raw_crop[:2,31:34,31:34]=True
        raw_full=np.zeros((4,256,256),bool);raw_full[:2,127:130,127:130]=True
        points,views,pixels,z,ledger=C.cloud_c(inputs,crops,raw_crop,raw_full)
        self.assertEqual([row['baseline_candidate_points'] for row in ledger],[1,1,0,0])
        self.assertEqual([row['accepted_points'] for row in ledger],[1,0,0,0])
        self.assertEqual([row['rejected_points'] for row in ledger],[0,1,0,0])
        self.assertEqual(len(points),1);self.assertEqual(views.tolist(),[0])
        np.testing.assert_allclose(points[0],[-.005,-.005,1.],rtol=0,atol=1e-14)
        self.assertEqual(z.tolist(),[1.]);self.assertEqual(pixels.tolist(),[[128,128]])
        self.assertEqual(ledger[0]['reasons'],['accepted_front_GL3_halfpixel_only'])

    def test_fixed140_denominator_unknown_is_not_deleted_or_filled(self):
        rows=[]
        for i in range(224):
            eligible=i<140;available=i<126
            rows.append(dict(scene_id=f'synthetic{i}',condition='near_goal' if eligible else 'sphere_only',
                A=dict(measurement=dict(refiner_available=eligible)),
                C=dict(measurement=dict(refiner_available=available),
                       error_xyz_m=[.001,0,0] if available else None)))
        result=C.stats(rows,'C')
        self.assertEqual(result['samples'],224);self.assertEqual(result['fixed_baseline_eligible'],140)
        self.assertEqual(result['available'],126);self.assertEqual(result['coverage'],.9)
        self.assertEqual(len(result['missing_from_baseline']),14)
        self.assertEqual(result['available_measurement_error']['n'],126)
        self.assertEqual(result['negatives'],84);self.assertEqual(result['negative_falseavailable'],0)
        for row in rows:
            row['C']['measurement']['refiner_available']=False;row['C']['error_xyz_m']=None
        none=C.stats(rows,'C');self.assertEqual(none['fixed_baseline_eligible'],140)
        self.assertEqual(none['coverage'],0.);self.assertEqual(none['available_measurement_error']['n'],0)
        self.assertIsNone(none['available_measurement_error']['p95_m'])

    def test_no_accepted_points_cannot_reuse_baseline_centre_or_clear_crop_guard(self):
        inp=dict(public_sphere_center=np.array([0.,0.,1.]))
        baseline=dict(reasons=[],centre=[0,0,1],diagnostic_centre_only=[0,0,1],refiner_available=True)
        result,_=C.measurement_c(inp,baseline,np.empty((0,3)),np.empty(0,np.int64))
        self.assertFalse(result['refiner_available']);self.assertIsNone(result['centre'])
        self.assertIsNone(result['diagnostic_centre_only']);self.assertEqual(result['measured_points'],0)
        guard='raw_Cube_mask_touches_crop_boundary_before_erosion'
        baseline['reasons']=[guard]
        points=np.repeat([[0.,0.,1.025]],60,0);views=np.repeat([0,1,2],20)
        kept,_=C.measurement_c(inp,baseline,points,views)
        self.assertFalse(kept['refiner_available']);self.assertIn(guard,kept['reasons'])
        self.assertIsNone(kept['centre'])


if __name__=='__main__':unittest.main()
