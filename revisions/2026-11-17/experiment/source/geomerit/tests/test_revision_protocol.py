import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import tempfile
import numpy as np
import pandas as pd
import yaml
from geomerit.io_las import read_las_file
from geomerit.features import build_features
from geomerit.revision_data import TrainingFeatureState, feature_columns
from geomerit.revision_models import boundary_flags, fit_backend, training_weights, training_subset
from geomerit.revision_decision import adjust, geometric_pool, decide, scores_from_cm, confusion, prior
from geomerit.labels import TAIL_IDS


class ProtocolTests(unittest.TestCase):
    def test_las_declared_null_and_depth_fallback(self):
        text = '~Well\nUWI. w1 : id\nNULL. -1234 : null\n~Curve\nDEPT.M : depth\nDEPTH_MD.M : depth\nGR.API : gamma\n~Ascii\n1 -1234 20\n2 2 -1234\n'
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'x.las'; p.write_text(text)
            df = read_las_file(p)
        np.testing.assert_array_equal(df.DEPTH_MD, [1, 2])
        self.assertTrue(np.isnan(df.GR.iloc[1]))
        self.assertEqual(df.source_row.tolist(), [0, 1])

    def test_training_only_category_and_coordinate_state(self):
        train = pd.DataFrame(dict(well_id=['a','a','b'], x_loc=[0,0,10], y_loc=[0,0,20], GROUP=['a','a','b']))
        state = TrainingFeatureState.fit(train, ['dist_to_xy_centroid'], ['GROUP'])
        test = pd.DataFrame(dict(well_id=['c'], x_loc=[100], y_loc=[200], GROUP=['never_seen']))
        out = state.transform(test)
        np.testing.assert_allclose(state.center, [5, 10])
        self.assertEqual(out[0, 1], -1)
        self.assertNotIn('never_seen', state.categories['GROUP'])

    def test_boundary_stops_at_gaps_and_covers_both_sides(self):
        y = [0,0,1,1,2,2]
        flags = boundary_flags(y, ['w']*6, [0,.15,.30,.45,10,10.15], radius=1)
        np.testing.assert_array_equal(flags, [1,1,1,1,0,0])
        direct = boundary_flags([0,1], ['a','b'], [0,0], radius=1)
        self.assertFalse(direct.any())

    def test_context_stops_at_gaps(self):
        df = pd.DataFrame(dict(well_id=['a']*4, DEPTH_MD=[0,.1,10,10.1], GR=[1,2,100,110]))
        df, _ = build_features(df, ['GR'], ['GR'], context_gap_max=1.)
        self.assertTrue(np.isnan(df.GR_grad.iloc[2]))
        self.assertLess(df.GR_w2_mean.iloc[1], 3)
        self.assertGreater(df.GR_w2_mean.iloc[2], 99)

    def test_explicit_missing_ablation_keeps_outlier_encoding(self):
        manifest = dict(geo_numeric=['GR_enc','GR_well_robust_z','GR_is_missing','GR_is_outlier','missing_count','outlier_count'], categorical=['curve_suite_id'])
        num, cat = feature_columns(manifest, dict(metadata='logs', no_missing=True))
        self.assertIn('GR_observed_enc', num)
        self.assertIn('GR_is_outlier', num)
        self.assertNotIn('GR_is_missing', num)
        self.assertNotIn('curve_suite_id', cat)

    def test_no_class_weight_does_not_disable_boundary(self):
        train = pd.DataFrame(dict(label_idx=[0,0,0,0,1,1,1,1], well_id=['a']*8,
                  DEPTH_MD=np.arange(8)*.1, confidence=[1]*8))
        cfg = dict(weights=dict(rho=.9995, cap=8))
        w = training_weights(train, dict(no_class_weight=True), cfg)
        self.assertGreater(w[3], w[0])
        w = training_weights(train, dict(no_class_weight=True, no_boundary_weight=True), cfg)
        np.testing.assert_allclose(w, 1.)

    def test_prior_and_gate_keep_absent_support_zero(self):
        p = np.zeros((2, 12), np.float32); p[:, [0,5]] = [.8,.2]
        experts = np.ones_like(p)
        selected = dict(tau=.3, tail_gamma=2., tail_threshold=.5, dolomite_gamma=8., dolomite_threshold=.1)
        q = adjust(p, prior([0,0,5]), experts, selected)
        self.assertTrue((q[:, 9] == 0).all())
        np.testing.assert_allclose(q.sum(axis=1), 1.)

    def test_geomean_normalizes_exponents(self):
        p = np.zeros((1,12), np.float32); p[0,:2] = [.8,.2]
        q = geometric_pool(dict(lightgbm=p, xgboost=p, catboost=p), [1,.8,.8])
        np.testing.assert_allclose(q, p, atol=1e-7)

    def test_argmax_is_not_bayes(self):
        p = np.array([[.6,.4]], dtype=np.float32)
        cost = np.array([[0,1],[10,0]], dtype=np.float32)
        self.assertEqual(decide(p, cost, 'argmax')[0], 0)
        self.assertEqual(decide(p, cost, 'bayes')[0], 1)

    def test_scores_include_all_fixed_classes(self):
        s = scores_from_cm(confusion([0,0], [0,0]), np.ones((12,12))-np.eye(12))
        self.assertEqual(s['weighted_f1'], 1.)
        self.assertAlmostEqual(s['macro_f1'], 1/12)

    def test_cap_keeps_rare_classes(self):
        y = np.r_[np.zeros(1000, int), np.array(TAIL_IDS)]
        idx = training_subset(y, 100, 123)
        self.assertTrue(set(TAIL_IDS).issubset(set(y[idx])))
        self.assertLessEqual(len(idx), 100)

    def test_noncontiguous_class_mapping_all_backends(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/revision_20261103.yaml').read_text())
        cfg['models']['lightgbm']['params'].update(min_data_in_leaf=3, num_threads=2, num_leaves=7)
        cfg['models']['xgboost']['params']['nthread'] = 2
        cfg['models']['catboost']['params']['thread_count'] = 2
        rng = np.random.default_rng(123)
        x = rng.normal(size=(120,4)).astype(np.float32)
        y = np.array([0,4,9]*40)
        for backend in ['lightgbm','xgboost','catboost']:
            with self.subTest(backend=backend):
                model = fit_backend(backend, x, y, np.ones(120), 12, [], cfg, 123, rounds_override=3)
                p = model.predict(x[:8])
                np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
                self.assertTrue((p[:, [1,2,3,5,6,7,8,10,11]] == 0).all())
                self.assertTrue((p[:, [0,4,9]] > 0).all())
        model = fit_backend('lightgbm', x, np.full(120, 9), np.ones(120), 12, [], cfg, 123, 3)
        self.assertTrue((model.predict(x[:2])[:, 9] == 1).all())


if __name__ == '__main__': unittest.main(verbosity=2)
