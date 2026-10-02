import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from future_split_guard import validate


class FutureSplitGuardTests(unittest.TestCase):
    def setUp(self):
        self.components = [{'sample_id': s, 'component_id': 'C1', 'task_area': 'knot'} for s in ['a','b']]
        self.policy = [{'component_id':'C1','force_future_development':'1'}]
        self.rows = [{'sample_id':s,'split':'development','task_area':'knot'} for s in ['a','b']]

    def test_safe_development(self):
        self.assertTrue(validate(self.rows,self.components,self.policy)['passed'])

    def test_historical_migration_rejected(self):
        for r in self.rows:
            r['split']='confirmation'
        self.assertFalse(validate(self.rows,self.components,self.policy)['passed'])

    def test_unexposed_component_still_cannot_be_split(self):
        self.policy[0]['force_future_development']='0'
        self.rows[1]['split']='final_holdout'
        self.assertEqual(validate(self.rows,self.components,self.policy)['violations'][0]['reason'],'component_split')

    def test_background_binding(self):
        bg=[{'uniprot_accession':'P12345','component_id':'C1','split':'confirmation'}]
        self.assertFalse(validate(self.rows,self.components,self.policy,bg,bg)['passed'])

    def test_false_background_component_fails(self):
        bg=[{'uniprot_accession':'P12345','component_id':'C2','split':'development'}]
        audited=[{'uniprot_accession':'P12345','component_id':'C1'}]
        with self.assertRaises(ValueError):
            validate(self.rows,self.components,self.policy,bg,audited)

    def test_label_only_is_not_global_approval(self):
        self.assertFalse(validate(self.rows,self.components,self.policy)['ready_for_global_split'])

    def test_false_native_component_fails(self):
        self.rows[0]['component_id']='forged'
        with self.assertRaises(ValueError):
            validate(self.rows,self.components,self.policy)

    def test_partial_global_map_fails(self):
        self.policy.append({'component_id':'missing','force_future_development':'0'})
        with self.assertRaises(ValueError):
            validate(self.rows,self.components,self.policy,[],[])

    def test_changed_universe_fails(self):
        with self.assertRaises(ValueError):
            validate(self.rows[:-1],self.components,self.policy)

    def test_duplicate_identity_fails(self):
        with self.assertRaises(ValueError):
            validate(self.rows+[self.rows[0]],self.components,self.policy)


if __name__ == '__main__':
    unittest.main()
