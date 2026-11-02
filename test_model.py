import json
import unittest
from pathlib import Path
import numpy as np
import train_model as m

class Tests(unittest.TestCase):
    def test_ols_exact(self):
        rng=np.random.default_rng(9);X=rng.normal(size=(100,4));coef=np.array([2,3,-4,5,6]);y=np.c_[np.ones(100),X]@coef
        np.testing.assert_allclose(m.fit(X,y),coef,atol=1e-10)

    def test_heat_balance(self):
        self.assertAlmostEqual(m.next_T(22,300,0,1),22+60*80/(50*2000))
        self.assertGreater(m.next_T(25,300,1000,1),m.next_T(25,300,0,1))

    def test_sensor_safety(self):
        coef=np.array([1000,-8,-.12,-4,8])
        self.assertEqual(m.choose(float('nan'),0,40,0,coef),(0,0))
        self.assertEqual(m.choose(25,0,80,0,coef),(0,0))
        self.assertEqual(m.simulate(coef,'intelligent',60)['status'],'sensor_error')

    def test_export_and_control(self):
        r=json.loads((Path(__file__).parent/'report.json').read_text(encoding='utf-8'))
        train=np.genfromtxt(Path(__file__).parent/'dataset.csv',delimiter=',',names=True,encoding='utf-8-sig',dtype=None)
        train_groups=set(train['batch'][train['split']=='train']);test_groups=set(train['batch'][train['split']=='test'])
        self.assertFalse(train_groups&test_groups);self.assertEqual(len(train_groups),80);self.assertEqual(len(test_groups),20)
        y=np.array([x['mu_mPas'] for x in r['test']]);pred=np.array([x['predicted'] for x in r['test']]);actual=m.metrics(y,pred)
        for key,value in actual.items():self.assertAlmostEqual(value,r['metrics'][key])
        for mode,case in r['demo'].items():
            self.assertAlmostEqual(case['trace'][0]['T'],25)
            for x in case['trace']:
                self.assertTrue(20<=x['T']<=45);self.assertTrue(100<=x['n']<=500)
                self.assertTrue(0<=x['heater']<=1500)
            expected=sum((m.power(x['n'])+x['heater']/1000)*m.PARAMS['dt']/60 for x in case['trace'][:-1])
            self.assertAlmostEqual(expected,case['energy'])
            if case['success']:
                self.assertTrue(450<=case['mu_final']<=550);self.assertGreaterEqual(case['H_final'],95)

if __name__=='__main__':unittest.main()
