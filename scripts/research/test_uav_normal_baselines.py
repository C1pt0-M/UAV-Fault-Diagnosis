"""Checks for evaluation leakage and alarm-event semantics."""
import unittest
import numpy as np
from run_uav_normal_baselines import contexts, evaluate_alarm, load_case


class ProtocolChecks(unittest.TestCase):
    def test_history_excludes_target_and_future(self):
        x=np.arange(220*26,dtype=np.float32).reshape(220,26)
        before,target=contexts(x,np.array([150]))
        changed=x.copy();changed[150:]+=10000
        after,new_target=contexts(changed,np.array([150]))
        np.testing.assert_array_equal(before,after)
        self.assertFalse(np.array_equal(target,new_target))
        self.assertEqual(before.shape,(1,30,26))
        self.assertEqual(before[0,0,14],0)
        self.assertEqual(target[0,14],x[150,14]-x[0,14])

    def test_prefault_alarm_does_not_count_as_detection(self):
        t=np.arange(10)*.1;truth=t>=.5
        r=evaluate_alarm(t,truth,[(.5,1.)],np.ones(10)*2,1.)
        self.assertEqual(r['false_alarm_events'],1)
        self.assertEqual(r['detected_events'],0)
        self.assertEqual(r['prefault_alarm_case'],1)

    def test_alarm_requires_three_decisions_and_records_delay(self):
        t=np.arange(10)*.1;truth=t>=.5
        r=evaluate_alarm(t,truth,[(.5,1.)],np.where(truth,2.,0.),1.)
        self.assertEqual(r['detected_events'],1)
        self.assertEqual(r['false_alarm_events'],0)
        self.assertAlmostEqual(r['detected_delays_s'][0],.2)

    def test_short_spike_does_not_trigger(self):
        t=np.arange(10)*.1
        scores=np.array([0,2,2,0,0,0,0,0,0,0])
        r=evaluate_alarm(t,np.zeros(10,dtype=bool),[],scores,1.)
        self.assertEqual(r['false_alarm_events'],0)

    def test_external_domain_rejected_before_read(self):
        with self.assertRaises(AssertionError):
            load_case({'domain':'REAL','split':'external_test'},[])


if __name__=='__main__': unittest.main()
