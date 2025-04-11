
from mains import build_net
from bayesnet_SOLN import *
import unittest

class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        self.alarm = build_net('alarm.txt')

    def test_get_joint_prob(self):
        # P(b, ~e, a, j, ~m) = 0.000253
        query = {'burglary': 't',
                 'earthquake': 'f',
                 'alarm': 't',
                 'johncalls': 't',
                 'marycalls': 'f'
                 }
        self.assertLess(abs(self.alarm.get_joint_prob(query) - 0.000253), 1E-6)

        # P(~b, ~e, ~a, j, ~m) = 0.0493
        query = {'burglary': 'f',
            'earthquake': 'f',
            'alarm': 'f',
            'johncalls': 't',
            'marycalls': 'f'
            }
        self.assertLess(abs(self.alarm.get_joint_prob(query) - 0.0493), 1E-5)


        # P(b, e, a, ~j, ~m) = 0.000000057
        query = {'burglary': 't',
            'earthquake': 't',
            'alarm': 't',
            'johncalls': 'f',
            'marycalls': 'f'
            }
        self.assertLess(abs(self.alarm.get_joint_prob(query) - 0.000000057), 1E-11)

    def test_get_marginal_prob(self):
        # P(b) = 0.001
        query = {'burglary': 't'}
        self.assertLess(abs(self.alarm.get_marginal_prob(query) - 0.001), 1E-9)

        # P(~b, e, a) = 0.999 * 0.002 * 0.29 = 0.00057942
        query = {'burglary': 'f',
                 'earthquake': 't',
                 'alarm': 't'
                }
        self.assertLess(abs(self.alarm.get_marginal_prob(query) - 0.00057942), 1E-9)

        # P(a) = P(~b, ~e, a) +         0.000997002  
        #        P(~b, e, a) +          0.00057942
        #        P(b, ~e, a) +          0.00093812
        #        P(b, e, a)             0.0000019
        #      = 0.002516442
        query = {'alarm': 't'}
        self.assertLess(abs(self.alarm.get_marginal_prob(query) - 0.002516442), 1E-10)

        # P(a, ~j) = 0.0002516442
        query = {'alarm': 't',
                 'johncalls': 'f'
                 }
        self.assertLess(abs(self.alarm.get_marginal_prob(query) - 0.0002516442), 1E-11)

        # P(a, j, ~m) = 0.00067943934
        query = {'alarm': 't',
                 'johncalls': 't',
                 'marycalls': 'f'
                 }
        self.assertLess(abs(self.alarm.get_marginal_prob(query) - 0.00067943934), 1E-11)


    def test_get_conditional_prob(self):

        # P(m | a) = 0.7
        query = {'marycalls': 't'}
        evidence = {'alarm': 't'}
        self.assertLess(abs(self.alarm.get_conditional_prob(query, evidence) - 0.7), 1E-7)

        # P(a | e) = P(a | e, b) P(b) + P(a | e, ~b) P(~b)
        #          = 0.95 * 0.001 + 0.29 * 0.999
        #          = 0.29066
        query = {'alarm': 't'}
        evidence = {'earthquake': 't'}
        self.assertLess(abs(self.alarm.get_conditional_prob(query, evidence) - 0.29066), 1E-7)

        # P(~e, b | a) = 0.372796194
        query = {'earthquake': 'f',
                 'burglary': 't'}
        evidence = {'alarm': 't'}
        self.assertLess(abs(self.alarm.get_conditional_prob(query, evidence) - 0.372796194), 1E-7)

        # SCRATCH NOTES for the query below
        # These are for ~b; the ones for b are in lecture slides
        # e,a:    0.29 * (0.9 * 0.7) = 0.1827
        # e,~a:  0.71 * (0.05 * 0.01) = 0.000355
        # ~e,a:  0.001 * (0.9 * 0.7) = 0.00063
        # ~e,~a: 0.999 * (0.05 * 0.01) = 0.0004995
        # 0.999 * (0.002 * (0.1827 + 0.000355) + 0.998 * (0.00063 + 0.0004995))
        # = 0.001491857649 = P(~b, j, m)
        #
        # P(b, j, m) = 0.00059224259
        #
        # P(B | j,m) = [0.715829, 0.284171] for [b=False, b=True]

        query = {'burglary': 't'}
        evidence = {'johncalls': 't',
                    'marycalls': 't'
        }
        self.assertLess(abs(self.alarm.get_conditional_prob(query, evidence) - 0.284171), 1E-5)


    def test_get_conditional_prob_distribution(self):

        # P(M | a) = [0.7, 0.3]
        query = ['marycalls']
        evidence = {'alarm': 't'}
        res = self.alarm.get_conditional_prob_distribution(query, evidence)
        self.assertLess(abs(res[('t',)] - 0.7), 1E-7)
        self.assertLess(abs(res[('f',)] - 0.3), 1E-7)

        # P(a | e) = P(a | e, b) P(b) + P(a | e, ~b) P(~b)
        #          = 0.95 * 0.001 + 0.29 * 0.999
        #          = 0.29066
        query = ['alarm']
        evidence = {'earthquake': 't'}
        res = self.alarm.get_conditional_prob_distribution(query, evidence)
        self.assertLess(abs(res[('t',)] - 0.29066), 1E-7)
        self.assertLess(abs(res[('f',)] - (1.0 - 0.29066)), 1E-7)

        # P(~e, ~b | a) = 0.396195104
        # P(~e, b | a) = 0.372796194
        # P(e, ~b | a) = 0.230253667
        # P(e, b | a) = 0.00075503429
        query = ['earthquake', 'burglary']
        evidence = {'alarm': 't'}
        res = self.alarm.get_conditional_prob_distribution(query, evidence)
        self.assertLess(abs(res[('f','f')] - 0.396195104), 1E-9)
        self.assertLess(abs(res[('f','t')] - 0.372796194), 1E-9)
        self.assertLess(abs(res[('t','f')] - 0.230253667), 1E-9)
        self.assertLess(abs(res[('t','t')] - 0.00075503429), 1E-9)

        # P(B | j,m) = [0.715829, 0.284171] for [b=False, b=True]
        query = ['burglary']
        evidence = {'johncalls': 't',
                    'marycalls': 't'
        }
        res = self.alarm.get_conditional_prob_distribution(query, evidence)
        self.assertLess(abs(res[('t',)] - 0.284171), 1E-6)
        self.assertLess(abs(res[('f',)] - (1.0-0.284171)), 1E-6)
        

    def test_get_markov_blanket(self):
        res = self.alarm.get_markov_blanket('earthquake')
        self.assertEqual(len(res), 2)
        self.assertTrue('burglary' in res)
        self.assertTrue('alarm' in res)

        res = self.alarm.get_markov_blanket('burglary')
        self.assertEqual(len(res), 2)
        self.assertTrue('earthquake' in res)
        self.assertTrue('alarm' in res)

        res = self.alarm.get_markov_blanket('alarm')
        self.assertEqual(len(res), 4)
        self.assertTrue('earthquake' in res)
        self.assertTrue('burglary' in res)
        self.assertTrue('johncalls' in res)
        self.assertTrue('marycalls' in res)

        res = self.alarm.get_markov_blanket('johncalls')
        self.assertEqual(len(res), 1)
        self.assertTrue('alarm' in res)

        res = self.alarm.get_markov_blanket('marycalls')
        self.assertEqual(len(res), 1)
        self.assertTrue('alarm' in res)

if __name__ == '__main__':
    unittest.main()