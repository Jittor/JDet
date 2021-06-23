import unittest
import jdet
from jdet.config import init_cfg, get_cfg

class TestConfig(unittest.TestCase):
    # test basic
    def test1(self):
        init_cfg("1.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {'a':1,'b':{'a':2,'b':3}}
        assert(ans==cfg.dump())
        assert(cfg.a==1)
        assert(cfg.b=={'a':2,'b':3})
        assert(cfg.b.a==2)
        assert(cfg.b.b==3)

    # test _base_
    def test2(self):
        init_cfg("2.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {'a':1, 'b':{'a':5, 'b':3, 'c':6}}
        assert(ans==cfg.dump())

    # test _cover_
    def test3(self):
        init_cfg("3.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {'a':1, 'b':{'a':5, 'c':6}}
        assert(ans==cfg.dump())

    # test root _cover_
    def test4(self):
        init_cfg("4.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {'b':{'a':5, 'c':6, 'd':{}}, 'c':{'d':[12,34]}}
        assert(ans==cfg.dump())

    # test config in subdir
    def test5(self):
        init_cfg("5/5.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {'a':1, 'b':{'a':5, 'b':3, 'c':6}, 'd':1}
        assert(ans==cfg.dump())

    # test tree config
    def test6(self):
        init_cfg("6.yaml")
        cfg = get_cfg()
        print(cfg)
        ans = {
            'a':1,
            'b':{
                'a':2,
                'b':8,
                'c':{'a':0}
            },
            'c':0
        }
        assert(ans==cfg.dump())
    
    # test .py config
    def test7(self):
        init_cfg("7.py")
        cfg = get_cfg()
        print(cfg)
        ans = {
            'a':1,
            'b':2
        }
        assert(ans==cfg.dump())

    # test .py _base_
    def test8(self):
        init_cfg("8.py")
        cfg = get_cfg()
        print(cfg)
        ans = {
            'a':3,
            'c':2
        }
        assert(ans==cfg.dump())

    # test do some computation in .py
    def test9(self):
        init_cfg("9.py")
        cfg = get_cfg()
        print(cfg)
        ans = {
            'gpus':[1,2,3,4],
            'n_gpus':4,
            'id':1,
            'root':'1',
            'path':'1/output'
        }
        assert(ans==cfg.dump())

    # test mixture case
    def test10(self):
        init_cfg("10/10.py")
        cfg = get_cfg()
        print(cfg)
        ans = {
            'b':2,
            'c':3,
            'd': {
                'a':50,
                'b':20,
                'c':30
            },
            'e': {
                'a':0,
                'c':1,
            },
            'x':2,
            'y':1,
            'z':6
        }
        assert(ans==cfg.dump())

if __name__ == "__main__":
    unittest.main()