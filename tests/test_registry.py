import unittest
from jdet.config.registry import Registry

class TestRegistry(unittest.TestCase):

    def test(self):
        register = Registry()

        @register.register("C1")
        @register.register("C2")
        class C(object):
            def display(self):
                print("C",id(self))

        @register.register("foo1")
        def foo():
            print("foo")

        register.register("foo2",foo)

        register.C1().display()
        register["C2"]().display()
        
        register.foo1()
        register["foo2"]()


if __name__ == "__main__":
    unittest.main()
            
