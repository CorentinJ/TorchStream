from z3 import *

x = BitVecVal(0b11010111, 8)
d = BitVec("d", 8)
b = Extract(0, 0, d)

s = Solver()
s.add(d == x)
s.check()
m = s.model()

print(format(m.evaluate(b).as_long(), "0{}b".format(1)))
