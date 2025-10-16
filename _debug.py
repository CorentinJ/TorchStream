from z3 import *

N = 10
bv = BitVec("bv", N)
lp = Int("lp")

s = Solver()


s.add(lp >= 0, lp < 5)
s.add(ULE(BitVecVal(1 << (N - 1), N), bv))

# s.add(bv & BitVecVal(0b0010101010, N) != 0)

# True kernel is 10101 with left pad 2

in_nan_idx = 2
in_size = 5
out_nan_idx = (0, 2)
out_size = 3






s.check()

m = s.model()
bv_int = m[bv].as_long()
bv_str = format(bv_int, "0{}b".format(N))
print(bv_str)
