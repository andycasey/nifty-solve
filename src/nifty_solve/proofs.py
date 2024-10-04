
import numpy as np
import pylab as plt
from scipy.special import gamma
import finufft
np.random.seed(8675309) # MAGIC to enforce reproducibility

n = 23
T = 1.5 # range
deltaomega = 0.5 * np.pi / T # frequency spacing for Fourier series

# generate like a Ward
# ts = T * np.arange(0.5 / n, 1., 1. / n)

# generate like a Hogg (harder problem!)
# ts = np.sort(T * np.random.uniform(size=n))

# generate intermediate between Hogg and Ward
ts = (T * (np.arange(0.5 / n, 1., 1. / n)
      + (0.75 / n) * (np.random.uniform(size=n) - 0.5))) # magic 0.75
ti = np.linspace(-np.pi, np.pi, 2048)

P = 7

A_me = finufft.Plan(2, (P, ))
A_me.setpts(ts)

me_indices = (
    (P // 2, 1, 1j),
    (P // 2 - 1, -1j, 1),
    (P // 2 - 1, 1, -1j),
    #(1, 1j), # --> 0 always
    (P // 2 - 2, -1j, 1),
    (P // 2 - 2, 1, -1j), 
    (P // 2 - 3, -1j, 1),
    (P // 2 - 3, 1, -1j),
)

A = []
for i in range(len(me_indices)):
    #me_index, me_value, alt_me_value = me_indices[i]
    v = np.zeros(P).astype(np.complex128)
    v[i] = 1#j if i % 2 == 0 else 1
    A.append(A_me.execute(v))

A = np.array(A).T
fig, ax = plt.subplots(figsize=(4, 4))
for i in range(A.shape[1]):
    ax.plot(ts, np.sign(A[:, i]) * np.abs(A[:, i]), label=f"mode {i}")

n_modes = 7
finufft_kwargs = dict()
_f1_plan = finufft.Plan(1, (n_modes, ), **finufft_kwargs) # A.T
_f2_plan = finufft.Plan(2, (n_modes, ), **finufft_kwargs) # A
for plan in (_f1_plan, _f2_plan):
    plan.setpts(ts)

G = _f2_plan.execute
GT = _f1_plan.execute
GTD = lambda c: finufft.nufft1d1(ts, c, n_modes, n_trans=c.shape[0])

x = np.random.normal(0, 1, size=n_modes) + np.random.normal(0, 1, size=n_modes) * 1j

ivar = np.atleast_2d((1/np.random.normal(0, 1, size=n)**2).astype(np.complex128))

def sliding_windows(a, W):
    a = np.asarray(a)
    p = np.zeros(W-1,dtype=a.dtype)
    b = np.concatenate((p,a,p))
    s = b.strides[0]
    strided = np.lib.stride_tricks.as_strided
    return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))


rho = sliding_windows([0.2, 1, 0.7], n)[:, 1:-1].astype(np.complex128)
Cinv = rho * (np.sqrt(ivar.T) @ np.sqrt(ivar))

Y = A @ x

assert np.allclose(G(x), A @ x)
assert np.allclose(GT(Y), (A.T @ Y)[::-1])


assert np.allclose(
    np.array(list(map(GT, Cinv.T))),
    (A.T @ Cinv).T[:, ::-1]
)
assert np.allclose(
    finufft.nufft1d1(ts, Cinv, P),
    np.array(list(map(GT, Cinv)))
)

p = finufft.Plan(1, (n_modes, ), n_trans=n)
p.setpts(ts)

assert np.allclose(
    p.execute(Cinv),
    finufft.nufft1d1(ts, Cinv, P)
)

assert np.allclose(
    p.execute(Cinv.T).T,
    (A.T @ Cinv)[::-1, :]
)

def ATCinvA(Cinv):
    return finufft.nufft1d1(
        ts,
        finufft.nufft1d1(
            ts,
            Cinv,
            P,
            isign=-1
        ).T,
        P,
        isign=-1
    ).T



expected = A.T @ Cinv @ A
actual = finufft.nufft1d1(
    ts,
    finufft.nufft1d1(
        ts,
        Cinv,
        P,
    ).T,
    P
).T[::-1, ::-1]

assert np.allclose(expected, actual)

actual = finufft.nufft1d1(
    ts,
    finufft.nufft1d1(
        ts,
        Cinv,
        P,
        isign=-1
    ).T,
    P,
    isign=-1
).T

assert np.allclose(expected, actual)

p1 = finufft.Plan(1, (n_modes, ), n_trans=n, isign=-1)
p1.setpts(ts)

p2 = finufft.Plan(1, (n_modes, ), n_trans=n_modes, isign=-1)
p2.setpts(ts)

actual = p2.execute(p1.execute(Cinv).T).T

assert np.allclose(expected, actual)

expected = A.T @ Cinv @ Y

actual = finufft.nufft1d1(ts, Cinv, P, isign=-1).T @ Y

assert np.allclose(expected, actual)

