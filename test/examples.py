from jaxinterp.interpreter import interpret
import jax
import jax.numpy as jnp
from jax import lax

def f(xs):
    return jax.vmap(lambda x: x ** 2 + 1)(xs)

def g(xs, ys):
    zs = jax.vmap(lambda x, y: x ** y + y)(xs, ys)
    return {'xs': xs, 'ys': ys, 'zs': zs}

def h(zs):
    return {'xs': zs['xs'], 'ys': zs['ys']}

def i(w):
    return lax.fori_loop(0, w, lambda i, x: x + i, 0)

if __name__ == "__main__":
    print(interpret(f)(jnp.array([1.,2.,3.])))
    print(interpret(g)(jnp.array([5.,3.,2.]), jnp.array([1.,2.,3.])))
    print(interpret(h)({'xs': jnp.array([1.,3.]), 'ys': jnp.array([5., 9.])}))
    print(interpret(i)(jnp.array(11)))
