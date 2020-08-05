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


def j(xs):
    return lax.map(lambda x: x ** 2 + 1, xs)


def k(xs):
    return lax.cond(jnp.sum(xs >= 0), lambda _: xs, lambda _: jnp.zeros_like(xs), ())


def l(xs):
    return lax.scan(lambda c, a: (c + 1, a ** c), 1., xs, reverse=True)


if __name__ == "__main__":
    print(interpret(f, check_functions=True)(jnp.array([1., 2., 3.])))
    print(interpret(g, check_functions=True)(jnp.array([5., 3., 2.]), jnp.array([1., 2., 3.])))
    print(interpret(h, check_functions=True)({'xs': jnp.array([1., 3.]), 'ys': jnp.array([5., 9.])}))
    print(interpret(i, check_functions=True)(jnp.array(11)))
    print(interpret(j, check_functions=True)(jnp.array([1., 2., 3.])))
    print(interpret(k, check_functions=True)(jnp.array([-10, 0., 5.])))
    print(interpret(k, check_functions=True)(jnp.array([-10, -30., -5.])))
    print(interpret(jax.grad(lambda xs: jnp.sum(f(xs))),
                    check_functions=True)(jnp.array([1., 2., 3.])))
    print(interpret(l, check_functions=True)(jnp.array([8., 9., 10., 11.])))
    print(interpret(jax.grad(lambda xs: jnp.sum(l(xs)[1])),
                    check_functions=True)(jnp.array([8., 9., 10., 11.])))
