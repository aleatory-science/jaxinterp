# Code is based on https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html and
# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
from itertools import repeat

import numpy as np
from functools import wraps

from jax import api_util
from jax import core
from jax import lax
from jax import xla
from jax import pxla
from jax import linear_util as lu
from jax import tree_util
import jax.numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax.util import safe_map, split_list


def _make_jaxpr_with_consts(fun, stage_out):
    def pv_like(x):
        # ShapedArrays are abstract values that carry around
        # shape and dtype information
        aval = ShapedArray(np.shape(x), np.result_type(x))
        return pe.PartialVal.unknown(aval)

    @wraps(fun)
    def jaxpr_const_maker(*args, **kwargs):
        # Set up fun for transformation
        wrapped = lu.wrap_init(fun)
        # Flatten input args
        jax_args, in_tree = tree_util.tree_flatten((args, kwargs))
        # Transform fun to accept flat args
        # and return a flat list result
        jaxtree_fun, out_tree = api_util.flatten_fun(wrapped, in_tree)
        # Abstract and partial-val's flat args
        pvals = safe_map(pv_like, jax_args)
        # Trace function into Jaxpr
        jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals, stage_out=stage_out)
        return jaxpr, consts, (in_tree, out_tree())

    return jaxpr_const_maker


def _reverse(cval):
    if cval is core.unit:
        return core.unit
    elif cval is list:
        return list(reversed(cval))
    else:
        return cval[::-1]


def _stack(aval, cval, reverse=False):
    if reverse:
        cval = _reverse(cval)
    if aval.aval is core.abstract_unit:
        return core.unit
    else:
        return jnp.stack(cval, 0)


def _zip(xs):
    length = max(len(x) if not x is core.unit else 0 for x in xs)
    return zip(*[x if not x is core.unit else repeat(core.unit, length) for x in xs])


def _interpret_jaxpr(jaxpr, consts, *args, check_functions=False):
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    def go_while(cond, body, *vals):
        while _interpret_jaxpr(cond, (), *vals)[0]:
            vals = _interpret_jaxpr(body, (), *vals)
        return vals

    def go_scan(body, length, xs, init, consts, reverse):
        num_carry = len(init)
        if xs is None:
            xs = [None] * length
        if reverse:
            xs = list(map(_reverse, xs))
        carry = init
        ys = []
        zxs = _zip(xs)
        for x in zxs:
            res = _interpret_jaxpr(body, (), *consts, *carry, *x)
            carry, y = split_list(res, [num_carry])
            ys.append(y)
        _, yavals = split_list(body.outvars, [num_carry])
        ys = list(map(lambda *x: _stack(*x, reverse), yavals, zip(*ys)))
        return [*carry, *ys]

    def go_cond(branches, index, *vals):
        return _interpret_jaxpr(branches[index], (), *vals)

    # Bind args and consts to environment
    write(core.unitvar, core.unit)
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)
        if eqn.primitive is xla.xla_call_p:
            outvals = _interpret_jaxpr(eqn.params['call_jaxpr'], (), *invals)
            _check_prim_out(check_functions, eqn, invals, outvals)
        elif eqn.primitive is lax.while_p:
            outvals = go_while(eqn.params['cond_jaxpr'].jaxpr, eqn.params['body_jaxpr'].jaxpr, *invals)
            _check_prim_out(check_functions, eqn, invals, outvals)
        elif eqn.primitive is lax.scan_p:
            consts, carry_init, rest = split_list(invals, [eqn.params['num_consts'], eqn.params['num_carry']])
            outvals = go_scan(eqn.params['jaxpr'].jaxpr, eqn.params['length'], rest, carry_init, consts,
                              eqn.params['reverse'])
            _check_prim_out(check_functions, eqn, invals, outvals)
        elif eqn.primitive is lax.cond_p:
            outvals = go_cond(safe_map(lambda x: x.jaxpr, eqn.params['branches']), *invals)
            _check_prim_out(check_functions, eqn, invals, outvals)
        else:
            # `bind` is how a primitive is called
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, outvals)
        # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)


def _check_prim_out(check_functions, eqn, invals, outvals):
    if check_functions:
        checkoutvals = eqn.primitive.bind(*invals, **eqn.params)
        for ov, cov in zip(outvals, checkoutvals):
            assert np.all(ov == cov)


def interpret(fun, stage_out=False, check_functions=False):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr, consts, (_, out_tree) = _make_jaxpr_with_consts(fun, stage_out)(*args, **kwargs)
        args = [leaf for arg in args for leaf in tree_util.tree_leaves(arg)]
        out = _interpret_jaxpr(jaxpr, consts, *args, check_functions=check_functions)
        return tree_util.tree_unflatten(out_tree, out)

    return wrapped
