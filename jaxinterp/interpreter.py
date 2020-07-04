# Code is based on https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html

import numpy as np
from functools import wraps

from jax import api_util
from jax import core
from jax import lax
from jax import xla
from jax import pxla
from jax import linear_util as lu
from jax import tree_util
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax.util import safe_map

def _make_jaxpr_with_consts(fun):
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
    jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals) 
    return jaxpr, consts, (in_tree, out_tree())
  return jaxpr_const_maker

def _interpret_jaxpr(jaxpr, consts, *args):
  # Mapping from variable -> value
  env = {}
  
  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  write(core.unitvar, core.unit)
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Loop through equations and evaluate primitives using `bind`
  for eqn in jaxpr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)  
    if eqn.primitive is xla.xla_call_p:
      _interpret_jaxpr(eqn.params['call_jaxpr'], (), *invals)
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

def interpret(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    jaxpr, consts, (_, out_tree) = _make_jaxpr_with_consts(fun)(*args, **kwargs)
    args = [leaf for arg in args for leaf in tree_util.tree_leaves(arg)]
    out = _interpret_jaxpr(jaxpr, consts, *args)
    return tree_util.build_tree(out_tree, out)
  return wrapped
