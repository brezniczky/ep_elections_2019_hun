"""
A memoize style decorator for iterative REPL work, primarily aimed at

The point is a currently "imperfect, but good start" workaround for disk cache's
memoize that clears contents when the wrapped function gets redefined with a
different source code.

(Imperfect: does not detect changes in the code of functions that are called by
the wrapped one.)
"""
from diskcache import Cache, Index
import inspect
from functools import wraps


"""
I should consider applying a singleton pattern. This _cache gets recreated on
each import.

# Singleton/ClassVariableSingleton.py
class SingleTone(object):
    __instance = None
    def __new__(cls, val):
        if SingleTone.__instance is None:
            SingleTone.__instance = object.__new__(cls)
        SingleTone.__instance.val = val
        return SingleTone.__instance
"""
_caches = {}

_caches_source_revision_cache = Index("_caches_source_revision_cache")


def _get_source_revision_of_cache(fun_impl_key):
    """ Raises KeyError if not found. """
    return _caches_source_revision_cache[fun_impl_key]


def _set_source_revision_of_cache(fun_impl_key, source_version):
    _caches_source_revision_cache[fun_impl_key] = source_version


def analytic_memoize(f):
    """ A memoize style disk cache that clears itself when the memoized function
        changes.

        Beware it is blind to dependencies (changes in called functions) at this
        point!
    """

    fun_impl_key = f.__qualname__
    """ Unfortunately a lot of things just change on each load:
        f.__code__.co_code.__hash__()
        f.__code__.__hash__()
        f.__hash__()

        Not sure what else I'd tried but I arrived at this in the end.
    """
    source_revision = inspect.getsource(f)

    try:
        fun_cache = _caches[fun_impl_key]

        if _get_source_revision_of_cache(fun_impl_key) != source_revision:
            fun_cache.clear()
            _set_source_revision_of_cache(fun_impl_key, source_revision)
    except KeyError:
        fun_cache = Cache()
        _set_source_revision_of_cache(fun_impl_key, source_revision)
        _caches[fun_impl_key] = fun_cache

    @fun_cache.memoize()
    @wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapped


if __name__ == "__main__":
    """ bit of demo """


    @analytic_memoize
    def memoized2(x):
        print("#1 is calculating for x =", x)
        return x * x


    y1 = memoized2(5)
    y1 = memoized2(5)


    @analytic_memoize
    def memoized2(x):
        print("#2 is calculating for x =", x)
        return x ** 2 + 1


    y2 = memoized2(5)
    y2 = memoized2(5)

    assert y1 != y2

""" Might better flag this up with the DiskCache maintainers, if it's just not
    me being unable to quickly find out how to achieve this with their stuff.


    cache = Cache()


    @cache.memoize()
    def memoized():
        return 5

    x1 = memoized()


    @cache.memoize()
    def memoized():
        return 10

    x2 = memoized()


    assert x1 != x2


    # a mentioned "housekeeping" item
    #
    # https://pypi.org/project/percache/
    #
    # is to clear out the cache on function changes
    #
    # I don't want to have to use something complex like AirFlow for this
    #

"""
