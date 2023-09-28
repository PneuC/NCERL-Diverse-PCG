import time


def record_time(foo):
    def __inner(*args, **kwargs):
        start_time = time.time()
        res = foo(*args, **kwargs)
        return res, time.time() - start_time
    return __inner

