#!/usr/bin/env python
# This file is carbonara library from Gnocchi (https://github.com/gnocchixyz/gnocchi)

import six
import time
import numpy
import re
import struct
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def showcuda():
    a = numpy.random.randn(4,4)

    a = a.astype(numpy.float32)

    a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
        __global__ void doublify(float *a)
        {
          int idx = threadIdx.x + threadIdx.y*4;
          a[idx] *= 2;
        }
        """)

    func = mod.get_function("doublify")
    func(a_gpu, block=(4,4,1))

    a_doubled = numpy.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print "original array:"
    print a
    print "doubled with kernel:"
    print a_doubled

    # alternate kernel invocation ---------------------------------------------

    func(cuda.InOut(a), block=(4, 4, 1))
    print "doubled with InOut:"
    print a

    # part 2 ------------------------------------------------------------------

    import pycuda.gpuarray as gpuarray
    a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
    a_doubled = (2*a_gpu).get()

    print "original array:"
    print a_gpu
    print "doubled with gpuarray:"
    print a_doubled


UNIX_UNIVERSAL_START64 = numpy.datetime64("1970", 'ns')
ONE_SECOND = numpy.timedelta64(1, 's')


class BeforeEpochError(Exception):
    """Error raised when a timestamp before Epoch is used."""

    def __init__(self, timestamp):
        self.timestamp = timestamp
        super(BeforeEpochError, self).__init__(
            "%s is before Epoch" % timestamp)


def datetime64_to_epoch(dt):
    return (dt - UNIX_UNIVERSAL_START64) / ONE_SECOND


def round_timestamp(ts, freq):
    return UNIX_UNIVERSAL_START64 + numpy.floor(
        (ts - UNIX_UNIVERSAL_START64) / freq) * freq


def make_timeseries(timestamps, values):
    TIMESERIES_ARRAY_DTYPE = [('timestamps', '<datetime64[ns]'),
                              ('values', '<d')]
    l = len(timestamps)
    if l != len(values):
        raise ValueError("Timestamps and values must have the same length")
    arr = numpy.zeros(l, dtype=TIMESERIES_ARRAY_DTYPE)
    arr['timestamps'] = timestamps
    arr['values'] = values
    return arr


class TimeSerie(object):
    def __init__(self, ts=None):
        if ts is None:
            ts = make_timeseries([], [])
        self.ts = ts

    def __eq__(self, other):
        return (isinstance(other, TimeSerie) and
                numpy.all(self.ts == other.ts))

    @property
    def timestamps(self):
        return self.ts['timestamps']

    @property
    def values(self):
        return self.ts['values']

    @property
    def first(self):
        try:
            return self.timestamps[0]
        except IndexError:
            return

    @property
    def last(self):
        try:
            return self.timestamps[-1]
        except IndexError:
            return


class AggregatedTimeSerie(TimeSerie):

    _AGG_METHOD_PCT_RE = re.compile(r"([1-9][0-9]?)pct")

    PADDED_SERIAL_LEN = struct.calcsize("<?d")
    COMPRESSED_SERIAL_LEN = struct.calcsize("<Hd")
    COMPRESSED_TIMESPAMP_LEN = struct.calcsize("<H")

    def __init__(self, sampling, aggregation_method, ts=None):
        super(AggregatedTimeSerie, self).__init__(ts)
        self.sampling = sampling
        self.aggregation_method = aggregation_method

    def resample(self, sampling):
        return AggregatedTimeSerie.from_grouped_serie(
            self.group_serie(sampling), sampling, self.aggregation_method)

    @classmethod
    def from_data(cls, sampling, aggregation_method, timestamps,
                  values):
        return cls(sampling=sampling,
                   aggregation_method=aggregation_method,
                   ts=make_timeseries(timestamps, values))

    @staticmethod
    def _get_agg_method(aggregation_method):
        q = None
        m = AggregatedTimeSerie._AGG_METHOD_PCT_RE.match(aggregation_method)
        if m:
            q = float(m.group(1))
            aggregation_method_func_name = 'quantile'
        else:
            if not hasattr(GroupedTimeSeries, aggregation_method):
                raise "UnknownAggregationMethod(aggregation_method)"
            aggregation_method_func_name = aggregation_method
        return aggregation_method_func_name, q

    @classmethod
    def from_grouped_serie(cls, grouped_serie, sampling, aggregation_method):
        agg_name, q = cls._get_agg_method(aggregation_method)
        return cls(sampling, aggregation_method,
                   ts=cls._resample_grouped(grouped_serie, agg_name,
                                            q))

    def group_serie(self, granularity, start=None):
        # NOTE(jd) Our whole serialization system is based on Epoch, and we
        # store unsigned integer, so we can't store anything before Epoch.
        # Sorry!
        if len(self.ts) != 0 and self.first < UNIX_UNIVERSAL_START64:
            raise BeforeEpochError(self.first)

        return GroupedTimeSeries(self.ts, granularity, start)

    @staticmethod
    def _resample_grouped(grouped_serie, agg_name, q=None):
        agg_func = getattr(grouped_serie, agg_name)
        return agg_func(q) if agg_name == 'quantile' else agg_func()

    @classmethod
    def benchmark(cls):
        """Run a speed benchmark!"""

        POINTS_PER_SPLIT = 3600
        points = POINTS_PER_SPLIT * 1000
        points = 2 * 1024 * 6
        points = 1024*1024*6
        points = int(points)
        sampling = numpy.timedelta64(5, 's')
        resample = numpy.timedelta64(35, 's')

        now = numpy.datetime64("2015-04-03 23:11")
        timestamps = numpy.sort(numpy.array(
            [now + i * sampling
             for i in six.moves.range(points)]))

        for title, values in [
                ("Simple continuous range", six.moves.range(points)),
                ("Simple continuous range", [0] * (points)),
        ]:
            def per_sec(t1, t0):
                return 1 / ((t1 - t0) / serialize_times)

            print(title)
            serialize_times = 50
            ts = cls.from_data(sampling, 'mean', timestamps, values)
            pts = ts.ts.copy()

            for agg in ['sum', 'inhere']:
                serialize_times = 10
                ts = cls(ts=pts, sampling=sampling,
                         aggregation_method=agg)
                t0 = time.time()
                for i in six.moves.range(serialize_times):
                    ts.resample(resample)
                t1 = time.time()
                print("  resample(%s) speed: %.2f Hz"
                      % (agg, per_sec(t1, t0)))


class GroupedTimeSeries(object):
    def __init__(self, ts, granularity, start=None):
        # NOTE(sileht): The whole class assumes ts is ordered and don't have
        # duplicate timestamps, it uses numpy.unique that sorted list, but
        # we always assume the orderd to be the same as the input.
        self.granularity = granularity
        self.start = start
        if start is None:
            self._ts = ts
            self._ts_for_derive = ts
        else:
            self._ts = ts[numpy.searchsorted(ts['timestamps'], start):]
            start_derive = start - granularity
            self._ts_for_derive = ts[
                numpy.searchsorted(ts['timestamps'], start_derive):
            ]

        self.indexes = round_timestamp(self._ts['timestamps'], granularity)
        self.tstamps, self.counts = numpy.unique(self.indexes,
                                                 return_counts=True)
        self.a = self._ts['values']
        self.a = self.a.copy(order='C').astype(numpy.float32)
        self.a_gpu = cuda.mem_alloc(self.a.size * self.a.dtype.itemsize)

    def sum(self):
        summod = SourceModule("""
            __global__ void v1(float *a, int *i) {
              int perthread=i[0];
              int counter = i[0]-1;
              int col = blockIdx.x * blockDim.x + threadIdx.x;
              int row = blockIdx.y * blockDim.y + threadIdx.y;
              int index = col * perthread + row;
              for(;counter;counter--)
                  atomicAdd(a+index,  a[index+counter]);
            }

            __global__ void v2(float *a, int *i) {
              int perthread=i[0];
              int counter = i[0]-1;
              int blklimit=1024;
              int blk=0;
              for (blk=0;blk<blklimit;blk++) {
                  int counter = i[0]-1;
                  int col = blk * blockDim.x + threadIdx.x;
                  int row = blockIdx.y * blockDim.y + threadIdx.y;
                  int index = col * perthread + row;
                  //a[index] = 0;
                  for(;counter;counter--)
                      atomicAdd(a+index,  a[index+counter]);
                      //a[index+counter] = 0;
                      //a[index] += a[index+counter];
              }
            }

            """)

#        a = numpy.random.randn(4, 4)
#        self.a = a.astype(numpy.float32)
#        self.a = self.a.copy(order='C').astype(numpy.float32)
#        self.a_gpu = cuda.mem_alloc(self.a.size * self.a.dtype.itemsize)
#        t0 = time.time()
        cuda.memcpy_htod(self.a_gpu, self.a)
        func = summod.get_function("v1")
        func(self.a_gpu, cuda.In(numpy.array([6])), block=(1024, 1, 1), grid=(1024, 1))
        a_doubled = numpy.empty_like(self.a)
        cuda.memcpy_dtoh(a_doubled, self.a_gpu)
#        t1 = time.time()
#        print("%0.7f, %s" % (t1-t0, numpy.all(a_doubled == 0)))
#        print numpy.all(a_doubled == 0) 

#        t0 = time.time()
#        cuda.memcpy_htod(self.a_gpu, self.a)
#        func = summod.get_function("v2")
#        func(self.a_gpu, cuda.In(numpy.array([6])), block=(1024, 1, 1))
#        a_doubled = numpy.empty_like(self.a)
#        cuda.memcpy_dtoh(a_doubled, self.a_gpu)
#        t1 = time.time()
#        print("%0.7f, %s" % (t1-t0, numpy.all(a_doubled == 0)))

#        print "original array:"
#        print self.a
#        print "doubled with kernel:"
       #a_gpu.free()

    def inhere(self):
        t0 = time.time()
        dat = make_timeseries(self.tstamps, numpy.bincount(
            numpy.repeat(numpy.arange(self.counts.size), self.counts),
            weights=self._ts['values']))
        t1 = time.time()
        print("%0.7f" % (t1-t0))
        return dat
AggregatedTimeSerie.benchmark()
