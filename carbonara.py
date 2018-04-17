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

#cuda.initialize_profiler()

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
            if not hasattr(GroupedGpuBasedTimeSeries, aggregation_method):
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

        return GroupedGpuBasedTimeSeries(self.ts, granularity, start)

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
        points = 512*1024*6
        points = 1024*1024*7
        points = int(points)
        sampling = numpy.timedelta64(5, 's')
        resample = numpy.timedelta64(35, 's')

        now = numpy.datetime64("2015-04-03 23:11")
        timestamps = numpy.sort(numpy.array( [now + i * sampling for i in six.moves.range(points)]))

        for title, values in [
                ("Simple continuous range", six.moves.range(points)),
                #("Simple continuous range", [0] * (points)),
        ]:
            def per_sec(t1, t0):
                return 1 / ((t1 - t0) / serialize_times)

            print(title)
            serialize_times = 50
            ts = cls.from_data(sampling, 'mean', timestamps, values)
            pts = ts.ts.copy()

            for agg in ['cpu_sum', 'gpu_sum']:
            #for agg in ['sum']:
                serialize_times = 3 if agg.endswith('pct') else 1
                ts = cls(ts=pts, sampling=sampling,
                         aggregation_method=agg)
                t0 = time.time()
                for i in six.moves.range(serialize_times):
                    ts.resample(resample)
                t1 = time.time()
                print("  resample(%s) speed: %.2f Hz"
                      % (agg, per_sec(t1, t0)))


class GroupedGpuBasedTimeSeries(object):
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
        self.junk = cuda.mem_alloc(self.a.dtype.itemsize)

    def gpu_sum(self):
        summod = SourceModule("""
            /*
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
              int blklimit=1024;
              int blk=1;
              int row, col, index;
              int counter = 0;
              for (blk=1;blk<blklimit;blk++) {
                  counter = perthread-1;
                  col = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x*blk*perthread;
                  row = blockIdx.y * blockDim.y + threadIdx.y;
                  int tid = threadIdx.x;
                  int x = 0;
                  for(;counter;counter--){
                      x += a[index + counter];
                  }
                  a[index] = x;
              }
            }
            */

            __global__ void v3(float *a, int *i) {
              int perthread=i[0];
              int row, col, index, counter;
              extern __shared__ float inter[];
              col = blockIdx.x * blockDim.x + threadIdx.x;
              row = blockIdx.y * blockDim.y + threadIdx.y;
              int tid = threadIdx.x;
              for(counter=0;counter<perthread;counter++){
                  index = col + blockDim.x*counter + row;
                  inter[blockDim.x * counter + tid] = a[index];
              }

              __syncthreads();
              int x = 0;
              for(counter=0;counter<perthread;counter++){
                  x += inter[perthread*tid + counter];
              }
              a[index] = x;
            }
            """, options=['--generate-line-info'], keep=True)

        self.reduce_by = int(self.granularity / numpy.timedelta64(5, 's'))
        func = summod.get_function("v3")
        #func(self.a_gpu, cuda.In(numpy.array([6])), block=(1024, 1, 1), grid=(512, 1))
        #func(self.a_gpu, cuda.In(numpy.array([6])), block=(1024, 1, 1), grid=(1024, 1))
        t0 = time.time()
        cuda.start_profiler()
        self.a_gpu = cuda.mem_alloc(self.a.size * self.a.dtype.itemsize)
        cuda.memcpy_htod(self.a_gpu, self.a)
        #func(self.a_gpu, cuda.In(numpy.array([6])), block=(1024, 1, 1))
        gridsize = 1024
        func(self.a_gpu, cuda.In(numpy.array(self.reduce_by)),
                block=(1024, 1, 1), grid=(gridsize, 1),
                shared=1024*self.reduce_by*self.a.dtype.itemsize
                )
        a_doubled = cuda.from_device_like(self.a_gpu, self.a)
        cuda.stop_profiler()
        t1 = time.time()
        print("gpu %0.7f" % (1000*(t1-t0)))
        return a_doubled

    def cpu_sum(self):
        t0 = time.time()
        dat = make_timeseries(self.tstamps, numpy.bincount(
            numpy.repeat(numpy.arange(self.counts.size), self.counts),
            weights=self._ts['values']))
        t1 = time.time()
        print("cpu %0.7f" % (1000*(t1-t0)))
        return dat
AggregatedTimeSerie.benchmark()
