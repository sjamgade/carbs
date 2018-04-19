#!/usr/bin/env python
# This file is carbonara library from Gnocchi (https://github.com/gnocchixyz/gnocchi)

import six
import time
import math
import numpy
import re
import random
import struct
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from jinja2 import Template

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



krnl_round_timestamps = SourceModule("""
            #include "math_functions.h"
            __global__ void round_timestamp(unsigned long long *a, unsigned long long freq) {{
              int col;
              unsigned long long UNIX_UNIVERSAL_START64={UNIX_UNIVERSAL_START64};
              col = blockIdx.x * blockDim.x + threadIdx.x;
              a[col] = (UNIX_UNIVERSAL_START64 + ((a[col] - UNIX_UNIVERSAL_START64) / freq) * freq);
              }}
            """.format(UNIX_UNIVERSAL_START64=numpy.int64(UNIX_UNIVERSAL_START64)),
                        options=['--generate-line-info'],
                        keep=True).get_function('round_timestamp')


def gpuround_timestamp(ts, freq):
    gpu_in = cuda.to_device(numpy.ascontiguousarray(ts.view('uint64')))
    krnl_round_timestamps(gpu_in, numpy.int64(freq*1e9), block=(1024,1,1), grid=(ts.size//1024,1))
    dt = cuda.from_device(gpu_in, ts.size, 'uint64')
    return dt




krnl_gpu_sum = SourceModule("""
        __global__ void gpu_sum(float *a, float *op, int perthread) {
          int col, index, counter;
          extern __shared__ float inter[];
          col = blockIdx.x * blockDim.x + threadIdx.x;
          int tid = threadIdx.x;
          #pragma unroll
          for(counter=0;counter<perthread;counter++){
              index = col + blockDim.x*counter;
              inter[blockDim.x * counter + tid] = a[index];
          }

          __syncthreads();
          int x = 0;
          #pragma unroll
          for(counter=0;counter<perthread;counter++){
              x += inter[perthread*tid + counter];
          }
          op[col] = x;
        }
        """, options=['--generate-line-info'], keep=True).get_function("gpu_sum")



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
        usegpu = True if self.aggregation_method.startswith('gpu_') else False
        return AggregatedTimeSerie.from_grouped_serie(
            self.group_serie(sampling, usegpu=usegpu), sampling, self.aggregation_method)

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

    def group_serie(self, granularity, start=None, usegpu=False):
        # NOTE(jd) Our whole serialization system is based on Epoch, and we
        # store unsigned integer, so we can't store anything before Epoch.
        # Sorry!
        if len(self.ts) != 0 and self.first < UNIX_UNIVERSAL_START64:
            raise BeforeEpochError(self.first)

        return GroupedGpuBasedTimeSeries(self.ts, granularity, start, usegpu=usegpu)

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
        points = 1024*1024*6
        points = int(points)
        sampling = numpy.timedelta64(5, 's')
        resample = numpy.timedelta64(30, 's')

        now = numpy.datetime64("2015-04-03 23:11")
        timestamps = numpy.sort(numpy.array( [now + i * sampling for i in six.moves.range(points)]))

        for title, values in [
                ("Simple continuous range", six.moves.range(points)),
        ]:
            def per_sec(t1, t0):
                return 1 / ((t1 - t0) / serialize_times)

            print(title)
            serialize_times = 50
            ts = cls.from_data(sampling, 'mean', timestamps, values)
            pts = ts.ts.copy()

            for agg in ['gpu_sum', 'cpu_sum']:
            #for agg in ['sum']:
                if agg.startswith('gpu_'):
                    cuda.start_profiler()
                serialize_times = 3 if agg.endswith('pct') else 1
                ts = cls(ts=pts, sampling=sampling,
                         aggregation_method=agg)
                t0 = time.time()
                for i in six.moves.range(serialize_times):
                    ts.resample(resample)
                t1 = time.time()
                if agg.startswith('gpu_'):
                    cuda.stop_profiler()
                print("  resample(%s) speed: %.2f Hz, time: %.5f msec"
                      % (agg, per_sec(t1, t0), 1000*(t1-t0)/serialize_times))


class GroupedGpuBasedTimeSeries(object):
    def __init__(self, ts, granularity, start=None, usegpu=False):
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

        if usegpu:
            ret = gpuround_timestamp(self._ts['timestamps'], granularity)
            self.indexes = ret.view(dtype="datetime64[ns]")
        else:
            self.indexes = round_timestamp(self._ts['timestamps'], granularity)

        self.a = self._ts['values']
        self.a = self.a.copy(order='C').astype(numpy.float32)
        self.junk = cuda.mem_alloc(self.a.dtype.itemsize)
        self.reduce_by = int(self.granularity / numpy.timedelta64(5, 's'))

        #self.in_gpu = cuda.pagelocked_empty(self.a.nbytes, numpy.float32, "C")
        self.in_gpu = cuda.mem_alloc(self.a.nbytes)
        cuda.memcpy_htod_async(self.in_gpu, self.a)
        self.ret_gpu = cuda.mem_alloc(int(self.a.size* self.a.dtype.itemsize / self.reduce_by))
        self.tstamps, self.counts = numpy.unique(self.indexes,
                                                 return_counts=True)


    def gpu_sum(self):
        t0 = time.time()
#        in_gpu = cuda.mem_alloc(self.a.size * self.a.dtype.itemsize)
#        cuda.memcpy_htod(in_gpu, self.a)
#        ret_gpu = cuda.mem_alloc(int(self.a.size* self.a.dtype.itemsize / self.reduce_by))
        gridsize = 1024
        krnl_gpu_sum(self.in_gpu, self.ret_gpu, numpy.int32(self.reduce_by),
                block=(1024, 1, 1), grid=(gridsize, 1),
                shared=1024*self.reduce_by*self.a.dtype.itemsize
                )
        ret = numpy.empty(self.a.size / self.reduce_by, numpy.float32)
        cuda.memcpy_dtoh(ret, self.ret_gpu)
        dat = make_timeseries(self.tstamps, ret)
        t1 = time.time()
        print("  time to aggregate  %0.7f msec" % (1000*(t1-t0)))
        return dat

    def cpu_sum(self):
        t0 = time.time()
        dat = make_timeseries(self.tstamps, numpy.bincount(
            numpy.repeat(numpy.arange(self.counts.size), self.counts),
            weights=self._ts['values']))
        t1 = time.time()
        print("  time to aggregate  %0.7f msec" % (1000*(t1-t0)))
        return dat
AggregatedTimeSerie.benchmark()
