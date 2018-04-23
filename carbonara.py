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


krnl_uniques = SourceModule("""
                __global__ void count_uniqs(unsigned long long *a,
                                            int *uniqs,
                                            float *in,
                                            float *op) {
                  int col, count, should, sum;
                  unsigned long long here;

                  col = blockIdx.x * blockDim.x + threadIdx.x;
                  int index = col;
                  count = 0;
                  sum = 0;
                  here = a[index];
                  if (col > 0){
                      if (here != a[col-1]){
                          sum = in[col];
                          should = 1;
                          //atomicAdd(&unques, 1);
                      }
                      else { should = 0; }
                   }
                   else {
                       should=1;
                   }
                   //__syncthreads();

                  if (should) {
                      count=1;
                      col++;
                      while (here == a[col]){
                          sum += in[col];
                          count++;
                          col++;
                          if (col >= gridDim.x * blockDim.x)
                            break;
                      }
                  }
                  op[index] = sum;
                  uniqs[index] = count;
                }
                """).get_function('count_uniqs')

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


def gpuround_timestamp(ts, freq, dev_ptr=True):
    strm = stream=cuda.Stream()
    gpu_in = cuda.mem_alloc(ts.nbytes)
    cuda.memcpy_htod_async(gpu_in, numpy.ascontiguousarray(ts.view('uint64')), stream=strm)
    krnl_round_timestamps(gpu_in, numpy.int64(freq*1e9),
                            block=(1024,1,1),
                            grid=(ts.size//1024,1),
                            stream=strm)
    if dev_ptr:
        dt = gpu_in
    else:
        dt = cuda.from_device(gpu_in, ts.size, 'uint64', strm)
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

        points = 32*1024
        points = int(points)
        sampling = numpy.timedelta64(5, 's')
        resample = numpy.timedelta64(50, 's')

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
                print("resample(%s) speed: %.2f Hz, time: %.5f msec"
                      % (agg, per_sec(t1, t0), 1000*(t1-t0)/serialize_times))


class GroupedGpuBasedTimeSeries(object):
    def __init__(self, ts, granularity, start=None, usegpu=False):
        # NOTE(sileht): The whole class assumes ts is ordered and don't have
        # duplicate timestamps, it uses numpy.unique that sorted list, but
        # we always assume the orderd to be the same as the input.
        self.granularity = granularity
        self.start = start
        self.streams = {}
        if start is None:
            self._ts = ts
            self._ts_for_derive = ts
        else:
            # this else is never traversed
            self._ts = ts[numpy.searchsorted(ts['timestamps'], start):]
            start_derive = start - granularity
            self._ts_for_derive = ts[
                numpy.searchsorted(ts['timestamps'], start_derive):
            ]

        if usegpu:
            t0 = time.time()
            self.indexes = gpuround_timestamp(self._ts['timestamps'], granularity, True)
            self.a = self._ts['values'].astype(numpy.float32, order="C")
            self.in_gpu = cuda.mem_alloc(self.a.nbytes)
            cuda.memcpy_htod_async(self.in_gpu, self.a)
            #self.indexes = ret.view(dtype="datetime64[ns]")

            self.uniqs_gpu = cuda.mem_alloc(self.a.size * numpy.dtype('int32').itemsize)
            self.sum_gpu = cuda.mem_alloc(self.a.nbytes)
            self.streams['tstamps'] = cuda.Stream()
            #krnl_uniques(cuda.In(self.indexes.view('uint64')),
            krnl_uniques(self.indexes,
                            self.uniqs_gpu,
                            self.in_gpu,
                            self.sum_gpu,
                            block=(1024,1,1),
                            grid=(self.a.size//1024,1),
                            stream=self.streams['tstamps'])

            
            self.tcounts = numpy.empty_like(self.a, dtype='int32')
            self.sum_done = numpy.empty_like(self.a, dtype='float32')
            cuda.memcpy_dtoh_async(self.tcounts, self.uniqs_gpu, stream=self.streams['tstamps'])
            #cuda.memcpy_dtoh(self.tcounts, self.uniqs_gpu)
            cuda.memcpy_dtoh_async(self.sum_done, self.sum_gpu, stream=self.streams['tstamps'])
            #cuda.memcpy_dtoh(self.sum_done, self.sum_gpu)
            t1 = time.time()
            print("gpu uniques t1 %0.7f" % (1000*(t1-t0)))
            #self.debuguniques()

        else:
            t0 = time.time()
            self.indexes = round_timestamp(self._ts['timestamps'], granularity)
            self.tstamps, self.counts = numpy.unique(self.indexes, return_counts=True)
            t1 = time.time()
            print("uniques %0.7f" % (1000*(t1-t0)))
                    

    def gpu_sum(self):
        del self.sum_gpu
        self.streams['tstamps'].synchronize()
        self.counts = self.tcounts[self.tcounts > 0]
        self.tstamps = self._ts['timestamps'][self.tcounts >0 ]
        return self.sum_done

    def cpu_sum(self):
        t0 = time.time()
        dat = make_timeseries(self.tstamps, numpy.bincount(
            numpy.repeat(numpy.arange(self.counts.size), self.counts),
            weights=self._ts['values']))
        t1 = time.time()
        print("  time to aggregate  %0.7f msec" % (1000*(t1-t0)))
        return dat


    def debuguniques(self):
        self.indexes_c = round_timestamp(self._ts['timestamps'], granularity)
        t0 = time.time()
        self.tstamps_c, self.counts_c = numpy.unique(self.indexes,
                                                 return_counts=True)
        t1 = time.time()
        print("uniques %0.7f" % (1000*(t1-t0)))
                
        try:
            assert all(self.tstamps_c == self.tstamps)
        except:
            ret = cuda.from_device(self.sum_gpu, self.a.size, numpy.float32)
            import pdb;pdb.set_trace()
        try:
            assert all(self.counts_c == self.counts)
        except:
            ret = cuda.from_device(self.sum_gpu, self.a.size, numpy.float32)
            import pdb;pdb.set_trace()


#POINTS_PER_SPLIT = 3600
#points = POINTS_PER_SPLIT * 1000
#points = 2 * 1024 * 6
#points = 512*1024*6
#points = 1024*1024*6
#points = int(points)
#sampling = numpy.timedelta64(5, 's')
#resample = numpy.timedelta64(30, 's')
#
#now = numpy.datetime64("2015-04-03 23:11")
#timestamps = numpy.sort(numpy.array( [now + i * sampling for i in six.moves.range(points)]))
#a  = numpy.array(six.moves.range(points), dtype=numpy.float32, order="C")
#cuda.start_profiler()
#in_gpu = cuda.mem_alloc(a.nbytes)
#cuda.memcpy_htod_async(in_gpu, a, stream=cuda.Stream())
#sum_gpu = cuda.mem_alloc(a.nbytes)
#rounded = gpuround_timestamp(timestamps.astype('datetime64[ns]'), resample)
#d_counts = cuda.mem_alloc(a.size * numpy.dtype('int32').itemsize)
#krnl_uniques(cuda.In(rounded), d_counts, in_gpu, sum_gpu, block=(1024,1,1), grid=(1024*6,1))
#ret = cuda.from_device(sum_gpu, a.size, numpy.float32)
#cuda.stop_profiler()
#print rounded
#print ret
AggregatedTimeSerie.benchmark()
