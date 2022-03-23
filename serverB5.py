"""   YPL, JLL, 2021.9.14 - 2022.1.26
(YPN) jinn@Liu:~/YPN/B5$ python serverB5.py
"""
import os
import zmq
import six
import numpy
import random
import logging
import argparse
from numpy.lib.format import header_data_from_array_1_0
from datagenB5 import datagen

BATCH_SIZE = 2

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)

def send_arrays(socket, arrays, stop=False):
  if arrays:
      # The buffer protocol only works on contiguous arrays
    arrays = [numpy.ascontiguousarray(array) for array in arrays]
  if stop:
    headers = {'stop': True}
    socket.send_json(headers)
  else:
    headers = [header_data_from_array_1_0(array) for array in arrays]
    socket.send_json(headers, zmq.SNDMORE)
    for array in arrays[:-1]:
      socket.send(array, zmq.SNDMORE)
    socket.send(arrays[-1])

def recv_arrays(socket):
  headers = socket.recv_json()
  if 'stop' in headers:
    raise StopIteration

  arrays = []
  for header in headers:
    data = socket.recv()
    buf = buffer_(data)
    array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
    array.shape = header['shape']
    if header['fortran_order']:
      array.shape = header['shape'][::-1]
      array = array.transpose()
    arrays.append(array)

  return arrays

def client_generator(port=5557, host="localhost", hwm=20):
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.set_hwm(hwm)
  socket.connect("tcp://{}:{}".format(host, port))
  logger.info('client started')
  while True:
    data = recv_arrays(socket)
    yield tuple(data)

def start_server(data_stream, port=5557, hwm=20):
  logging.basicConfig(level='INFO')
  context = zmq.Context()
  socket = context.socket(zmq.PUSH)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

    # it = itertools.tee(data_stream)
  it = data_stream
  logger.info('server started')
  while True:
    try:
      data = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data)))
    except StopIteration:
      it = data_stream
      data = None
      stop = True
      logger.debug("sending StopIteration")

    send_arrays(socket, data, stop=stop)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Data Server')
  parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
  parser.add_argument('--buffer', dest='buffer', type=int, default=20, help='High-water mark. Increasing this increses buffer and memory usage.')
  parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Serve validation dataset instead.')
  args, more = parser.parse_known_args()

    # Project B Part
  all_dirs = os.listdir('/home/jinn/dataB')
  all_yuvs = ['/home/jinn/dataB/'+i+'/yuv.h5' for i in all_dirs]
    #print('#---  all_yuvs =', all_yuvs)
  random.seed(0) # makes the random numbers predictable
  random.shuffle(all_yuvs)

  train_len  = int(0.5*len(all_yuvs))
  valid_len  = int(0.5*len(all_yuvs))
  train_files = all_yuvs[: train_len]
  valid_files = all_yuvs[train_len: train_len + valid_len]
  print('#---serverB5  len(all_yuvs) =', len(all_yuvs))
  print('#---serverB5  len(train_files) =', len(train_files))
  print('#---serverB5  len(valid_files) =', len(valid_files))

  if args.validation:
      camera_files = valid_files
  else:
      camera_files = train_files

  data_s = datagen(BATCH_SIZE, camera_files)
  start_server(data_s, port=args.port, hwm=args.buffer)
