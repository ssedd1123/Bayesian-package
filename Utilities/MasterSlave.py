from collections import deque
import numpy as np
import math
import wx
import wx.lib.agw.speedmeter as SM
import sys
import time
import re
from mpi4py import MPI
import tempfile
import shutil
import pickle
import os

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('START', 'END', 'ERROR', 'READY', 'IO', 'EXIT', 'NOTHING')

class OutputPipe(object):
  def __init__(self, comm, root):
    self.comm = comm
    self.root = root

  def write(self, output):
    if output.rstrip():
      self.comm.send(output, tag=tags.IO, dest=self.root)

  def flush(self):
    pass

class MasterSlave(object):
  def __init__(self, comm, func):
    self.comm = comm
    self.rank = comm.Get_rank()
    self.size = comm.Get_size()
    self.nworkers = self.size - 1
    self.nworking = 0
    self.func = func
    self.results = []
    if self.rank != 0:
      sys.stdout = OutputPipe(comm, 0)

  def Submit(self, **kwargs):
    self.results = []
    if self.rank == 0:
      if self.nworking != 0:
        raise RuntimeError('Current working ranks > 0. Are you submitting new jobs while the old one are still running? This function is not yet supported')

      self.nworking = self.nworkers
      for worker in range(1, self.size):
        self.comm.send(kwargs, tag=tags.START, dest=worker)

  def IsRunning(self, duration=0.05):
    if self.rank == 0:
      if self.nworking == 0:
        return False
 
      #source, tag, result = polling_receive(self.comm)
      if duration > 0:
        time.sleep(duration)
      if self.comm.Iprobe(source=MPI.ANY_SOURCE):
        status = MPI.Status()
        result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        received = True

        if tag == tags.END:
          self.nworking = self.nworking - 1
          self.results.append(result)
        elif tag == tags.ERROR:
          self.nworking = self.nworking - 1
        self.stdout = (source, result, tag)
      else:
        self.stdout = (0, None, tags.NOTHING)
      return True

    return False
    
  def WaitForAll(self):
    while self.IsRunning(0):
      pass
    
  def Close(self):
    # Listen to remaining classes
    self.WaitForAll()

    if self.rank == 0:
      for worker in range(1, self.size):
        self.comm.send(None, tag=tags.EXIT, dest=worker)

  def EventLoop(self):
    if self.rank != 0:
      while True:
        sleep_seconds = 0.05
        if sleep_seconds > 0:
            while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
                time.sleep(sleep_seconds)

        status = MPI.Status()
        kwargs = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if source == 0 and tag == tags.START:
          try:
            result = self.func(**kwargs)
          except Exception as e:
            self.comm.send(e, tag=tags.ERROR, dest=0)
          else:
            self.comm.send(result, tag=tags.END, dest=0)
        elif source == 0 and tag == tags.EXIT:
          break
    

     
