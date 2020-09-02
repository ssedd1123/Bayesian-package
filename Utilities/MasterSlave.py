from collections import deque
import numpy as np
import math
import wx
import wx.lib.agw.speedmeter as SM
import sys
import time
import re
from mpi4py import MPI
import dill
MPI.pickle.__init__(dill.dumps, dill.loads)

import random
import tempfile
import shutil
import pickle
import os

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('FUNC', 'START', 'END', 'ERROR', 'READY', 'IO', 'EXIT', 'NOTHING', 'REFRESH_RATE', 'REFRESH_SMEAR')

class OutputPipe(object):
  def __init__(self, comm, root, refresh_interval=5, refresh_delay=0):
    self.comm = comm
    self.root = root
    self.refresh_interval = refresh_interval
    self.smeared_refresh_interval = refresh_interval # smear the refresh interval so the update from all workers won't come all at once which skew the speedometer
    self.last_refresh = time.time()
    self.last_sentence = None
    self.refresh_delay = 0

  def write(self, output):
    current_time = time.time()
    last_sentence = output.rstrip()
    if last_sentence:       
      self.last_sentence = last_sentence
      if current_time - self.last_refresh > self.smeared_refresh_interval:# + self.refresh_delay*random.uniform(-1, 1):
        self.smeared_refresh_interval = self.refresh_interval + self.refresh_delay*random.uniform(-1, 1)
        self.comm.send(self.last_sentence, tag=tags.IO, dest=self.root)
        self.last_refresh = current_time

  def flush(self):
    pass
    #if self.last_sentence:
    #  self.comm.send(last_sentence, tag=tags.IO, dest=self.root)

  def CustomFlush(self):
    if self.last_sentence:
      self.comm.send(self.last_sentence, tag=tags.IO, dest=self.root)
    

class MasterSlave(object):
  def __init__(self, comm, refresh_interval=5):
    self.comm = comm
    self.rank = comm.Get_rank()
    self.size = comm.Get_size()
    self.nworkers = self.size - 1
    self.nworking = 0
    self.func = None
    self.results = []
    if self.rank != 0:
      sys.stdout = OutputPipe(comm, 0, refresh_interval)
    self.EventLoop()

  def __del__(self):
    self.Close()

  def RefreshRate(self, refresh_interval):
    for worker in range(1, self.size):
      self.comm.send(refresh_interval, tag=tags.REFRESH_RATE, dest=worker)

  def RefreshSmear(self, refresh_smear):
    for worker in range(1, self.size):
      self.comm.send(refresh_smear, tag=tags.REFRESH_SMEAR, dest=worker)

  def Submit(self, func, **kwargs):
    self.results = []
    if self.rank == 0:
      if self.nworking != 0:
        raise RuntimeError('Current working ranks > 0. Are you submitting new jobs while the old one are still running? This function is not yet supported')
      for worker in range(1, self.size):
        self.comm.send(func, tag=tags.FUNC, dest=worker)

      self.nworking = self.nworkers
      for worker in reversed(range(1, self.size)):
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
            sys.stdout.CustomFlush()
            self.comm.send(result, tag=tags.END, dest=0)
        elif source == 0 and tag == tags.FUNC:
            self.func = kwargs
        elif source == 0 and tag == tags.REFRESH_RATE:
            sys.stdout.refresh_interval = kwargs
        elif source == 0 and tag == tags.REFRESH_SMEAR:
            sys.stdout.refresh_delay = kwargs
        elif source == 0 and tag == tags.EXIT:
          sys.exit(0)
    

     
