import tempfile
import shutil
import random
import pickle
import os
import math
import re
import sys
import time
from collections import deque
import multiprocessing as mp
import traceback

import numpy as np
import wx
import wx.lib.agw.speedmeter as SM
class ThreadsException(Exception):
    """
    Save exceptions from each thread
    throw them in a bundle when all threds complete/exit out
    """

    def __init__(self, threads, exceptions):
        self.threads = threads
        self.exceptions = exceptions
        self.message = 'Exceptions are thrown from some threads.\n'
        for i, exception in zip(threads, exceptions):
            self.message += 'Thread %d:\n' % i
            self.message += '%s\n' % str(exception)
        super().__init__(self.message)


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


tags = enum(
    "FUNC",
    "START",
    "END",
    "ERROR",
    "READY",
    "IO",
    "EXIT",
    "NOTHING",
    "REFRESH_RATE",
    "REFRESH_SMEAR",
)


class OutputPipe(object):
    def __init__(self, cconn, refresh_interval=5, refresh_delay=0):
        self._cconn = cconn
        self.refresh_interval = refresh_interval
        # smear the refresh interval so the update from all workers won't come
        # all at once which skew the speedometer
        self.smeared_refresh_interval = refresh_interval
        self.last_refresh = time.time()
        self.last_sentence = None
        self.refresh_delay = refresh_delay

    def write(self, output):
        current_time = time.time()
        last_sentence = output.rstrip()
        if last_sentence:
            self.last_sentence = last_sentence
            if (
                current_time - self.last_refresh > self.smeared_refresh_interval
            ):  # + self.refresh_delay*random.uniform(-1, 1):
                self.smeared_refresh_interval = (
                    self.refresh_interval + self.refresh_delay * random.uniform(-1, 1)
                )
                self._cconn.send((self.last_sentence, tags.IO))
                self.last_refresh = current_time

    def flush(self):
        pass
        # if self.last_sentence:
        #  self.comm.send(last_sentence, tag=tags.IO, dest=self.root)

    def CustomFlush(self):
        if self.last_sentence:
            self._cconn.send((self.last_sentence, tags.IO))

class Process(mp.Process):
    def __init__(self, refresh_invertal, refresh_delay, target, **kwargs):
        self.__pconn, self.__cconn = mp.Pipe()
        self.calculation_ended = False
        def wrapper(**kwargs):
            orig_stdout = sys.stdout
            sys.stdout = OutputPipe(self.__cconn, refresh_invertal, refresh_delay)
            result = target(**kwargs)
            sys.stdout.CustomFlush()
            self.__cconn.send((result, tags.END))
            sys.stdout = orig_stdout

        super().__init__(target=wrapper, kwargs=kwargs)

    def run(self):
        try:
            mp.Process.run(self)
        except Exception as e:
            self.__cconn.send((e, tags.ERROR))

    def RecvIO(self):
        if self.__pconn.poll():
            res = self.__pconn.recv()
            if res[1] == tags.END:
                self.calculation_ended = True
            return res
        else:
            return None

    def IsRunning(self):
        return not self.calculation_ended


class MasterSlave:
    def __init__(self, comm, refresh_interval=5, ncores=5):
        self.process = []
        self.refresh_interval = refresh_interval
        self.refresh_smear = 0
        self.nworkers = ncores
        self.size = ncores
        self.results = []
        self.exception_list = []
        self.exception_threads = []
        self.nworking = 0

    def __del__(self):
        self.Close()

    def Close(self):
        for process in self.process:
            if process.IsRunning():
                process.join()

    def RefreshRate(self, refresh_interval):
        self.refresh_interval = refresh_interval 

    def RefreshSmear(self, refresh_smear):
        self.refresh_smear = refresh_smear

    def Submit(self, func, **kwargs):
        self.process = []
        self.results = []
        for worker in range(self.nworkers):
            self.process.append(Process(self.refresh_interval, self.refresh_smear, 
                target=func, **kwargs))
            self.process[-1].start()
        self.nworking = self.nworkers

    def IsRunning(self, duration=0.05):
        if sum([p.IsRunning() for p in self.process]) == 0:
            self.process = []
            self.nworking = 0
            """
            Throw exceptions from each threads when calculation is done
            """
            if len(self.exception_list) > 0:
                """
                Throw exceptions from each threads when calculation is done
                """
                ex = ThreadsException(
                    self.exception_threads, self.exception_list)
                self.exception_list = []
                self.exception_threads = []
                raise ex
            return False            

        if duration > 0:
            time.sleep(duration)

        for idx, p in enumerate(self.process):
            if not p.IsRunning():
                continue
            result = p.RecvIO()
            if result is not None:
                tag = result[1]
                result = result[0]
                if tag == tags.END:
                    p.join()
                    self.nworking = self.nworking - 1
                    self.results.append(result)
                    result = None
                elif tag == tags.ERROR:
                    self.exception_list.append(result)
                    self.exception_threads.append(idx)
                self.stdout = (idx, result, tag)
                return True # only get the latest stdout from one thread at a time
        self.stdout = (0, None, tags.NOTHING)
        return True
  
    def WaitForAll(self):
        while self.IsRunning(0):
            pass


if __name__ == '__main__':
    def func(a, b):
        t = time.time()
        while time.time() - t < 10:
            print('calculating')
        return a + b

    mslave = MasterSlave(None,1,2)
    mslave.RefreshSmear(0.5)
    mslave.Submit(func=func, a=2, b=3)     
    while mslave.IsRunning():
        if mslave.stdout[2] == tags.IO:
            print(mslave.stdout[1])
    print(mslave.results)
    mslave.Close()
        
