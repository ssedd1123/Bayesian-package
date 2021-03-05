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

import dill
import numpy as np
import wx
import wx.lib.agw.speedmeter as SM
from mpi4py import MPI

MPI.pickle.__init__(dill.dumps, dill.loads)


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
    def __init__(self, comm, root, refresh_interval=5, refresh_delay=0):
        self.comm = comm
        self.root = root
        self.refresh_interval = refresh_interval
        # smear the refresh interval so the update from all workers won't come
        # all at once which skew the speedometer
        self.smeared_refresh_interval = refresh_interval
        self.last_refresh = time.time()
        self.last_sentence = None
        self.refresh_delay = 0

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
                self.comm.send(self.last_sentence, tag=tags.IO, dest=self.root)
                self.last_refresh = current_time

    def flush(self):
        pass
        # if self.last_sentence:
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
        self.exception_threads = []
        self.exception_list = []
        self.EventLoop()

    def __del__(self):
        self.Close()

    def RefreshRate(self, refresh_interval):
        for worker in range(1, self.size):
            self.comm.send(
                refresh_interval,
                tag=tags.REFRESH_RATE,
                dest=worker)

    def RefreshSmear(self, refresh_smear):
        for worker in range(1, self.size):
            self.comm.send(refresh_smear, tag=tags.REFRESH_SMEAR, dest=worker)

    def Submit(self, func, **kwargs):
        self.results = []
        if self.rank == 0:
            if self.nworking != 0:
                raise RuntimeError(
                    "Current working ranks > 0. Are you submitting new jobs while the old one are still running? This function is not yet supported"
                )
            for worker in range(1, self.size):
                self.comm.send(func, tag=tags.FUNC, dest=worker)

            self.nworking = self.nworkers
            for worker in reversed(range(1, self.size)):
                self.comm.send(kwargs, tag=tags.START, dest=worker)

    def IsRunning(self, duration=0.05):
        if self.rank == 0:
            if self.nworking == 0:
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

            # source, tag, result = polling_receive(self.comm)
            if duration > 0:
                time.sleep(duration)
            if self.comm.Iprobe(source=MPI.ANY_SOURCE):
                status = MPI.Status()
                result = self.comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                )
                tag = status.Get_tag()
                source = status.Get_source()
                received = True

                if tag == tags.END:
                    self.nworking = self.nworking - 1
                    self.results.append(result)
                    self.stdout = (source, None, tags.END)
                else:
                    if tag == tags.ERROR:
                        self.exception_list.append(result)
                        self.exception_threads.append(source)
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
                kwargs = self.comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                )
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
