"""Simple timer which can be used to get the wall/cpu time of a process.

Implements the "with x:" to measure timings more easily. 
"""


from time import process_time
import time

class CpuTimer:
	def __init__(self):
		self._start_time = process_time()
		self._paused = True
		self._pause_time = process_time()
		self._paused_time = 0


	def start_timer(self):
		if self._paused:
			self._paused = False
			self._paused_time = process_time() - self._pause_time
		else:
			print("Warning: timer has already started")



	def __enter__(self):
		self.start_timer()

	def __exit__(self, exc_type, exc_value, tb):
		self.pause_timer()

	def pause_timer(self):
		if self._paused:
			print("Warning: timer already paused")
		else:
			self._paused = True
			self._pause_time = process_time()


	def get_time(self) -> float:
		if self._paused: #If current paused time needs to be subtracted as well
			return process_time() - self._start_time - self._paused_time - (process_time() - self._pause_time)
		else: #If only total paused time needs to be subtracted
			return process_time() - self._start_time - self._paused_time


class WallTimer:
	def __init__(self):
		self._start_time = time.time()
		self._paused = True
		self._pause_time = time.time()
		self._paused_time = 0


	def __enter__(self):
		self.start_timer()

	def __exit__(self, exc_type, exc_value, tb):
		self.pause_timer()

	def start_timer(self):
		if self._paused:
			self._paused = False
			self._paused_time = time.time() - self._pause_time
		else:
			print("Warning: timer has already started")

	def pause_timer(self):
		if self._paused:
			print("Warning: timer already paused")
		else:
			self._paused = True
			self._pause_time = time.time()


	def get_time(self) -> float:
		if self._paused: #If current paused time needs to be subtracted as well
			return time.time() - self._start_time - self._paused_time - (time.time() - self._pause_time)
		else: #If only total paused time needs to be subtracted
			return time.time() - self._start_time - self._paused_time




if __name__ == "__main__":
	print("Now testing timer")
	thetimer = CpuTimer()

	print(thetimer.get_time())

	thetimer.start_timer()
	time.sleep(1.0)
	thetimer.pause_timer()
	for i in range(100000):
		print("kaas")
	print(thetimer.get_time())
	print("Done")
