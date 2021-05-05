
import numpy as np
import threading

class NumpyQueue(): #TODO: make threadsafe
    def __init__(self, size, roll_when_full=False, dtype='float32'):
        self._arr = np.empty(size, dtype=dtype)
        self._cur_idx = 0
        self.roll_when_full = roll_when_full

    def append(self, values):
        if self._cur_idx + len(values) > len(self._arr):
            if self.roll_when_full: #if roll (overwrite start of queue) to make room
                self._arr = np.roll(self._arr, len(self._arr) - self._cur_idx - len(values))
                self._arr[max(-len(self._arr), -len(values)): ] = values[:min(len(values),len(self._arr))]
            elif self._cur_idx < len(self._arr): #if at least 1 position free
                self._arr[self._cur_idx : ] = values[0 : len(self._arr) - self._cur_idx] #Add as many items as possible
            self._cur_idx = len(self._arr)
        else:
            self._arr[self._cur_idx: self._cur_idx + len(values)] = values
            self._cur_idx += len(values)
    
    def peek_idx(self, idx): 
        return self._arr[idx].copy() #TODO: copy or not? is this safe

    def peek(self, count):
        return self._arr[:count].copy()

    def __len__(self):
        return self._cur_idx

    def __str__(self):
        return str(self._arr)
    
    def pop(self, count):
        assert count >= 0 
        if self._cur_idx < count:
            raise Exception(f'{count} exceeds amount of items in queue ({self._cur_idx + 1})')

        vals = self._arr[0: count].copy()
        self._cur_idx -=count
        self._arr = np.roll(self._arr, -count)
        return vals

class ThreadNumpyQueue(NumpyQueue): #TODO: although thread unsafe methods are now safe, typing the functions out and only using the locks at the last line might be more efficient
    
    #TODO: return copies? Otherwise it might still go wrong (althoug not sure)
    def __init__(self, *args, **kwargs):
        super(ThreadNumpyQueue, self).__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def append(self, values):
        with self._lock:
            return super(ThreadNumpyQueue, self).append(values)
    
    def peek_idx(self, idx):
        with self._lock:
            return super(ThreadNumpyQueue, self).peek_idx(idx)

    def peek(self, count):
        with self._lock:
            return super(ThreadNumpyQueue, self).peek(count)

    def __len__(self):
        return self._cur_idx

    def __str__(self):
        with self._lock:
            return super(ThreadNumpyQueue, self).__str__()
    
    def pop(self, count):
        with self._lock:
            return super(ThreadNumpyQueue, self).pop(count)


if __name__ == "__main__":
    testqueue = NumpyQueue(5000, False)

    for i in range(10000):
        testqueue.append([i])
        testqueue.pop(1)
    
    assert len(testqueue) == 0

    for i in range(10000):
        testqueue.append([i])
    
    assert np.all(testqueue.pop(5000) == np.array(range(0, 5000)))
    
    testqueue = NumpyQueue(5000, True)

    for i in range(10000):
        testqueue.append([i])
    assert np.all(testqueue.pop(5000) == np.array(range(5000, 10000)))
    assert len(testqueue) == 0


    randarr = np.random.rand(100)
    testqueue = NumpyQueue(5000, True, dtype=np.float64)
    for i in range(50):
        testqueue.append(randarr)
    for i in range(50):
        returned_arr = testqueue.pop(len(randarr))
        assert np.all(randarr == returned_arr)

    testqueue.append(range(100))
    for i in range(10):
        testqueue.append(range(5))
        testqueue.pop(10)
    # print(f"Len testqueue: {len(testqueue)}")
    # for i in testqueue.pop(50):
    #     print(i)

    assert np.all(testqueue.pop(50) == np.array( [*range(5)] * 10))

    for i in range(1, 10):
        testqueue.append([i] * i)
        testqueue.pop(max(1, i-1))

    print(f"len is now: {len(testqueue)}")
    for i in testqueue.pop(len(testqueue)):
        print(i)

    for i in range(10):
        testqueue.append([i]*100)

    testqueue.pop(900)
    assert np.all(testqueue.pop(100) == np.array([9] * 100))
        

    print("All assertions valid!")
    
