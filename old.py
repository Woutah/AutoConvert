import numpy as np    
    # @property
    # def unprocessed_frames(self):
    #     return self._unprocessed_frames
    
    # @unprocessed_frames.setter
    # def unprocessed_frames(self, value): #set to new value
    #     self.unprocessed_frames_counter.acquire()
    #     self._unprocessed_frames = value 
    #     if self._unprocessed_frames > self.processing_buffer_size:
    #         self._unprocessed_frames = self.processing_buffer_size
    #     self.unprocessed_frames_counter.release()

    # def unprocessed_frames_change(self, change):
    #     self.unprocessed_frames_counter.acquire()
    #     self._unprocessed_frames += change
    #     if self._unprocessed_frames > self.processing_buffer_size: #Make sure it does not exceed 
    #         self._unprocessed_frames = self.processing_buffer_size
    #     self.unprocessed_frames_counter.release()

    # def continuous_recorder(self):
    #     log.info("Opening recording stream...")
    #     while True:
    #         self.chunk_queue.append(np.frombuffer(self.recording_stream.read(CHUNK * 10), "Float32")) #Continuously append frames and play them back
    # def process_chunk(self):

if __name__ == "__main__":
    test = np.empty((10, 20))
    test[0:10] = [np.array(range(20))] * 10 
    print(test)

        
    # def process_recording(self, in_data, frame_count, time_info, status_flags):
    #     # log.info(f"Received some data!  Frame count {frame_count}")
    #     # log.info(f"{self.chunk_queue}\n")
    #     log.info(f"Unprocessed frames: {self.unprocessed_frames}\n")
    #     # self.chunk_queue.append(*(np.frombuffer(in_data, 'Float32')))
    #     self.chunk_queue.extend(in_data)
    #     self.unprocessed_frames_change(frame_count) #add these frames to the "unprocessed" frames
        
    #     return(in_data, pyaudio.paContinue)