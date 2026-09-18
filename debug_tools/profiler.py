# Import dependencies
import time

'''
Usage Instructions:

Near the top the eye_pipeline, set PROFILE = True to turn on the profiler. T
Then, create a Profiler object (profiler = Profiler(PROFILE))
Next, run profiler.start() where you want to start timing. Use profiler.checkpoint("_Name_of_Checkpoint_") after each event you want to time.
Replace _Name_of_Checkpoint_ with specific event name.
Finally, run profiler.end() to end timing. It will return the overall time elapsed from .start() to .end()
'''

# Define Class for Profiler     
class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.start_time = None
        self.last_time = None

    # Enable Profiler
    def start(self):
        if self.enabled:
            self.start_time = time.perf_counter()
            self.last_time = self.start_time

    # Profiler Checkpoint. Prints time from last checkpoint to current checkpoint
    def checkpoint(self, name):
        if self.enabled:
            current_time = time.perf_counter()

            print(f"{name}: {current_time - self.last_time:.4f} s")

            self.last_time = current_time

    # Clear time from last checkpoint if in between operations do not want to be measured. An 'unnamed' checkpoint
    def clear(self):
        if self.enabled:
            self.last_time = time.perf_counter()

    # Profiler prints total time since it was enabled.
    def end(self):
        if self.enabled:
            current_time = time.perf_counter()

            print(f"Total frame: {current_time - self.start_time:.4f} s")
            print()