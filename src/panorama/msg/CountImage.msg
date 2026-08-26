Header header          # Synchronization timestamp for the event window
time init_time         # Timestamp of the first event in the window
int32[] count_image    # Channel-major tensor; each channel is row-major H x W