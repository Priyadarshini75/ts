from deep_sort_realtime.deepsort_tracker import DeepSort

def get_tracker():
    return DeepSort(max_age=30)
