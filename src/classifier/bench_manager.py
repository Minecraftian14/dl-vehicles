from time import perf_counter


class Timer:

    def __init__(self, name, record_history=False):
        self.name = name
        self.record_history = record_history
        self.start_time = self.drag_time = self.end_time = self.total_time = 0
        self.history = []
        self.event_count = 0
        self.average = 0.0

    def start(self):
        self.drag_time = self.start_time = perf_counter()

    def end(self):
        self.end_time = perf_counter()
        self.total_time = self.end_time - self.start_time
        self.average = (self.average * self.event_count + self.total_time) / (self.event_count + 1)
        self.event_count += 1
        if self.record_history:  self.history.append(self.total_time)

    def since(self):
        return perf_counter() - self.start_time

    def drag(self, time: float):
        if perf_counter() - self.drag_time >= time:
            self.drag_time += time
            return True
        return False


class TypedTimer:

    def __init__(self, name, record_history=False):
        self.name = name
        self.record_history = record_history
        self.timers: dict[str, Timer] = {}

    def start(self, event_name: str):
        if event_name not in self.timers:
            self.timers[event_name] = timer = Timer(f'{self.name}.{event_name}', record_history=self.record_history)
        else:
            timer = self.timers[event_name]
        timer.start()

    def end(self, event_name: str):
        self.timers[event_name].end()

    def since(self, event_name: str):
        return self.timers[event_name].since()

    def drag(self, event_name: str, time: float):
        return self.timers[event_name].drag(time)
