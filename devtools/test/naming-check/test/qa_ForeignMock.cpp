// expect-file: ok
// stands in for third_party/foreign_lib.h, which this unit does not include: the names stay the library's
struct Foreign_Event { // expect: ok
    int type = 0;
};

int Foreign_PushEvent(Foreign_Event* event) { return event->type; } // expect: ok

int main() {
    Foreign_Event event;
    return Foreign_PushEvent(&event);
}
