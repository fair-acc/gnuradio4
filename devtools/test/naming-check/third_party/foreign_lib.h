#ifndef FOREIGN_LIB_H
#define FOREIGN_LIB_H

namespace ForeignUi {
int Render();
}

namespace outer_lib::NodeEditor {}

struct Foreign_Event {
    int type;
};

extern "C" int Foreign_PushEvent(Foreign_Event* event);

#endif
