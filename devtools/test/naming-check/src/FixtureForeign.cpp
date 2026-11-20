// expect-file: ok
// another library's names, reopened: `third_party/` and `ThirdPartyToast.hpp` are excluded, so what they declare is foreign
#include "../third_party/foreign_lib.h"
#include "ThirdPartyToast.hpp"

namespace ForeignUi {           // expect: ok
int Render() { return 1; }      // expect: ok
int RenderTwice() { return 2; } // expect: R5 -> renderTwice
} // namespace ForeignUi

namespace outer_lib::NodeEditor { // expect: ok
} // namespace outer_lib::NodeEditor

int fixtureToast() { return VendorNotify::Show_Toast(1); } // expect: ok
