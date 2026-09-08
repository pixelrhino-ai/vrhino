#include "vrhino/architecture.h"

#include "vrhino/error.h"

namespace vrhino {

std::unique_ptr<Architecture> create_architecture(const VrmModel& model) {
    if (model.architecture_id() == "wan") return make_wan_architecture(model);
    if (model.architecture_id() == "hunyuan_video") return make_hunyuan_video_architecture(model);
    if (model.architecture_id() == "ltx_v0_9_1") return make_ltx_architecture(model);
    if (model.architecture_id() == "mochi") return make_mochi_architecture(model);
    if (model.architecture_id() == "cogvideox_2b") return make_cogvideox_architecture(model);
    throw Error("Native architecture implementation is not registered: " + model.architecture_id());
}

}  // namespace vrhino
