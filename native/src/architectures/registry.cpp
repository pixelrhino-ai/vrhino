#include "vrhino/architecture.h"

#include "vrhino/error.h"

namespace vrhino {

std::unique_ptr<Architecture> create_architecture(const VrmModel& model) {
    require(model.graph().at("schema_version").integer() == 1,
            "Package graph requires the owning architecture factory overload");
    if (model.architecture_id() == "wan") return make_wan_architecture(model);
    if (model.architecture_id() == "hunyuan_video") return make_hunyuan_video_architecture(model);
    if (model.architecture_id() == "ltx_v0_9_1") return make_ltx_architecture(model);
    if (model.architecture_id() == "mochi") return make_mochi_architecture(model);
    if (model.architecture_id() == "cogvideox_2b") return make_cogvideox_architecture(model);
    throw Error("Native architecture implementation is not registered: " + model.architecture_id());
}

std::unique_ptr<Architecture> create_architecture(std::shared_ptr<const VrmModel> model) {
    require(model != nullptr, "Missing model owner");
    const auto schema = model->graph().at("schema_version").integer();
    // Legacy adapters explicitly borrow backing. Keep their established factory
    // contract separate rather than silently discarding a managed owner.
    require(schema != 1, "Schema1 uses the caller-retained architecture factory");
    require(schema == 2, "Unsupported architecture package schema");
    // Family capability registration, never product-name dispatch.
    if (model->architecture_id() == "wan") return make_wan_package_architecture(std::move(model));
    throw Error("Registered family does not support package graph admission: " + model->architecture_id());
}
}  // namespace vrhino
