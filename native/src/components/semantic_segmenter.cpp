#include "vrhino/semantic_segmenter.h"

#include <string>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

const Tensor* optional(const WeightMap& weights, const std::string& name) {
    return weights.find(name);
}

Tensor conv(Backend& backend, const WeightMap& weights, const Tensor& input,
            const std::string& prefix, int stride = 1, int padding = 0) {
    return backend.conv2d(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          {stride, stride}, {padding, padding}, {1, 1}, 1);
}

Tensor channel_parameter(Backend& backend, const Tensor& value) {
    return backend.reshape(value, {1, value.numel(), 1, 1});
}

Tensor batch_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix) {
    Tensor mean = channel_parameter(backend, weights.at(prefix + ".running_mean"));
    Tensor variance = channel_parameter(backend, weights.at(prefix + ".running_var"));
    Tensor scale = channel_parameter(backend, weights.at(prefix + ".weight"));
    Tensor bias = channel_parameter(backend, weights.at(prefix + ".bias"));
    Tensor centered = backend.add(input, backend.mul(mean, scalar_f32(-1.0f)));
    Tensor denominator = backend.sqrt(backend.add(variance, scalar_f32(1.0e-5f)));
    return backend.add(backend.mul(backend.div(centered, denominator), scale), bias);
}

Tensor conv_bn_relu(Backend& backend, const WeightMap& weights,
                    const Tensor& input, const std::string& prefix,
                    int stride = 1, int padding = 0) {
    return backend.activation(batch_norm(backend, weights,
        conv(backend, weights, input, prefix + ".conv", stride, padding),
        prefix + ".bn"), Activation::Relu);
}

Tensor sigmoid(Backend& backend, const Tensor& input) {
    return backend.div(scalar_f32(1.0f), backend.add(scalar_f32(1.0f),
        backend.exp(backend.mul(input, scalar_f32(-1.0f)))));
}

Tensor global_average(Backend& backend, const Tensor& input) {
    Tensor sum = backend.reduce_sum(input, 2, true);
    sum = backend.reduce_sum(sum, 3, true);
    return backend.mul(sum, scalar_f32(1.0f /
        static_cast<float>(input.dim(2) * input.dim(3))));
}

Tensor basic_block(Backend& backend, const WeightMap& weights,
                   const Tensor& input, const std::string& prefix,
                   int stride, bool projection) {
    Tensor hidden = conv(backend, weights, input, prefix + ".conv1", stride, 1);
    hidden = backend.activation(batch_norm(backend, weights, hidden,
        prefix + ".bn1"), Activation::Relu);
    hidden = conv(backend, weights, hidden, prefix + ".conv2", 1, 1);
    hidden = batch_norm(backend, weights, hidden, prefix + ".bn2");
    Tensor shortcut = input;
    if (projection) {
        shortcut = conv(backend, weights, input, prefix + ".downsample.0", stride);
        shortcut = batch_norm(backend, weights, shortcut,
                              prefix + ".downsample.1");
    }
    return backend.activation(backend.add(shortcut, hidden), Activation::Relu);
}

Tensor attention_refinement(Backend& backend, const WeightMap& weights,
                            const Tensor& input, const std::string& prefix) {
    Tensor feature = conv_bn_relu(backend, weights, input, prefix + ".conv", 1, 1);
    Tensor gate = global_average(backend, feature);
    gate = conv(backend, weights, gate, prefix + ".conv_atten");
    gate = sigmoid(backend, batch_norm(backend, weights, gate,
                                      prefix + ".bn_atten"));
    return backend.mul(feature, gate);
}

void observe(SemanticSegmenter2DObservation* observation,
             const std::string& name, const Tensor& value) {
    if (observation) observation->tensors.emplace(name, value);
}

void validate_bisenet_graph(const Json& graph) {
    const Json& config = graph.at("config");
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "semantic_segmenter_2d" &&
            graph.at("entry_point").string() == "execute" &&
            config.at("input_height").integer() == 512 &&
            config.at("input_width").integer() == 512 &&
            config.at("class_count").integer() == 19 &&
            config.at("bilinear_align_corners").boolean(),
            "unsupported semantic segmenter graph");
}

std::vector<int64_t> json_integers(const Json& value) {
    std::vector<int64_t> result;
    for (const Json& item : value.array()) result.push_back(item.integer());
    return result;
}

Tensor nhwc_to_nchw(Backend& backend, const Tensor& value) {
    require(value.ndim() == 4, "NHWC tensor must have rank four");
    return backend.permute(value, {0, 3, 1, 2});
}

Tensor nchw_to_nhwc(Backend& backend, const Tensor& value) {
    require(value.ndim() == 4, "NCHW tensor must have rank four");
    return backend.permute(value, {0, 2, 3, 1});
}

void validate_xenoformer_graph(const Json& graph) {
    const Json& config = graph.at("config");
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "semantic_segmenter_2d" &&
            graph.at("entry_point").string() == "execute" &&
            config.at("dtype").string() == "float32" &&
            config.at("architecture").string() == "xenoformer_fpn" &&
            config.at("input_height").integer() == 256 &&
            config.at("input_width").integer() == 256 &&
            config.at("class_count").integer() == 6 &&
            config.at("tensor_count").integer() == 373 &&
            config.at("operator_count").integer() == 175 &&
            graph.at("tensor_shapes").array().size() == 373 &&
            graph.at("operators").array().size() == 175,
            "unsupported xenoformer semantic segmenter graph");
}

Tensor execute_xenoformer_graph(Backend& backend, const WeightMap& weights,
        const Json& graph, const Tensor& image,
        SemanticSegmenter2DObservation* observation) {
    validate_xenoformer_graph(graph);
    require(image.shape() == std::vector<int64_t>({1,3,256,256}) &&
            image.dtype() == DType::F32,
            "xenoformer semantic segmenter input must be [1,3,256,256] FP32");
    std::vector<Tensor> tensors(373);
    tensors[0] = nchw_to_nhwc(backend, image);
    const auto& shapes = graph.at("tensor_shapes").array();
    const auto& integer_constants = graph.at("integer_constants").object();
    auto dynamic = [&](int64_t index) -> const Tensor& {
        require(index >= 0 && index < static_cast<int64_t>(tensors.size()) &&
                tensors[static_cast<size_t>(index)].defined(),
                "semantic segmenter graph references unavailable tensor");
        return tensors[static_cast<size_t>(index)];
    };
    auto weight = [&](int64_t index) -> const Tensor& {
        return weights.at("tensor." + std::to_string(index));
    };
    auto i32 = [&](int64_t index) {
        const auto found = integer_constants.find(std::to_string(index));
        require(found != integer_constants.end(),
                "semantic segmenter graph integer constant is unavailable");
        return json_integers(found->second);
    };
    auto shape = [&](int64_t index) {
        require(index >= 0 && index < static_cast<int64_t>(shapes.size()),
                "semantic segmenter graph shape index invalid");
        return json_integers(shapes[static_cast<size_t>(index)]);
    };
    auto observe_nhwc = [&](const std::string& name, int64_t index) {
        if (observation) observation->tensors.emplace(
            name, nhwc_to_nchw(backend, dynamic(index)));
    };

    for (size_t operation_index = 0;
         operation_index < graph.at("operators").array().size(); ++operation_index) {
        const Json& operation = graph.at("operators").array()[operation_index];
        const std::string& type = operation.at("type").string();
        const auto inputs = json_integers(operation.at("inputs"));
        const auto outputs = json_integers(operation.at("outputs"));
        const auto options = json_integers(operation.at("options"));
        require(outputs.size() == 1, "semantic segmenter operator output drift");
        Tensor output;
        if (type == "CONV_2D" || type == "DEPTHWISE_CONV_2D") {
            require(inputs.size() == 3 && options.size() ==
                    (type == "CONV_2D" ? 6 : 7),
                    "semantic segmenter convolution contract drift");
            Tensor input = nhwc_to_nchw(backend, dynamic(inputs[0]));
            const Tensor& kernel = weight(inputs[1]);
            const Tensor* bias = inputs[2] < 0 ? nullptr : &weight(inputs[2]);
            const int stride_height = static_cast<int>(options[1]);
            const int stride_width = static_cast<int>(options[2]);
            const int dilation_height = type == "CONV_2D"
                ? static_cast<int>(options[3]) : static_cast<int>(options[4]);
            const int dilation_width = type == "CONV_2D"
                ? static_cast<int>(options[4]) : static_cast<int>(options[5]);
            if (options[0] == 0) {
                const auto desired = shape(outputs[0]);
                const int64_t effective_height =
                    dilation_height * (kernel.dim(2) - 1) + 1;
                const int64_t effective_width =
                    dilation_width * (kernel.dim(3) - 1) + 1;
                const int64_t total_height = std::max<int64_t>(0,
                    (desired[1] - 1) * stride_height + effective_height - input.dim(2));
                const int64_t total_width = std::max<int64_t>(0,
                    (desired[2] - 1) * stride_width + effective_width - input.dim(3));
                input = backend.pad(input,{total_width/2,total_width-total_width/2,
                                           total_height/2,total_height-total_height/2});
            } else require(options[0] == 1,
                           "semantic segmenter convolution padding is unsupported");
            const int groups = type == "DEPTHWISE_CONV_2D"
                ? static_cast<int>(input.dim(1)) : 1;
            output = backend.conv2d(input,kernel,bias,
                {stride_height,stride_width},{0,0},
                {dilation_height,dilation_width},groups);
            const int64_t activation = type == "CONV_2D" ? options[5] : options[6];
            require(activation == 0 || activation == 1,
                    "semantic segmenter fused activation drift");
            if (activation == 1) output = backend.activation(output,Activation::Relu);
            output = nchw_to_nhwc(backend,output);
        } else if (type == "ADD" || type == "MUL") {
            require(inputs.size() == 2 && options == std::vector<int64_t>{0},
                    "semantic segmenter binary contract drift");
            auto operand = [&](int64_t index) -> const Tensor& {
                const Tensor* constant = weights.find("tensor."+std::to_string(index));
                return constant ? *constant : dynamic(index);
            };
            output = type == "ADD" ? backend.add(operand(inputs[0]),operand(inputs[1]))
                                   : backend.mul(operand(inputs[0]),operand(inputs[1]));
        } else if (type == "RESHAPE") {
            require(inputs.size() == 2, "semantic segmenter reshape contract drift");
            output = backend.reshape(dynamic(inputs[0]),shape(outputs[0]));
        } else if (type == "TRANSPOSE") {
            require(inputs.size() == 2 && options.empty(),
                    "semantic segmenter transpose contract drift");
            output = backend.permute(dynamic(inputs[0]),i32(inputs[1]));
        } else if (type == "SOFTMAX") {
            require(inputs.size() == 1 && options == std::vector<int64_t>{0x3f800000},
                    "semantic segmenter softmax option drift");
            output = backend.softmax(dynamic(inputs[0]),-1);
        } else if (type == "SUM") {
            require(inputs.size() == 2 && options == std::vector<int64_t>{1},
                    "semantic segmenter reduction option drift");
            const auto axes = i32(inputs[1]);
            require(axes.size() == 1, "semantic segmenter reduction axis drift");
            output = backend.reduce_sum(dynamic(inputs[0]),axes[0],true);
        } else if (type == "RESIZE_BILINEAR" ||
                   type == "RESIZE_NEAREST_NEIGHBOR") {
            require(inputs.size() == 2 && options == std::vector<int64_t>({0,1}),
                    "semantic segmenter resize option drift");
            const auto desired = shape(outputs[0]);
            Tensor input = nhwc_to_nchw(backend,dynamic(inputs[0]));
            if (type == "RESIZE_BILINEAR")
                output = backend.interpolate_bilinear_2d(
                    input,desired[1],desired[2],false);
            else
                output = backend.interpolate_nearest(input,
                    {static_cast<double>(desired[1])/input.dim(2),
                     static_cast<double>(desired[2])/input.dim(3)});
            output = nchw_to_nhwc(backend,output);
        } else if (type == "TRANSPOSE_CONV") {
            require(inputs.size() == 4 && options == std::vector<int64_t>({0,2,2,0}),
                    "semantic segmenter transposed convolution option drift");
            Tensor input = nhwc_to_nchw(backend,dynamic(inputs[2]));
            output = backend.conv_transpose2d(input,weight(inputs[1]),&weight(inputs[3]),
                                              {2,2},{0,0},{0,0},{1,1},1);
            output = nchw_to_nhwc(backend,output);
        } else {
            require(false,"unsupported semantic segmenter operator");
        }
        require(output.shape() == shape(outputs[0]),
                "semantic segmenter operator shape propagation drift");
        tensors[static_cast<size_t>(outputs[0])] = std::move(output);
        if (operation_index == 0) observe_nhwc("early_feature",outputs[0]);
        if (operation_index == 11) observe_nhwc("intermediate_feature",outputs[0]);
        if (operation_index == 145) observe_nhwc("decoder_feature",outputs[0]);
        if (operation_index == 172) observe_nhwc("pre_final_feature",outputs[0]);
    }
    Tensor logits = nhwc_to_nchw(backend,dynamic(372));
    if (observation) observation->tensors.emplace("semantic_logits",logits);
    return logits;
}

}  // namespace

SemanticSegmenter2DResult SemanticSegmenter2DComponentExecutor::execute(
        const Json& graph, const Tensor& image,
        SemanticSegmenter2DObservation* observation) {
    const Json* architecture = graph.at("config").find("architecture");
    if (architecture && architecture->string() == "xenoformer_fpn")
        return {execute_xenoformer_graph(backend_,weights_,graph,image,observation)};
    validate_bisenet_graph(graph);
    require(image.ndim() == 4 && image.dtype() == DType::F32 &&
            image.dim(1) == 3 && image.dim(2) == 512 && image.dim(3) == 512,
            "semantic segmenter input must be [B,3,512,512] FP32");

    Tensor hidden = conv(backend_, weights_, image, "cp.resnet.conv1", 2, 3);
    hidden = backend_.activation(batch_norm(backend_, weights_, hidden,
        "cp.resnet.bn1"), Activation::Relu);
    hidden = backend_.max_pool2d(hidden, {3, 3}, {2, 2}, {1, 1});
    hidden = basic_block(backend_, weights_, hidden, "cp.resnet.layer1.0", 1, false);
    hidden = basic_block(backend_, weights_, hidden, "cp.resnet.layer1.1", 1, false);
    hidden = basic_block(backend_, weights_, hidden, "cp.resnet.layer2.0", 2, true);
    Tensor feat8 = basic_block(backend_, weights_, hidden, "cp.resnet.layer2.1", 1, false);
    observe(observation, "early_backbone", feat8);
    hidden = basic_block(backend_, weights_, feat8, "cp.resnet.layer3.0", 2, true);
    Tensor feat16 = basic_block(backend_, weights_, hidden, "cp.resnet.layer3.1", 1, false);
    hidden = basic_block(backend_, weights_, feat16, "cp.resnet.layer4.0", 2, true);
    Tensor feat32 = basic_block(backend_, weights_, hidden, "cp.resnet.layer4.1", 1, false);
    observe(observation, "context_feature", feat32);

    Tensor average = conv_bn_relu(backend_, weights_, global_average(backend_, feat32),
                                  "cp.conv_avg");
    average = backend_.interpolate_nearest(average, {
        static_cast<double>(feat32.dim(2)), static_cast<double>(feat32.dim(3))});
    Tensor arm32 = attention_refinement(backend_, weights_, feat32, "cp.arm32");
    observe(observation, "attention_refinement", arm32);
    Tensor cp16 = backend_.interpolate_nearest(backend_.add(arm32, average), {2.0, 2.0});
    cp16 = conv_bn_relu(backend_, weights_, cp16, "cp.conv_head32", 1, 1);
    Tensor arm16 = attention_refinement(backend_, weights_, feat16, "cp.arm16");
    Tensor cp8 = backend_.interpolate_nearest(backend_.add(arm16, cp16), {2.0, 2.0});
    cp8 = conv_bn_relu(backend_, weights_, cp8, "cp.conv_head16", 1, 1);

    Tensor fused = conv_bn_relu(backend_, weights_,
        backend_.concat({feat8, cp8}, 1), "ffm.convblk");
    Tensor gate = global_average(backend_, fused);
    gate = backend_.activation(conv(backend_, weights_, gate, "ffm.conv1"),
                               Activation::Relu);
    gate = sigmoid(backend_, conv(backend_, weights_, gate, "ffm.conv2"));
    fused = backend_.add(fused, backend_.mul(fused, gate));
    observe(observation, "feature_fusion", fused);

    Tensor logits = conv_bn_relu(backend_, weights_, fused, "conv_out.conv", 1, 1);
    logits = conv(backend_, weights_, logits, "conv_out.conv_out");
    logits = backend_.interpolate_bilinear_2d(logits, 512, 512, true);
    observe(observation, "logits", logits);
    return {logits};
}

}  // namespace vrhino
