#include "vrhino/face_mask.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>

#include "vrhino/error.h"

namespace vrhino {
namespace {

std::vector<uint8_t> crop_bgr_to_rgb(
        const uint8_t* source, int32_t height, int32_t width,
        const std::array<int32_t, 4>& box) {
    const int32_t output_width = box[2] - box[0];
    const int32_t output_height = box[3] - box[1];
    std::vector<uint8_t> output(
        static_cast<size_t>(output_width) * output_height * 3, 0);
    for (int32_t y = 0; y < output_height; ++y) {
        const int32_t source_y = box[1] + y;
        if (source_y < 0 || source_y >= height) continue;
        for (int32_t x = 0; x < output_width; ++x) {
            const int32_t source_x = box[0] + x;
            if (source_x < 0 || source_x >= width) continue;
            const size_t source_offset =
                (static_cast<size_t>(source_y) * width + source_x) * 3;
            const size_t destination =
                (static_cast<size_t>(y) * output_width + x) * 3;
            output[destination] = source[source_offset + 2];
            output[destination + 1] = source[source_offset + 1];
            output[destination + 2] = source[source_offset];
        }
    }
    return output;
}

uint8_t opencv_bilinear_rgb_zero(const uint8_t* source,int32_t height,
        int32_t width,int channel,double x,double y){int32_t ix=static_cast<int32_t>(std::floor(x)),iy=static_cast<int32_t>(std::floor(y));double fx=std::floor((x-ix)*32.0+0.5)/32.0,fy=std::floor((y-iy)*32.0+0.5)/32.0;if(fx>=1.0){++ix;fx=0.0;}if(fy>=1.0){++iy;fy=0.0;}double value=0.0;for(int dy=0;dy<=1;++dy)for(int dx=0;dx<=1;++dx){const int32_t sx=ix+dx,sy=iy+dy;if(sx<0||sx>=width||sy<0||sy>=height)continue;value+=source[(static_cast<size_t>(sy)*width+sx)*3+channel]*(dx?fx:1.0-fx)*(dy?fy:1.0-fy);}return static_cast<uint8_t>(std::clamp(std::floor(value+0.5),0.0,255.0));}

std::vector<uint8_t> resize_bilinear_rgb_u8(
        const std::vector<uint8_t>& source, int32_t input_height,
        int32_t input_width, int32_t output_height, int32_t output_width) {
    // The frozen parser path uses Pillow Image.BILINEAR.  Pillow applies a
    // separable filter and rounds the 8-bit horizontal intermediate before the
    // vertical pass; a direct 2-D bilinear expression is not byte-equivalent.
    struct AxisCoefficient {
        int32_t first;
        std::vector<double> weights;
    };
    const auto coefficients = [](int32_t input_size, int32_t output_size) {
        std::vector<AxisCoefficient> result(static_cast<size_t>(output_size));
        const double scale = static_cast<double>(input_size) / output_size;
        const double filter_scale = std::max(1.0, scale);
        const double support = filter_scale;
        for (int32_t output = 0; output < output_size; ++output) {
            const double center = (output + 0.5) * scale;
            const int32_t first = std::max<int32_t>(0,
                static_cast<int32_t>(center - support + 0.5));
            const int32_t last = std::min<int32_t>(input_size,
                static_cast<int32_t>(center + support + 0.5));
            auto& item = result[static_cast<size_t>(output)];
            item.first = first;
            item.weights.resize(static_cast<size_t>(last - first));
            double total = 0.0;
            for (int32_t input = first; input < last; ++input) {
                const double weight = std::max(0.0,
                    1.0 - std::abs((input + 0.5 - center) / filter_scale));
                item.weights[static_cast<size_t>(input - first)] = weight;
                total += weight;
            }
            for (double& weight : item.weights) weight /= total;
        }
        return result;
    };
    const auto horizontal = coefficients(input_width, output_width);
    const auto vertical = coefficients(input_height, output_height);
    std::vector<uint8_t> intermediate(
        static_cast<size_t>(input_height) * output_width * 3);
    for (int32_t y = 0; y < input_height; ++y)
        for (int32_t x = 0; x < output_width; ++x)
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto& xc = horizontal[static_cast<size_t>(x)];
                double sum = 0.0;
                for (size_t index = 0; index < xc.weights.size(); ++index)
                    sum += source[(static_cast<size_t>(y) * input_width +
                                   xc.first + index) * 3 + channel] * xc.weights[index];
                intermediate[(static_cast<size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<double>(std::nearbyint(sum), 0, 255));
            }
    std::vector<uint8_t> output(
        static_cast<size_t>(output_height) * output_width * 3);
    for (int32_t y = 0; y < output_height; ++y)
        for (int32_t x = 0; x < output_width; ++x)
            for (int32_t channel = 0; channel < 3; ++channel) {
        const auto& yc = vertical[static_cast<size_t>(y)];
                double sum = 0.0;
                for (size_t index = 0; index < yc.weights.size(); ++index)
                    sum += intermediate[((static_cast<size_t>(yc.first) + index) *
                                         output_width + x) * 3 + channel] * yc.weights[index];
                output[(static_cast<size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<double>(std::nearbyint(sum), 0, 255));
            }
    return output;
}

std::vector<uint8_t> dilate_binary(
        const std::vector<uint8_t>& input, int32_t height, int32_t width,
        const std::vector<uint8_t>& kernel, int32_t kh, int32_t kw) {
    std::vector<uint8_t> output(input.size(), 0);
    const int32_t ay = kh / 2, ax = kw / 2;
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        bool set = false;
        for (int32_t ky = 0; ky < kh && !set; ++ky)
            for (int32_t kx = 0; kx < kw; ++kx) {
                if (!kernel[static_cast<size_t>(ky) * kw + kx]) continue;
                const int32_t iy = y + ky - ay, ix = x + kx - ax;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width &&
                    input[static_cast<size_t>(iy) * width + ix] == 255) {
                    set = true; break;
                }
            }
        output[static_cast<size_t>(y) * width + x] = set ? 255 : 0;
    }
    return output;
}

std::vector<uint8_t> erode_binary(
        const std::vector<uint8_t>& input, int32_t height, int32_t width,
        const std::vector<uint8_t>& kernel, int32_t kh, int32_t kw) {
    std::vector<uint8_t> output(input.size(), 0);
    const int32_t ay = kh / 2, ax = kw / 2;
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        bool set = true;
        for (int32_t ky = 0; ky < kh && set; ++ky)
            for (int32_t kx = 0; kx < kw; ++kx) {
                if (!kernel[static_cast<size_t>(ky) * kw + kx]) continue;
                const int32_t iy = y + ky - ay, ix = x + kx - ax;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width &&
                    input[static_cast<size_t>(iy) * width + ix] != 255) {
                    set = false; break;
                }
            }
        output[static_cast<size_t>(y) * width + x] = set ? 255 : 0;
    }
    return output;
}

std::vector<uint8_t> gaussian_blur_u8(
        const std::vector<uint8_t>& input, int32_t height, int32_t width,
        int32_t kernel_size) {
    const int32_t radius = kernel_size / 2;
    const double sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1.0) + 0.8;
    std::vector<double> kernel(kernel_size), temporary(input.size());
    double total = 0.0;
    for (int32_t index = -radius; index <= radius; ++index) {
        const double value = std::exp(-(index * index) / (2.0 * sigma * sigma));
        kernel[index + radius] = value; total += value;
    }
    for (double& value : kernel) value /= total;
    auto reflect101 = [](int32_t coordinate, int32_t length) {
        while (coordinate < 0 || coordinate >= length) {
            if (coordinate < 0) coordinate = -coordinate;
            else coordinate = 2 * length - coordinate - 2;
        }
        return coordinate;
    };
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        double sum = 0.0;
        for (int32_t k = -radius; k <= radius; ++k)
            sum += input[static_cast<size_t>(y) * width + reflect101(x + k, width)] *
                   kernel[k + radius];
        temporary[static_cast<size_t>(y) * width + x] = sum;
    }
    std::vector<uint8_t> output(input.size());
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        double sum = 0.0;
        for (int32_t k = -radius; k <= radius; ++k)
            sum += temporary[static_cast<size_t>(reflect101(y + k, height)) * width + x] *
                   kernel[k + radius];
        output[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(
            std::clamp<long>(std::lround(sum), 0, 255));
    }
    return output;
}

double bicubic_weight(double value) {
    value = std::abs(value);
    if (value < 1.0)
        return ((1.5 * value - 2.5) * value) * value + 1.0;
    if (value < 2.0)
        return ((-0.5 * value + 2.5) * value - 4.0) * value + 2.0;
    return 0.0;
}

struct ResizeCoefficients {
    int32_t first = 0;
    std::vector<double> weights;
};

std::vector<ResizeCoefficients> bicubic_coefficients(
        int32_t input_size, int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = 2.0 * filter_scale;
    std::vector<ResizeCoefficients> output(output_size);
    for (int32_t coordinate = 0; coordinate < output_size; ++coordinate) {
        const double center = (coordinate + 0.5) * scale;
        const int32_t first = std::max<int32_t>(0,
            static_cast<int32_t>(center - support + 0.5));
        const int32_t last = std::min<int32_t>(input_size,
            static_cast<int32_t>(center + support + 0.5));
        output[coordinate].first = first;
        output[coordinate].weights.resize(last - first);
        double total = 0.0;
        for (int32_t source = first; source < last; ++source) {
            const double weight = bicubic_weight(
                (source + 0.5 - center) / filter_scale);
            output[coordinate].weights[source - first] = weight;
            total += weight;
        }
        for (double& weight : output[coordinate].weights) weight /= total;
    }
    return output;
}

// Pillow's default resize for an L-mode image is separable bicubic. Its 8-bit
// implementation rounds after the horizontal pass; preserving that boundary
// is required for deterministic mask morphology.
std::vector<uint8_t> resize_bicubic_gray_u8(
        const std::vector<uint8_t>& input, int32_t input_height,
        int32_t input_width, int32_t output_height, int32_t output_width) {
    const auto horizontal_coefficients =
        bicubic_coefficients(input_width, output_width);
    const auto vertical_coefficients =
        bicubic_coefficients(input_height, output_height);
    std::vector<uint8_t> horizontal(
        static_cast<size_t>(input_height) * output_width);
    for (int32_t y = 0; y < input_height; ++y)
        for (int32_t x = 0; x < output_width; ++x) {
            const auto& coefficients = horizontal_coefficients[x];
            double sum = 0.0;
            for (size_t index = 0; index < coefficients.weights.size(); ++index)
                sum += input[static_cast<size_t>(y) * input_width +
                             coefficients.first + index] * coefficients.weights[index];
            horizontal[static_cast<size_t>(y) * output_width + x] =
                static_cast<uint8_t>(std::clamp<double>(std::nearbyint(sum), 0, 255));
        }
    std::vector<uint8_t> output(
        static_cast<size_t>(output_height) * output_width);
    for (int32_t y = 0; y < output_height; ++y)
        for (int32_t x = 0; x < output_width; ++x) {
            const auto& coefficients = vertical_coefficients[y];
            double sum = 0.0;
            for (size_t index = 0; index < coefficients.weights.size(); ++index)
                sum += horizontal[(static_cast<size_t>(coefficients.first) + index) *
                                  output_width + x] * coefficients.weights[index];
            output[static_cast<size_t>(y) * output_width + x] =
                static_cast<uint8_t>(std::clamp<double>(std::nearbyint(sum), 0, 255));
        }
    return output;
}

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

bool point_in_polygon(double x, double y, const std::vector<Point2D>& polygon) {
    bool inside = false;
    for (size_t current = 0, previous = polygon.size() - 1;
         current < polygon.size(); previous = current++) {
        const auto& a = polygon[current];
        const auto& b = polygon[previous];
        const bool crosses = (a.y > y) != (b.y > y);
        if (crosses && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}

std::vector<uint8_t> rasterize_polygon(
        int32_t height, int32_t width, const std::vector<Point2D>& polygon) {
    std::vector<uint8_t> output(static_cast<size_t>(height) * width, 0);
    if (polygon.size() < 3) return output;
    for (int32_t y = 0; y < height; ++y)
        for (int32_t x = 0; x < width; ++x)
            if (point_in_polygon(x + 0.5, y + 0.5, polygon))
                output[static_cast<size_t>(y) * width + x] = 255;
    return output;
}

std::vector<uint8_t> ellipse_kernel(int32_t size) {
    require(size > 0 && (size & 1) == 1, "morphology kernel must be positive odd");
    std::vector<uint8_t> kernel(static_cast<size_t>(size) * size, 0);
    const double radius = size / 2.0;
    for (int32_t y = 0; y < size; ++y)
        for (int32_t x = 0; x < size; ++x) {
            const double dx = (x - radius) / (radius + 0.5);
            const double dy = (y - radius) / (radius + 0.5);
            if (dx * dx + dy * dy <= 1.0)
                kernel[static_cast<size_t>(y) * size + x] = 1;
        }
    return kernel;
}

std::vector<uint8_t> retain_largest_component(
        const std::vector<uint8_t>& input, int32_t height, int32_t width) {
    std::vector<int32_t> component(input.size(), -1);
    std::vector<size_t> sizes;
    constexpr int32_t offsets[8][2] = {
        {-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        const size_t start = static_cast<size_t>(y) * width + x;
        if (!input[start] || component[start] >= 0) continue;
        const int32_t identity = static_cast<int32_t>(sizes.size());
        sizes.push_back(0);
        std::queue<std::pair<int32_t,int32_t>> pending;
        pending.push({x,y}); component[start] = identity;
        while (!pending.empty()) {
            const auto [current_x,current_y] = pending.front(); pending.pop();
            ++sizes[static_cast<size_t>(identity)];
            for (const auto& offset : offsets) {
                const int32_t nx = current_x + offset[0], ny = current_y + offset[1];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                const size_t index = static_cast<size_t>(ny) * width + nx;
                if (input[index] && component[index] < 0) {
                    component[index] = identity; pending.push({nx,ny});
                }
            }
        }
    }
    std::vector<uint8_t> output(input.size(), 0);
    if (sizes.empty()) return output;
    const int32_t selected = static_cast<int32_t>(
        std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end())));
    for (size_t index = 0; index < output.size(); ++index)
        if (component[index] == selected) output[index] = 255;
    return output;
}

}  // namespace

FaceParserPreparation preprocess_face_parser_bgr_u8(
        const uint8_t* source, int32_t height, int32_t width,
        const std::array<int32_t, 4>& face_box, double expand) {
    require(source && height > 0 && width > 0 && expand > 0.0 &&
            face_box[2] > face_box[0] && face_box[3] > face_box[1],
            "invalid face parser preprocessing contract");
    const int32_t center_x = (face_box[0] + face_box[2]) / 2;
    const int32_t center_y = (face_box[1] + face_box[3]) / 2;
    const int32_t half = static_cast<int32_t>(
        (std::max(face_box[2] - face_box[0], face_box[3] - face_box[1]) / 2) * expand);
    FaceParserPreparation result;
    result.crop_box = {center_x - half, center_y - half,
                       center_x + half, center_y + half};
    const int32_t crop_width = result.crop_box[2] - result.crop_box[0];
    const int32_t crop_height = result.crop_box[3] - result.crop_box[1];
    auto crop = crop_bgr_to_rgb(source, height, width, result.crop_box);
    result.resized_rgb = resize_bilinear_rgb_u8(crop, crop_height, crop_width,
                                                 512, 512);
    result.normalized_nchw = Tensor::host({1, 3, 512, 512}, DType::F32);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float standard_deviation[3] = {0.229f, 0.224f, 0.225f};
    float* output = result.normalized_nchw.data_as<float>();
    for (int32_t channel = 0; channel < 3; ++channel)
        for (int32_t y = 0; y < 512; ++y) for (int32_t x = 0; x < 512; ++x) {
            const float value = result.resized_rgb[
                (static_cast<size_t>(y) * 512 + x) * 3 + channel] / 255.0f;
            output[(static_cast<size_t>(channel) * 512 + y) * 512 + x] =
                (value - mean[channel]) / standard_deviation[channel];
        }
    return result;
}

SemanticSegmenterPreparation preprocess_semantic_segmenter_rgb_u8(
        const uint8_t* source,int32_t height,int32_t width,
        const std::array<int32_t,4>& crop_box){require(source&&height>0&&width>0&&crop_box[2]>crop_box[0]&&crop_box[3]>crop_box[1],"invalid semantic segmenter preprocessing contract");SemanticSegmenterPreparation result;result.crop_box=crop_box;result.resized_rgb.resize(256*256*3);result.normalized_nchw=Tensor::host({1,3,256,256},DType::F32);float* normalized=result.normalized_nchw.data_as<float>();const double crop_width=crop_box[2]-crop_box[0],crop_height=crop_box[3]-crop_box[1];for(int32_t y=0;y<256;++y)for(int32_t x=0;x<256;++x){const double source_x=crop_box[0]+(x+0.5)*crop_width/256.0-0.5,source_y=crop_box[1]+(y+0.5)*crop_height/256.0-0.5;for(int channel=0;channel<3;++channel){const uint8_t value=opencv_bilinear_rgb_zero(source,height,width,channel,source_x,source_y);result.resized_rgb[(static_cast<size_t>(y)*256+x)*3+channel]=value;normalized[(static_cast<size_t>(channel)*256+y)*256+x]=value/255.0f;}}return result;}

SemanticLabelMap semantic_argmax_first(const Tensor& logits) {
    require(logits.device().is_host() && logits.dtype() == DType::F32 &&
            logits.ndim() == 4 && logits.dim(0) == 1 && logits.dim(1) <= 255,
            "semantic argmax requires host [1,C,H,W] FP32 logits");
    SemanticLabelMap result{static_cast<int32_t>(logits.dim(3)),
                            static_cast<int32_t>(logits.dim(2)), {}};
    result.labels.resize(static_cast<size_t>(result.width) * result.height);
    const int64_t spatial = logits.dim(2) * logits.dim(3);
    for (int64_t position = 0; position < spatial; ++position) {
        float best = -std::numeric_limits<float>::infinity(); uint8_t selected = 0;
        for (int64_t channel = 0; channel < logits.dim(1); ++channel) {
            const float value = logits.data_as<float>()[channel * spatial + position];
            if (value > best) { best = value; selected = static_cast<uint8_t>(channel); }
        }
        result.labels[position] = selected;
    }
    return result;
}

AlphaMask build_landmark_constrained_alpha_mask(
        const SemanticLabelMap& labels,
        const std::vector<std::array<int32_t, 2>>& facial_landmarks,
        const std::array<int32_t, 4>& crop_box,
        const LandmarkConstrainedMaskConfig& config,
        LandmarkConstrainedMaskObservation* observation) {
    require(labels.width > 0 && labels.height > 0 &&
            labels.labels.size() == static_cast<size_t>(labels.width) * labels.height &&
            crop_box[2] > crop_box[0] && crop_box[3] > crop_box[1] &&
            config.jaw_first >= 0 && config.jaw_last >= config.jaw_first &&
            config.nose_left >= 0 && config.nose_center >= 0 &&
            config.nose_right >= 0 && config.mouth_first >= 0 &&
            config.mouth_last >= config.mouth_first &&
            config.close_kernel > 0 && (config.close_kernel & 1) == 1 &&
            config.mouth_dilation_kernel > 0 &&
            (config.mouth_dilation_kernel & 1) == 1 &&
            config.blur_kernel > 0 && (config.blur_kernel & 1) == 1,
            "invalid landmark-constrained mask contract");
    const int32_t maximum_index = std::max(
        {config.jaw_last, config.nose_left, config.nose_center,
         config.nose_right, config.mouth_last});
    require(static_cast<int32_t>(facial_landmarks.size()) > maximum_index,
            "landmark-constrained mask requires configured facial landmarks");

    const auto geometry_started = std::chrono::steady_clock::now();
    const double crop_width = crop_box[2] - crop_box[0];
    const double crop_height = crop_box[3] - crop_box[1];
    const auto transform = [&](int32_t index) {
        const auto& point = facial_landmarks[static_cast<size_t>(index)];
        // Exact inverse of the half-pixel crop sampling used by preprocessing.
        return Point2D{
            (point[0] + 0.5 - crop_box[0]) * labels.width / crop_width - 0.5,
            (point[1] + 0.5 - crop_box[1]) * labels.height / crop_height - 0.5};
    };

    std::vector<Point2D> lower_face;
    for (int32_t index = config.jaw_first; index <= config.jaw_last; ++index)
        lower_face.push_back(transform(index));
    lower_face.push_back(transform(config.nose_right));
    lower_face.push_back(transform(config.nose_center));
    lower_face.push_back(transform(config.nose_left));
    auto constraint = rasterize_polygon(labels.height, labels.width, lower_face);

    std::vector<Point2D> outer_mouth;
    for (int32_t index = config.mouth_first; index <= config.mouth_last; ++index)
        outer_mouth.push_back(transform(index));
    auto mouth = rasterize_polygon(labels.height, labels.width, outer_mouth);
    const auto mouth_kernel = ellipse_kernel(config.mouth_dilation_kernel);
    mouth = dilate_binary(mouth, labels.height, labels.width, mouth_kernel,
                          config.mouth_dilation_kernel,
                          config.mouth_dilation_kernel);
    const auto morphology_started = std::chrono::steady_clock::now();

    std::vector<uint8_t> seed(labels.labels.size(), 0);
    for (size_t index = 0; index < seed.size(); ++index) {
        const uint8_t label = labels.labels[index];
        const bool semantic = label == config.face_skin_label ||
            (config.include_body_skin_inside_constraint &&
             label == config.body_skin_label);
        if (constraint[index] && (semantic || mouth[index])) seed[index] = 255;
    }

    const auto close_kernel = ellipse_kernel(config.close_kernel);
    auto selected = dilate_binary(seed, labels.height, labels.width, close_kernel,
                                  config.close_kernel, config.close_kernel);
    selected = erode_binary(selected, labels.height, labels.width, close_kernel,
                            config.close_kernel, config.close_kernel);
    for (size_t index = 0; index < selected.size(); ++index) {
        const uint8_t label = labels.labels[index];
        const bool semantic = label == config.face_skin_label ||
            (config.include_body_skin_inside_constraint &&
             label == config.body_skin_label);
        if (!constraint[index] || (!semantic && !mouth[index])) selected[index] = 0;
    }
    selected = retain_largest_component(selected, labels.height, labels.width);

    const auto resize_started = std::chrono::steady_clock::now();

    const int32_t output_width = crop_box[2] - crop_box[0];
    const int32_t output_height = crop_box[3] - crop_box[1];
    auto resized = resize_bicubic_gray_u8(
        selected, labels.height, labels.width, output_height, output_width);
    auto blurred = gaussian_blur_u8(resized, output_height, output_width,
                                    config.blur_kernel);
    const auto completed = std::chrono::steady_clock::now();
    AlphaMask result{output_width, output_height, std::move(blurred),
                     "expanded_face_crop_pixels"};
    if (observation) {
        observation->geometry_constraint = std::move(constraint);
        observation->semantic_seed = std::move(seed);
        observation->mouth_restoration = std::move(mouth);
        observation->selected = std::move(selected);
        observation->geometry_ms = std::chrono::duration<double,std::milli>(
            morphology_started - geometry_started).count();
        observation->morphology_ms = std::chrono::duration<double,std::milli>(
            resize_started - morphology_started).count();
        observation->resize_blur_ms = std::chrono::duration<double,std::milli>(
            completed - resize_started).count();
    }
    return result;
}

AlphaMask build_musetalk_jaw_alpha_mask(
        const SemanticLabelMap& labels,
        const std::array<int32_t, 4>& face_box,
        const std::array<int32_t, 4>& crop_box,
        int32_t left_cheek_width, int32_t right_cheek_width,
        double lower_half_ratio, JawMaskObservation* observation) {
    require(labels.width == 512 && labels.height == 512 &&
            labels.labels.size() == 512 * 512 &&
            left_cheek_width >= 0 && right_cheek_width >= 0 &&
            lower_half_ratio >= 0.0 && lower_half_ratio <= 1.0,
            "invalid jaw alpha-mask contract");
    std::vector<uint8_t> face_region(512 * 512);
    for (size_t i = 0; i < face_region.size(); ++i)
        face_region[i] = labels.labels[i] == 1 ? 255 : 0;
    std::vector<uint8_t> cone(33 * 33, 0);
    for (int row = 10; row < 21; ++row) {
        const int width = 2 * (row - 10) + 1, begin = 16 - width / 2;
        std::fill(cone.begin() + row * 33 + begin,
                  cone.begin() + row * 33 + begin + width, 1);
    }
    for (int row = 21; row < 33; ++row)
        std::fill(cone.begin() + row * 33 + 6,
                  cone.begin() + row * 33 + 27, 1);
    auto dilated = dilate_binary(face_region, 512, 512, cone, 33, 33);
    std::vector<uint8_t> ellipse(3 * 35, 0);
    ellipse[17] = 1;
    std::fill(ellipse.begin() + 35, ellipse.begin() + 70, 1);
    ellipse[70 + 17] = 1;
    auto eroded = erode_binary(dilated, 512, 512, ellipse, 3, 35);
    eroded = erode_binary(eroded, 512, 512, ellipse, 3, 35);
    std::vector<uint8_t> selected(512 * 512, 0);
    for (int y = 0; y < 512; ++y) for (int x = 0; x < 512; ++x) {
        const size_t index = static_cast<size_t>(y) * 512 + x;
        const bool cheek = x <= 256 - left_cheek_width ||
                           x >= 256 + right_cheek_width;
        const bool region = cheek ? eroded[index] == 255 : dilated[index] == 255;
        if ((region && labels.labels[index] != 10) ||
            labels.labels[index] == 11 || labels.labels[index] == 12 ||
            labels.labels[index] == 13) selected[index] = 255;
    }

    const int32_t width = crop_box[2] - crop_box[0];
    const int32_t height = crop_box[3] - crop_box[1];
    require(width > 0 && height > 0, "invalid expanded parser crop");
    std::vector<uint8_t> resized = resize_bicubic_gray_u8(
        selected, 512, 512, height, width);
    std::vector<uint8_t> lower(static_cast<size_t>(width) * height, 0);
    const int32_t left = face_box[0] - crop_box[0];
    const int32_t top = face_box[1] - crop_box[1];
    const int32_t right = face_box[2] - crop_box[0];
    const int32_t bottom = face_box[3] - crop_box[1];
    const int32_t boundary = static_cast<int32_t>(height * lower_half_ratio);
    for (int32_t y = std::max(top, boundary); y < std::min(bottom, height); ++y)
        for (int32_t x = std::max(left, 0); x < std::min(right, width); ++x)
            lower[static_cast<size_t>(y) * width + x] =
                resized[static_cast<size_t>(y) * width + x];
    const int32_t blur_kernel = static_cast<int32_t>(
        (static_cast<int32_t>(0.05 * width) / 2) * 2 + 1);
    AlphaMask result{width, height,
        gaussian_blur_u8(lower, height, width, blur_kernel)};
    if (observation) {
        observation->class_one_region = std::move(face_region);
        observation->dilated = std::move(dilated);
        observation->eroded = std::move(eroded);
        observation->selected = std::move(selected);
        observation->lower_half = std::move(lower);
    }
    return result;
}

}  // namespace vrhino
