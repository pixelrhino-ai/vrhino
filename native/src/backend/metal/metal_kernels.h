#pragma once

namespace vrhino::metal {

inline constexpr const char* kMetalKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int MAX_DIMS = 8;
struct Meta { int rank; int pad; long shape[8]; long strides[8]; };

inline long broadcast_offset(long linear, constant Meta& output, constant Meta& input) {
    long offset = 0;
    int shift = output.rank - input.rank;
    for (int d = output.rank - 1; d >= 0; --d) {
        long coordinate = linear % output.shape[d]; linear /= output.shape[d];
        int id = d - shift;
        if (id >= 0 && input.shape[id] != 1) offset += coordinate * input.strides[id];
    }
    return offset;
}
inline long broadcast_offset_thread(long linear, thread Meta& output, constant Meta& input) {
    long offset = 0; int shift = output.rank - input.rank;
    for (int d = output.rank - 1; d >= 0; --d) { long coordinate = linear % output.shape[d]; linear /= output.shape[d]; int id = d - shift; if (id >= 0 && input.shape[id] != 1) offset += coordinate * input.strides[id]; }
    return offset;
}

inline float bf16_load(ushort value) { return as_type<float>(uint(value) << 16); }
inline ushort bf16_store(float value) {
    uint bits = as_type<uint>(value);
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

struct CastParams { ulong count; int source; int destination; };
kernel void cast_kernel(const device uchar* input [[buffer(0)]],
                        device uchar* output [[buffer(1)]],
                        constant CastParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    float value = 0.0f;
    if (p.source == 0) value = reinterpret_cast<const device float*>(input)[id];
    else if (p.source == 1) value = half(reinterpret_cast<const device half*>(input)[id]);
    else if (p.source == 2) value = bf16_load(reinterpret_cast<const device ushort*>(input)[id]);
    else if (p.source == 3) value = float(reinterpret_cast<const device long*>(input)[id]);
    else if (p.source == 4) value = float(reinterpret_cast<const device int*>(input)[id]);
    else value = float(input[id]);
    if (p.destination == 0) reinterpret_cast<device float*>(output)[id] = value;
    else if (p.destination == 1) reinterpret_cast<device half*>(output)[id] = half(value);
    else if (p.destination == 2) reinterpret_cast<device ushort*>(output)[id] = bf16_store(value);
    else if (p.destination == 3) reinterpret_cast<device long*>(output)[id] = long(value);
    else if (p.destination == 4) reinterpret_cast<device int*>(output)[id] = int(value);
    else output[id] = uchar(value != 0.0f);
}

struct BinaryParams { ulong count; int kind; int pad; Meta output; Meta a; Meta b; };
kernel void binary_f32(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                       device float* output [[buffer(2)]], constant BinaryParams& p [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    float av = a[broadcast_offset(id, p.output, p.a)];
    float bv = b[broadcast_offset(id, p.output, p.b)];
    output[id] = p.kind == 0 ? av + bv :
        (p.kind == 1 ? av * bv : (p.kind == 2 ? av / bv : max(av, bv)));
}

struct BiasParams { ulong count; long width; };
kernel void bias_f32(device float* output [[buffer(0)]], const device float* bias [[buffer(1)]],
                     constant BiasParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) < p.count) output[id] += bias[long(id) % p.width];
}

struct MatmulParams { ulong count; long m; long n; long k; int has_bias; };
kernel void matmul_f32(const device float* input [[buffer(0)]], const device float* weight [[buffer(1)]],
                       const device float* bias [[buffer(2)]], device float* output [[buffer(3)]],
                       constant MatmulParams& p [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long row = long(id) / p.n, column = long(id) % p.n; float sum = p.has_bias ? bias[column] : 0.0f;
    for (long inner = 0; inner < p.k; ++inner) sum += input[row * p.k + inner] * weight[column * p.k + inner];
    output[id] = sum;
}

kernel void batched_matmul_f32(const device float* a [[buffer(0)]],
                               const device float* b [[buffer(1)]],
                               device float* output [[buffer(2)]],
                               constant MatmulParams& p [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long column = long(id) % p.n;
    long row = (long(id) / p.n) % p.m;
    long batch = long(id) / (p.m * p.n);
    float sum = 0.0f;
    for (long inner = 0; inner < p.k; ++inner)
        sum += a[(batch * p.m + row) * p.k + inner] *
               b[(batch * p.k + inner) * p.n + column];
    output[id] = sum;
}

struct PermuteParams { ulong count; Meta input; Meta output; Meta dimensions; };
kernel void permute_f32(const device float* input [[buffer(0)]], device float* output [[buffer(1)]],
                        constant PermuteParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long remaining = id, source = 0;
    for (int d = p.output.rank - 1; d >= 0; --d) {
        long c = remaining % p.output.shape[d]; remaining /= p.output.shape[d];
        source += c * p.input.strides[p.dimensions.shape[d]];
    }
    output[id] = input[source];
}

struct SliceParams { ulong count; Meta input; Meta output; int dimension; int pad; long start; };
kernel void slice_f32(const device float* input [[buffer(0)]], device float* output [[buffer(1)]],
                      constant SliceParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long remaining = id, source = 0;
    for (int d = p.output.rank - 1; d >= 0; --d) {
        long c = remaining % p.output.shape[d]; remaining /= p.output.shape[d];
        if (d == p.dimension) c += p.start;
        source += c * p.input.strides[d];
    }
    output[id] = input[source];
}

struct GatherParams { ulong count; long indices_per_batch; long rows; long width; int batched; int index_dtype; };
kernel void indexed_gather_f32(const device float* table [[buffer(0)]],
                               const device uchar* indices [[buffer(1)]],
                               device float* output [[buffer(2)]],
                               constant GatherParams& p [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long logical = long(id) / p.width, column = long(id) % p.width;
    long batch = p.batched ? logical / p.indices_per_batch : 0;
    long local = p.batched ? logical % p.indices_per_batch : logical;
    long offset = p.batched ? batch * p.indices_per_batch + local : local;
    long row = p.index_dtype == 3 ? reinterpret_cast<const device long*>(indices)[offset]
                                  : long(reinterpret_cast<const device int*>(indices)[offset]);
    output[id] = table[(batch * p.rows + row) * p.width + column];
}

struct ConcatParams { ulong count; Meta input; Meta output; int dimension; int pad; long offset; };
kernel void concat_f32(const device float* input [[buffer(0)]], device float* output [[buffer(1)]],
                       constant ConcatParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.count) return;
    long remaining = id, target = 0;
    for (int d = p.input.rank - 1; d >= 0; --d) {
        long c = remaining % p.input.shape[d]; remaining /= p.input.shape[d];
        if (d == p.dimension) c += p.offset;
        target += c * p.output.strides[d];
    }
    output[target] = input[id];
}

struct NormParams { ulong count; long rows; long width; float eps; int has_weight; int has_bias; int axis; };
kernel void layer_norm_f32(const device float* input [[buffer(0)]], const device float* weight [[buffer(1)]],
                           const device float* bias [[buffer(2)]], device float* output [[buffer(3)]],
                           constant NormParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (long(row) >= p.rows) return;
    float sum = 0.0f, square = 0.0f;
    for (long c = 0; c < p.width; ++c) { float x=input[long(row)*p.width+c]; sum+=x; square+=x*x; }
    float mean=sum/float(p.width), inv=rsqrt(max(0.0f, square/float(p.width)-mean*mean)+p.eps);
    for (long c = 0; c < p.width; ++c) { float x=(input[long(row)*p.width+c]-mean)*inv; if(p.has_weight)x*=weight[c]; if(p.has_bias)x+=bias[c]; output[long(row)*p.width+c]=x; }
}
kernel void rms_norm_f32(const device float* input [[buffer(0)]], const device float* weight [[buffer(1)]],
                         device float* output [[buffer(2)]], constant NormParams& p [[buffer(3)]],
                         uint row [[thread_position_in_grid]]) {
    if (long(row) >= p.rows) return;
    float square=0.0f; for(long c=0;c<p.width;++c){float x=input[long(row)*p.width+c];square+=x*x;}
    float inv=rsqrt(square/float(p.width)+p.eps);
    for(long c=0;c<p.width;++c){float x=input[long(row)*p.width+c]*inv;if(p.has_weight)x*=weight[c];output[long(row)*p.width+c]=x;}
}

struct AxisNormParams { ulong count; Meta meta; int axis; float eps; int has_weight; };
kernel void rms_norm_axis_f32(const device float* input [[buffer(0)]], const device float* weight [[buffer(1)]],
                              device float* output [[buffer(2)]], constant AxisNormParams& p [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=p.count)return; long stride=p.meta.strides[p.axis], coordinate=(long(id)/stride)%p.meta.shape[p.axis], base=long(id)-coordinate*stride;
    float square=0.0f;for(long x=0;x<p.meta.shape[p.axis];++x){float v=input[base+x*stride];square+=v*v;}
    float value=input[id]*rsqrt(square/float(p.meta.shape[p.axis])+p.eps);if(p.has_weight)value*=weight[coordinate];output[id]=value;
}

struct UnaryParams { ulong count; int kind; float a; float b; };
inline float erf_approx(float x) {
    float sign_value = x < 0.0f ? -1.0f : 1.0f; x = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return sign_value * y;
}
kernel void activation_f32(const device float* input [[buffer(0)]], device float* output [[buffer(1)]],
                           constant UnaryParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=p.count)return;float x=input[id];
    if(p.kind==0)output[id]=x/(1.0f+exp(-x));
    else if(p.kind==1)output[id]=0.5f*x*(1.0f+erf_approx(x*0.7071067811865475f));
    else if(p.kind==2)output[id]=0.5f*x*(1.0f+tanh(0.7978845608028654f*(x+0.044715f*x*x*x)));
    else if(p.kind==3)output[id]=tanh(x);
    else output[id]=max(x,0.0f);
}
kernel void clamp_f32(const device float* input [[buffer(0)]], device float* output [[buffer(1)]],
                      constant UnaryParams& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if(ulong(id)<p.count)output[id]=clamp(input[id],p.a,p.b);
}

struct SinParams { ulong count; int width; int flip; };
kernel void sinusoidal_f32(const device float* positions [[buffer(0)]], const device float* inverse [[buffer(1)]],
                           device float* output [[buffer(2)]], constant SinParams& p [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=p.count*ulong(p.width))return;int half_width=p.width/2,column=int(id%uint(p.width));
    if(column>=2*half_width){output[id]=0.0f;return;}int logical=p.flip?((column+half_width)%(2*half_width)):column, frequency=logical%half_width;
    float phase=positions[id/uint(p.width)]*inverse[frequency];output[id]=logical<half_width?sin(phase):cos(phase);
}

struct RopeParams { ulong count; Meta x; Meta frequency; };
kernel void rope_f32(const device float* input [[buffer(0)]], const device float* cosine [[buffer(1)]],
                     const device float* sine [[buffer(2)]], device float* output [[buffer(3)]],
                     constant RopeParams& p [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=p.count)return;long width=p.x.shape[p.x.rank-1],pair=(long(id)%width)^1,paired=long(id)-long(id)%width+pair;
    float rotated=(id&1)?input[paired]:-input[paired];long f=broadcast_offset(id,p.x,p.frequency);output[id]=input[id]*cosine[f]+rotated*sine[f];
}

struct AttentionParams { int batch; int query_tokens; int key_tokens; int heads; int width; int causal; float scale; int has_mask; int has_bias; int observe; Meta mask; Meta bias; };
inline long attention_observation_index(int b,int query,int key,int head,constant AttentionParams& p){return ((long(b)*p.heads+head)*p.query_tokens+query)*p.key_tokens+key;}
inline float attention_raw_score(const device float* q,const device float* k,int b,int query,int key,int head,constant AttentionParams& p){float s=0.0f;for(int d=0;d<p.width;++d){long qi=(((long(b)*p.query_tokens+query)*p.heads+head)*p.width+d);long ki=(((long(b)*p.key_tokens+key)*p.heads+head)*p.width+d);s+=q[qi]*k[ki];}return s*p.scale;}
inline float attention_apply_bias(float raw,const device float* bias,int b,int query,int key,int head,constant AttentionParams& p){float s=raw;if(p.has_bias){Meta logical{};logical.rank=4;logical.shape[0]=p.batch;logical.shape[1]=p.heads;logical.shape[2]=p.query_tokens;logical.shape[3]=p.key_tokens;logical.strides[3]=1;logical.strides[2]=p.key_tokens;logical.strides[1]=long(p.query_tokens)*p.key_tokens;logical.strides[0]=long(p.heads)*p.query_tokens*p.key_tokens;long index=attention_observation_index(b,query,key,head,p);s+=bias[broadcast_offset_thread(index,logical,p.bias)];}return s;}
inline float attention_biased_score(const device float* q,const device float* k,const device float* bias,int b,int query,int key,int head,constant AttentionParams& p){return attention_apply_bias(attention_raw_score(q,k,b,query,key,head,p),bias,b,query,key,head,p);}
kernel void attention_f32(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],const device uchar* mask [[buffer(3)]],const device float* bias [[buffer(4)]],device float* output [[buffer(5)]],device float* observed_score [[buffer(6)]],device float* observed_biased_score [[buffer(7)]],device float* observed_softmax [[buffer(8)]],constant AttentionParams& p [[buffer(9)]],uint id [[thread_position_in_grid]]){
    int units=p.batch*p.query_tokens*p.heads;if(int(id)>=units)return;int head=int(id)%p.heads,query=(int(id)/p.heads)%p.query_tokens,b=int(id)/(p.heads*p.query_tokens);float maximum=-INFINITY;
    for(int key=0;key<p.key_tokens;++key){bool valid=!p.causal||key<=query;if(p.has_mask){Meta logical{};logical.rank=3;logical.shape[0]=p.batch;logical.shape[1]=p.query_tokens;logical.shape[2]=p.key_tokens;logical.strides[2]=1;logical.strides[1]=p.key_tokens;logical.strides[0]=long(p.query_tokens)*p.key_tokens;long index=((long(b)*p.query_tokens+query)*p.key_tokens+key);valid=valid&&mask[broadcast_offset_thread(index,logical,p.mask)];}float raw=attention_raw_score(q,k,b,query,key,head,p),biased=attention_apply_bias(raw,bias,b,query,key,head,p);if(p.observe){long oi=attention_observation_index(b,query,key,head,p);observed_score[oi]=raw;observed_biased_score[oi]=biased;}if(valid)maximum=max(maximum,biased);}
    float denominator=0.0f;for(int key=0;key<p.key_tokens;++key){bool valid=!p.causal||key<=query;if(p.has_mask){Meta logical{};logical.rank=3;logical.shape[0]=p.batch;logical.shape[1]=p.query_tokens;logical.shape[2]=p.key_tokens;logical.strides[2]=1;logical.strides[1]=p.key_tokens;logical.strides[0]=long(p.query_tokens)*p.key_tokens;long index=((long(b)*p.query_tokens+query)*p.key_tokens+key);valid=valid&&mask[broadcast_offset_thread(index,logical,p.mask)];}if(valid)denominator+=exp(attention_biased_score(q,k,bias,b,query,key,head,p)-maximum);}
    for(int d=0;d<p.width;++d){float value=0.0f;for(int key=0;key<p.key_tokens;++key){bool valid=!p.causal||key<=query;if(p.has_mask){Meta logical{};logical.rank=3;logical.shape[0]=p.batch;logical.shape[1]=p.query_tokens;logical.shape[2]=p.key_tokens;logical.strides[2]=1;logical.strides[1]=p.key_tokens;logical.strides[0]=long(p.query_tokens)*p.key_tokens;long index=((long(b)*p.query_tokens+query)*p.key_tokens+key);valid=valid&&mask[broadcast_offset_thread(index,logical,p.mask)];}float probability=valid?exp(attention_biased_score(q,k,bias,b,query,key,head,p)-maximum)/denominator:0.0f;if(p.observe&&d==0)observed_softmax[attention_observation_index(b,query,key,head,p)]=probability;if(valid){long vi=(((long(b)*p.key_tokens+key)*p.heads+head)*p.width+d);value+=probability*v[vi];}}long oi=(((long(b)*p.query_tokens+query)*p.heads+head)*p.width+d);output[oi]=value;}
}

struct PadParams { ulong count; Meta input; Meta output; Meta before; float value; int replicate; };
kernel void pad_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant PadParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long remaining=id,source=0;bool valid=true;for(int d=p.output.rank-1;d>=0;--d){long c=remaining%p.output.shape[d];remaining/=p.output.shape[d];c-=p.before.shape[d];if(c<0||c>=p.input.shape[d]){if(!p.replicate)valid=false;c=max(long(0),min(p.input.shape[d]-1,c));}source+=c*p.input.strides[d];}output[id]=valid?input[source]:p.value;
}
struct ReduceParams { ulong count; Meta input; Meta output; int dimension; };
kernel void reduce_sum_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant ReduceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long remaining=id,base=0;for(int d=p.output.rank-1;d>=0;--d){long c=remaining%p.output.shape[d];remaining/=p.output.shape[d];int input_dim=p.output.rank==p.input.rank?d:(d<p.dimension?d:d+1);if(input_dim!=p.dimension)base+=c*p.input.strides[input_dim];}float sum=0.0f;for(long c=0;c<p.input.shape[p.dimension];++c)sum+=input[base+c*p.input.strides[p.dimension]];output[id]=sum;
}
struct SoftmaxParams { ulong rows; long axis_size; long inner; };
kernel void softmax_axis_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant SoftmaxParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.rows)return;long outer=long(id)/p.inner,inner_index=long(id)%p.inner,base=outer*p.axis_size*p.inner+inner_index;float maximum=-INFINITY;for(long c=0;c<p.axis_size;++c)maximum=max(maximum,input[base+c*p.inner]);float denominator=0.0f;for(long c=0;c<p.axis_size;++c)denominator+=exp(input[base+c*p.inner]-maximum);for(long c=0;c<p.axis_size;++c)output[base+c*p.inner]=exp(input[base+c*p.inner]-maximum)/denominator;
}

struct GroupNormParams { int batch; int channels; long spatial; int groups; float eps; int has_weight; int has_bias; };
kernel void group_norm_f32(const device float* input [[buffer(0)]],const device float* weight [[buffer(1)]],const device float* bias [[buffer(2)]],device float* output [[buffer(3)]],constant GroupNormParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]){
    if(int(id)>=p.batch*p.groups)return;int b=int(id)/p.groups,group=int(id)%p.groups,channels_per=p.channels/p.groups;long base=(long(b)*p.channels+group*channels_per)*p.spatial,count=long(channels_per)*p.spatial;float sum=0.0f,square=0.0f;for(long i=0;i<count;++i){float x=input[base+i];sum+=x;square+=x*x;}float mean=sum/float(count),inv=rsqrt(max(0.0f,square/float(count)-mean*mean)+p.eps);for(int c=0;c<channels_per;++c)for(long pos=0;pos<p.spatial;++pos){long index=base+long(c)*p.spatial+pos;float x=(input[index]-mean)*inv;int absolute=group*channels_per+c;if(p.has_weight)x*=weight[absolute];if(p.has_bias)x+=bias[absolute];output[index]=x;}
}

struct MetaCountParams { ulong count; Meta input; Meta output; int axis; float eps; };
kernel void interpolate_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant MetaCountParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long remaining=id,source=0;for(int d=p.output.rank-1;d>=0;--d){long c=remaining%p.output.shape[d];remaining/=p.output.shape[d];source+=(c*p.input.shape[d]/p.output.shape[d])*p.input.strides[d];}output[id]=input[source];
}
struct Bilinear2DParams { ulong count; int channels,ih,iw,oh,ow,align_corners; };
kernel void interpolate_bilinear_2d_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant Bilinear2DParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long r=id;int ox=int(r%p.ow);r/=p.ow;int oy=int(r%p.oh);r/=p.oh;int c=int(r%p.channels),b=int(r/p.channels);
    float sy=p.align_corners&&p.oh>1?float(oy)*float(p.ih-1)/float(p.oh-1):(float(oy)+.5f)*float(p.ih)/float(p.oh)-.5f;
    float sx=p.align_corners&&p.ow>1?float(ox)*float(p.iw-1)/float(p.ow-1):(float(ox)+.5f)*float(p.iw)/float(p.ow)-.5f;
    sy=clamp(sy,0.0f,float(p.ih-1));sx=clamp(sx,0.0f,float(p.iw-1));int y0=int(floor(sy)),x0=int(floor(sx)),y1=min(y0+1,p.ih-1),x1=min(x0+1,p.iw-1);float ly=sy-y0,lx=sx-x0;long base=(long(b)*p.channels+c)*p.ih*p.iw;float top=input[base+y0*p.iw+x0]*(1-lx)+input[base+y0*p.iw+x1]*lx;float bottom=input[base+y1*p.iw+x0]*(1-lx)+input[base+y1*p.iw+x1]*lx;output[id]=top*(1-ly)+bottom*ly;
}
kernel void pixel_norm_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant MetaCountParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long coordinate=(long(id)/p.input.strides[p.axis])%p.input.shape[p.axis],base=long(id)-coordinate*p.input.strides[p.axis];float square=0.0f;for(long x=0;x<p.input.shape[p.axis];++x){float v=input[base+x*p.input.strides[p.axis]];square+=v*v;}output[id]=input[id]/sqrt(square/float(p.input.shape[p.axis])+p.eps);
}
kernel void l2_normalize_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant MetaCountParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long coordinate=(long(id)/p.input.strides[p.axis])%p.input.shape[p.axis],base=long(id)-coordinate*p.input.strides[p.axis];float square=0.0f;for(long x=0;x<p.input.shape[p.axis];++x){float v=input[base+x*p.input.strides[p.axis]];square+=v*v;}output[id]=input[id]/(sqrt(square)+p.eps);
}

struct Pool2DParams { ulong count; int channels; int ih,iw,oh,ow; int kh,kw,sh,sw,ph,pw; };
kernel void max_pool2d_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant Pool2DParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long r=id;int ox=int(r%p.ow);r/=p.ow;int oy=int(r%p.oh);r/=p.oh;int c=int(r%p.channels),b=int(r/p.channels);float value=-INFINITY;for(int ky=0;ky<p.kh;++ky){int iy=oy*p.sh-p.ph+ky;if(iy<0||iy>=p.ih)continue;for(int kx=0;kx<p.kw;++kx){int ix=ox*p.sw-p.pw+kx;if(ix<0||ix>=p.iw)continue;long source=((long(b)*p.channels+c)*p.ih+iy)*p.iw+ix;value=max(value,input[source]);}}output[id]=value;
}
kernel void exp_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant UnaryParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;output[id]=exp(input[id]);
}
kernel void sqrt_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant UnaryParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;output[id]=sqrt(input[id]);
}

struct ShuffleParams { ulong count; int b,c,t,h,w,ft,fh,fw; };
kernel void pixel_shuffle_f32(const device float* input [[buffer(0)]],device float* output [[buffer(1)]],constant ShuffleParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long r=id;int ow=int(r%(p.w*p.fw));r/=p.w*p.fw;int oh=int(r%(p.h*p.fh));r/=p.h*p.fh;int ot=int(r%(p.t*p.ft));r/=p.t*p.ft;int oc=int(r%p.c),ob=int(r/p.c);int iw=ow/p.fw,pw=ow%p.fw,ih=oh/p.fh,ph=oh%p.fh,it=ot/p.ft,pt=ot%p.ft,packed=(((oc*p.ft+pt)*p.fh+ph)*p.fw+pw);long source=((((long(ob)*(p.c*p.ft*p.fh*p.fw)+packed)*p.t+it)*p.h+ih)*p.w+iw);output[id]=input[source];
}

struct ConvParams { ulong count; int spatial; int batch; int in_channels; int out_channels; int groups; int in_size[3]; int out_size[3]; int kernel_size[3]; int stride[3]; int padding[3]; int dilation[3]; int has_bias; };
kernel void convolution_f32(const device float* input [[buffer(0)]],const device float* weight [[buffer(1)]],const device float* bias [[buffer(2)]],device float* output [[buffer(3)]],constant ConvParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long r=id;int ox[3]={0,0,0};for(int d=p.spatial-1;d>=0;--d){ox[d]=int(r%p.out_size[d]);r/=p.out_size[d];}int oc=int(r%p.out_channels),b=int(r/p.out_channels),out_per_group=p.out_channels/p.groups,in_per_group=p.in_channels/p.groups,group=oc/out_per_group;float sum=p.has_bias?bias[oc]:0.0f;
    int ktotal=p.kernel_size[0]*p.kernel_size[1]*p.kernel_size[2];for(int icg=0;icg<in_per_group;++icg)for(int kl=0;kl<ktotal;++kl){int kr=kl,kx[3]={0,0,0};for(int d=p.spatial-1;d>=0;--d){kx[d]=kr%p.kernel_size[d];kr/=p.kernel_size[d];}bool valid=true;int ix[3]={0,0,0};for(int d=0;d<p.spatial;++d){ix[d]=ox[d]*p.stride[d]-p.padding[d]+kx[d]*p.dilation[d];valid=valid&&ix[d]>=0&&ix[d]<p.in_size[d];}if(valid){int ic=group*in_per_group+icg;long input_index=long(b)*p.in_channels+ic;for(int d=0;d<p.spatial;++d)input_index=input_index*p.in_size[d]+ix[d];long weight_index=long(oc)*in_per_group+icg;for(int d=0;d<p.spatial;++d)weight_index=weight_index*p.kernel_size[d]+kx[d];sum+=input[input_index]*weight[weight_index];}}
    output[id]=sum;
}
struct ConvTranspose2DParams { ulong count; int batch,input_channels,output_channels,groups; int input_height,input_width,output_height,output_width; int kernel_height,kernel_width,stride_height,stride_width; int padding_height,padding_width,dilation_height,dilation_width; int has_bias; };
kernel void conv_transpose2d_f32(const device float* input [[buffer(0)]],const device float* weight [[buffer(1)]],const device float* bias [[buffer(2)]],device float* output [[buffer(3)]],constant ConvTranspose2DParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;long r=id;int ox=int(r%p.output_width);r/=p.output_width;int oy=int(r%p.output_height);r/=p.output_height;int oc=int(r%p.output_channels),b=int(r/p.output_channels),out_per_group=p.output_channels/p.groups,in_per_group=p.input_channels/p.groups,group=oc/out_per_group,local_output=oc%out_per_group;float value=p.has_bias?bias[oc]:0.0f;for(int local_input=0;local_input<in_per_group;++local_input){int ic=group*in_per_group+local_input;for(int ky=0;ky<p.kernel_height;++ky){int ny=oy+p.padding_height-ky*p.dilation_height;if(ny<0||ny%p.stride_height!=0)continue;int iy=ny/p.stride_height;if(iy>=p.input_height)continue;for(int kx=0;kx<p.kernel_width;++kx){int nx=ox+p.padding_width-kx*p.dilation_width;if(nx<0||nx%p.stride_width!=0)continue;int ix=nx/p.stride_width;if(ix>=p.input_width)continue;long ii=((long(b)*p.input_channels+ic)*p.input_height+iy)*p.input_width+ix,wi=((long(ic)*out_per_group+local_output)*p.kernel_height+ky)*p.kernel_width+kx;value+=input[ii]*weight[wi];}}}output[id]=value;
}

inline uint mul_hi(uint a,uint b){return uint((ulong(a)*ulong(b))>>32);}
inline uint4 philox_round(uint4 c,uint2 k){uint hi0=mul_hi(0xD2511F53u,c.x),lo0=0xD2511F53u*c.x,hi1=mul_hi(0xCD9E8D57u,c.z),lo1=0xCD9E8D57u*c.z;return uint4(hi1^c.y^k.x,lo1,hi0^c.w^k.y,lo0);}
inline uint4 philox10(uint4 c,uint2 k){for(int i=0;i<10;++i){c=philox_round(c,k);k+=uint2(0x9E3779B9u,0xBB67AE85u);}return c;}
struct RngParams { ulong count; ulong seed; ulong offset; };
kernel void rng_normal_f32(device float* output [[buffer(0)]],constant RngParams& p [[buffer(1)]],uint id [[thread_position_in_grid]]){
    if(ulong(id)>=p.count)return;uint4 c=uint4(uint(p.offset>>2),uint(p.offset>>34),id,0u);uint2 k=uint2(uint(p.seed),uint(p.seed>>32));uint4 value=philox10(c,k);float inv=2.3283064365386963e-10f;float u=float(value.x)*inv+inv*0.5f,v=float(value.y)*(inv*6.283185307179586f)+(inv*3.141592653589793f);output[id]=sqrt(-2.0f*log(u))*sin(v);
}
)METAL";

}  // namespace vrhino::metal
