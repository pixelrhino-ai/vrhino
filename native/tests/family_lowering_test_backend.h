#pragma once
// Small F32 host oracle for synthetic qualification only, not a product backend.
#include "neural_graph_test_backend.h"
#include "vrhino/tensor_util.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <set>

namespace vrhino::family_test {
class Backend final : public neural_graph::test::TinyBackend {
public:
    std::set<const void*> cacheable;
    std::map<const void*,Tensor> address_cache;
    std::vector<const void*> patch_weights;
    size_t hits=0, misses=0, rng_calls=0;
    std::string name() const override { return "synthetic-host"; }
    bool profiling_enabled() const override { return false; }
    void profile_region_begin(const std::string&) override {}
    void profile_region_end() override {}
    void synchronize() override { ++syncs; }
    Tensor copy_to_host(const Tensor& t) override { return t; }
    Tensor copy_to_device(const Tensor& t,DType dtype) override {
        require(t.dtype()==dtype,"Synthetic copy dtype");
        if (!cacheable.contains(t.data())) return t;
        auto found=address_cache.find(t.data());
        if (found!=address_cache.end()) { ++hits; return found->second; }
        ++misses;
        auto out=Tensor::host(t.shape(),t.dtype());
        std::memcpy(out.data(),t.data(),t.bytes());
        address_cache.emplace(t.data(),out); return out;
    }
    Tensor cast(const Tensor& t,DType dtype) override {
        begin("cast");
        if (t.dtype()==dtype) return t;
        require(t.dtype()==DType::I64 && dtype==DType::F32,"Synthetic cast");
        auto out=owned(t.shape());
        for (int64_t i=0;i<t.numel();++i) out.data_as<float>()[i]=static_cast<float>(t.data_as<int64_t>()[i]);
        return out;
    }
    Tensor reshape(const Tensor& t,const std::vector<int64_t>& requested) override {
        begin("reshape"); auto shape=requested; int64_t product=1; int infer=-1;
        for (size_t i=0;i<shape.size();++i) {
            if (shape[i]==-1) { require(infer==-1,"Synthetic reshape inference"); infer=static_cast<int>(i); }
            else product*=shape[i];
        }
        if (infer>=0) shape[infer]=t.numel()/product;
        return t.reshape(shape);
    }
    Tensor rng_normal(RngState& state,const std::vector<int64_t>& shape,DType dtype) override {
        begin("rng"); ++rng_calls; require(dtype==DType::F32,"Synthetic RNG dtype");
        auto out=owned(shape);
        for (int64_t i=0;i<out.numel();++i) out.data_as<float>()[i]=float((state.seed+state.offset+i)%17)/32;
        state.offset+=out.numel(); return out; // Sentinel, not a distribution qualification.
    }
    size_t peak_device_bytes() const override { return 0; }
    size_t weight_upload_bytes() const override { return 0; }
    double weight_upload_seconds() const override { return 0; }
    Tensor linear(const Tensor& x,const Tensor& w,const Tensor* b,DType,DType) override {
        begin("linear"); require(x.dim(-1)==w.dim(1),"Synthetic linear dimensions");
        auto shape=x.shape(); shape.back()=w.dim(0); auto out=owned(shape);
        int64_t in=w.dim(1), n=w.dim(0);
        for (int64_t row=0;row<x.numel()/in;++row) for (int64_t j=0;j<n;++j) {
            float sum=b?b->data_as<float>()[j]:0;
            for (int64_t k=0;k<in;++k) sum+=x.data_as<float>()[row*in+k]*w.data_as<float>()[j*in+k];
            out.data_as<float>()[row*n+j]=sum;
        }
        return out;
    }
    static std::vector<int64_t> coordinate(int64_t i,const Tensor& t) {
        std::vector<int64_t> c(t.ndim());
        for (int64_t k=t.ndim();k-->0;) { c[k]=i%t.dim(k); i/=t.dim(k); } return c;
    }
    static int64_t offset(const std::vector<int64_t>& c,const Tensor& t) {
        int64_t i=0; for (size_t k=0;k<c.size();++k) i+=c[k]*t.strides()[k]; return i;
    }
    Tensor permute(const Tensor& x,const std::vector<int64_t>& axes) override {
        begin("permute"); std::vector<int64_t> shape; for (auto a:axes) shape.push_back(x.dim(a)); auto out=owned(shape);
        for (int64_t i=0;i<out.numel();++i) {
            auto c=coordinate(i,out), source=c; for (size_t k=0;k<c.size();++k) source[axes[k]]=c[k];
            out.data_as<float>()[i]=x.data_as<float>()[offset(source,x)];
        } return out;
    }
    Tensor slice(const Tensor& x,int64_t axis,int64_t start,int64_t stop) override {
        begin("slice"); auto shape=x.shape(); shape[axis]=stop-start; auto out=owned(shape);
        for (int64_t i=0;i<out.numel();++i) { auto c=coordinate(i,out); c[axis]+=start; out.data_as<float>()[i]=x.data_as<float>()[offset(c,x)]; }
        return out;
    }
    std::vector<Tensor> split(const Tensor& x,const std::vector<int64_t>& sizes,int64_t axis) override {
        begin("split"); std::vector<Tensor> result; int64_t start=0;
        for (auto size:sizes) { result.push_back(slice(x,axis,start,start+size)); start+=size; } return result;
    }
    Tensor layer_norm(const Tensor& x,const Tensor* w,const Tensor* b,float eps) override {
        begin("layer_norm"); auto out=owned(x.shape()); const int64_t width=x.dim(-1);
        for (int64_t row=0;row<x.numel()/width;++row) {
            float mean=0,var=0; for (int64_t k=0;k<width;++k) mean+=x.data_as<float>()[row*width+k]; mean/=width;
            for (int64_t k=0;k<width;++k) { float d=x.data_as<float>()[row*width+k]-mean; var+=d*d; } var/=width;
            for (int64_t k=0;k<width;++k) out.data_as<float>()[row*width+k]=
                (x.data_as<float>()[row*width+k]-mean)/std::sqrt(var+eps)*(w?w->data_as<float>()[k]:1)+(b?b->data_as<float>()[k]:0);
        } return out;
    }
    Tensor rms_norm(const Tensor& x,const Tensor* w,float eps,int64_t axis,DType) override {
        begin("rms_norm"); require(axis==-1 || axis==x.ndim()-1,"Synthetic RMS axis"); auto out=owned(x.shape()); auto width=x.dim(-1);
        for (int64_t row=0;row<x.numel()/width;++row) {
            float sum=0; for (int64_t k=0;k<width;++k) { float v=x.data_as<float>()[row*width+k]; sum+=v*v; }
            for (int64_t k=0;k<width;++k) out.data_as<float>()[row*width+k]=x.data_as<float>()[row*width+k]/std::sqrt(sum/width+eps)*(w?w->data_as<float>()[k]:1);
        } return out;
    }
    Tensor activation(const Tensor& x,Activation kind) override {
        begin("activation"); auto out=owned(x.shape()); require(kind==Activation::Silu || kind==Activation::GeluTanh,"Synthetic activation");
        for (int64_t i=0;i<x.numel();++i) { float v=x.data_as<float>()[i]; out.data_as<float>()[i]=kind==Activation::Silu?
            v/(1+std::exp(-v)):0.5f*v*(1+std::tanh(0.7978845608f*(v+0.044715f*v*v*v))); } return out;
    }
    Tensor sinusoidal_embedding(const Tensor& t,int64_t width,bool flip,double shift,bool) override {
        begin("sinusoidal_embedding"); auto out=owned({t.numel(),width});
        for (int64_t i=0;i<t.numel();++i) for (int64_t k=0;k<width/2;++k) {
            const double phase=t.data_as<float>()[i]*std::exp(-std::log(10000.0)*k/(width/2-shift));
            out.data_as<float>()[i*width+k]=flip?std::cos(phase):std::sin(phase);
            out.data_as<float>()[i*width+width/2+k]=flip?std::sin(phase):std::cos(phase);
        } return out;
    }
    Tensor rope_nd(const Tensor& x,const Tensor& c,const Tensor& s) override {
        begin("rope"); auto out=owned(x.shape()); auto width=x.dim(3), heads=x.dim(2);
        for (int64_t i=0;i<x.numel();i+=2) {
            const int64_t f=(i/(heads*width))*width+i%width;
            out.data_as<float>()[i]=x.data_as<float>()[i]*c.data_as<float>()[f]-x.data_as<float>()[i+1]*s.data_as<float>()[f];
            out.data_as<float>()[i+1]=x.data_as<float>()[i+1]*c.data_as<float>()[f+1]+x.data_as<float>()[i]*s.data_as<float>()[f+1];
        } return out;
    }
    Tensor attention(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor* mask,bool causal,float scale,const Tensor* bias,AttentionObservation*) override {
        begin("attention"); require(!mask && !causal && !bias && q.dim(0)==1,"Synthetic attention variant");
        auto out=owned(q.shape()); auto H=q.dim(2),D=q.dim(3),K=k.dim(1); if (scale==0) scale=1/std::sqrt(float(D));
        for (int64_t t=0;t<q.dim(1);++t) for (int64_t h=0;h<H;++h) {
            std::vector<float> scores(K); float maximum=-INFINITY,total=0;
            for (int64_t j=0;j<K;++j) { float dot=0; for (int64_t d=0;d<D;++d) dot+=q.data_as<float>()[(t*H+h)*D+d]*k.data_as<float>()[(j*H+h)*D+d]; scores[j]=dot*scale; maximum=std::max(maximum,scores[j]); }
            for (auto& score:scores) { score=std::exp(score-maximum); total+=score; }
            for (int64_t d=0;d<D;++d) { float sum=0; for (int64_t j=0;j<K;++j) sum+=scores[j]/total*v.data_as<float>()[(j*H+h)*D+d]; out.data_as<float>()[(t*H+h)*D+d]=sum; }
        } return out;
    }
    Tensor conv3d(const Tensor& x,const Tensor& w,const Tensor* b,const std::vector<int>& stride,const std::vector<int>& padding,const std::vector<int>& dilation,int groups) override {
        begin("conv3d"); patch_weights.push_back(w.data());
        require(x.dim(0)==1 && groups==1 && padding==std::vector<int>({0,0,0}) && dilation==std::vector<int>({1,1,1}),"Synthetic convolution variant");
        auto out=owned({1,w.dim(0),x.dim(2)/stride[0],x.dim(3)/stride[1],x.dim(4)/stride[2]});
        for (int64_t i=0;i<out.numel();++i) { auto c=coordinate(i,out); float sum=b?b->data_as<float>()[c[1]]:0;
            for (int64_t channel=0;channel<x.dim(1);++channel) for (int64_t t=0;t<w.dim(2);++t) for (int64_t h=0;h<w.dim(3);++h) for (int64_t z=0;z<w.dim(4);++z)
                sum+=x.data_as<float>()[offset({0,channel,c[2]*stride[0]+t,c[3]*stride[1]+h,c[4]*stride[2]+z},x)]*w.data_as<float>()[offset({c[1],channel,t,h,z},w)];
            out.data_as<float>()[i]=sum;
        } return out;
    }
    Tensor pad(const Tensor& x,const std::vector<int64_t>& padding,float value,PadMode mode) override {
        begin("pad"); require(padding.size()==4 && padding[0]==0 && padding[1]==0 && padding[2]==0 && mode==PadMode::Constant,"Synthetic text padding");
        auto shape=x.shape(); shape[1]+=padding[3]; auto out=owned(shape);
        for (int64_t i=0;i<out.numel();++i) { auto c=coordinate(i,out); out.data_as<float>()[i]=c[1]<x.dim(1)?x.data_as<float>()[offset(c,x)]:value; } return out;
    }
};
}  // namespace vrhino::family_test
