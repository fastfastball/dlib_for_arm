// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_sIMD4F_Hh_
#define DLIB_sIMD4F_Hh_

#include "simd_check.h"
#include "simd4i.h"
#include <cmath>
#include <iostream>

namespace dlib
{

#ifdef DLIB_HAVE_SSE2
    class simd4f
    {
    public:
        typedef float type;

        inline simd4f() {}
        inline simd4f(float f) { x = _mm_set1_ps(f); }
        inline simd4f(float r0, float r1, float r2, float r3) { x = _mm_setr_ps(r0,r1,r2,r3); }
        inline simd4f(const __m128& val):x(val) {}
        inline simd4f(const simd4i& val):x(_mm_cvtepi32_ps(val)) {}

        inline simd4f& operator=(const simd4i& val)
        {
            x = simd4f(val);
            return *this;
        }

        inline simd4f& operator=(const float& val)
        {
            x = simd4f(val);
            return *this;
        }

        inline simd4f& operator=(const __m128& val)
        {
            x = val;
            return *this;
        }

        inline operator __m128() const { return x; }

        // truncate to 32bit integers
        inline operator __m128i() const { return _mm_cvttps_epi32(x); }

        inline void load_aligned(const type* ptr)  { x = _mm_load_ps(ptr); }
        inline void store_aligned(type* ptr) const { _mm_store_ps(ptr, x); }
        inline void load(const type* ptr)          { x = _mm_loadu_ps(ptr); }
        inline void store(type* ptr)         const { _mm_storeu_ps(ptr, x); }

        inline unsigned int size() const { return 4; }
        inline float operator[](unsigned int idx) const 
        {
            float temp[4];
            store(temp);
            return temp[idx];
        }

    private:
        __m128 x;
    };

    class simd4f_bool
    {
    public:
        typedef float type;

        inline simd4f_bool() {}
        inline simd4f_bool(const __m128& val):x(val) {}

        inline simd4f_bool& operator=(const __m128& val)
        {
            x = val;
            return *this;
        }

        inline operator __m128() const { return x; }


    private:
        __m128 x;
    };
#elif defined(DLIB_HAVE_NEON)	
    class simd4f
    {
    public:
        typedef float type;

        inline simd4f() {}
        inline simd4f(float f) { x = vdupq_n_f32(f); }
        inline simd4f(float r0, float r1, float r2, float r3)
		{
			float __attribute__ ((aligned (16))) data[4] = { r0, r1, r2, r3 };
			x = vld1q_f32(data); 
		}
        inline simd4f(const float32x4_t& val):x(val) {}
        inline simd4f(const simd4i& val):x(vcvtq_f32_s32(val)) {}

        inline simd4f& operator=(const simd4i& val)
        {
            x = simd4f(val);
            return *this;
        }

        inline simd4f& operator=(const float& val)
        {
            x = simd4f(val);
            return *this;
        }

        inline simd4f& operator=(const float32x4_t& val)
        {
            x = val;
            return *this;
        }

        inline operator float32x4_t() const { return x; }

        // truncate to 32bit integers
        inline operator int32x4_t() const { return vcvtq_s32_f32(x); }

        inline void load_aligned(const type* ptr)  { x = vld1q_f32(ptr); }
        inline void store_aligned(type* ptr) const { vst1q_f32(ptr, x); }
        inline void load(const type* ptr)          { x = vld1q_f32(ptr); }
        inline void store(type* ptr)         const { vst1q_f32(ptr, x); }

        inline unsigned int size() const { return 4; }
        inline float operator[](unsigned int idx) const 
        {
            float temp[4];
            store(temp);
            return temp[idx];
        }

    private:
        float32x4_t x;
    };

    class simd4f_bool
    {
    public:
        typedef float type;

        inline simd4f_bool() {}
        inline simd4f_bool(const float32x4_t& val):x(val) {}
        inline simd4f_bool(float r0, float r1, float r2, float r3)
		{
			float __attribute__ ((aligned (16))) data[4] = { r0, r1, r2, r3 };
			x = vld1q_f32(data); 
		}

        inline simd4f_bool& operator=(const float32x4_t& val)
        {
            x = val;
            return *this;
        }

        inline operator float32x4_t() const { return x; }


    private:
        float32x4_t x;
    };	
#else
    class simd4f
    {
    public:
        typedef float type;

        inline simd4f() {}
        inline simd4f(float f) { x[0]=f; x[1]=f; x[2]=f; x[3]=f; }
        inline simd4f(float r0, float r1, float r2, float r3) { x[0]=r0; x[1]=r1; x[2]=r2; x[3]=r3;}
        inline simd4f(const simd4i& val) { x[0]=val[0]; x[1]=val[1]; x[2]=val[2]; x[3]=val[3];}

        // truncate to 32bit integers
        inline operator simd4i::rawarray() const 
        { 
            simd4i::rawarray temp;
            temp.a[0] = (int32)x[0];
            temp.a[1] = (int32)x[1];
            temp.a[2] = (int32)x[2];
            temp.a[3] = (int32)x[3];
            return temp;
        }

        inline simd4f& operator=(const float& val)
        {
            *this = simd4f(val);
            return *this;
        }

        inline simd4f& operator=(const simd4i& val)
        {
            x[0] = val[0];
            x[1] = val[1];
            x[2] = val[2];
            x[3] = val[3];
            return *this;
        }


        inline void load_aligned(const type* ptr)
        {
            x[0] = ptr[0];
            x[1] = ptr[1];
            x[2] = ptr[2];
            x[3] = ptr[3];
        }

        inline void store_aligned(type* ptr) const
        {
            ptr[0] = x[0];
            ptr[1] = x[1];
            ptr[2] = x[2];
            ptr[3] = x[3];
        }

        inline void load(const type* ptr)
        {
            x[0] = ptr[0];
            x[1] = ptr[1];
            x[2] = ptr[2];
            x[3] = ptr[3];
        }

        inline void store(type* ptr) const
        {
            ptr[0] = x[0];
            ptr[1] = x[1];
            ptr[2] = x[2];
            ptr[3] = x[3];
        }

        inline unsigned int size() const { return 4; }
        inline float operator[](unsigned int idx) const { return x[idx]; }

    private:
        float x[4];
    };

    class simd4f_bool
    {
    public:
        typedef float type;

        inline simd4f_bool() {}
        inline simd4f_bool(bool r0, bool r1, bool r2, bool r3) { x[0]=r0; x[1]=r1; x[2]=r2; x[3]=r3;}

        inline bool operator[](unsigned int idx) const { return x[idx]; }
    private:
        bool x[4];
    };
#endif

// ----------------------------------------------------------------------------------------

    inline std::ostream& operator<<(std::ostream& out, const simd4f& item)
    {
        float temp[4];
        item.store(temp);
        out << "(" << temp[0] << ", " << temp[1] << ", " << temp[2] << ", " << temp[3] << ")";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline simd4f operator+ (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_add_ps(lhs, rhs);
#elif defined(DLIB_HAVE_NEON)	
		return vaddq_f32(lhs, rhs);
#else
        return simd4f(lhs[0]+rhs[0],
                      lhs[1]+rhs[1],
                      lhs[2]+rhs[2],
                      lhs[3]+rhs[3]);
#endif
    }
    inline simd4f& operator+= (simd4f& lhs, const simd4f& rhs) 
    { return lhs = lhs + rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4f operator- (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_sub_ps(lhs, rhs);
#elif defined(DLIB_HAVE_NEON)
		return vsubq_f32(lhs, rhs);
#else
        return simd4f(lhs[0]-rhs[0],
                      lhs[1]-rhs[1],
                      lhs[2]-rhs[2],
                      lhs[3]-rhs[3]);
#endif
    }
    inline simd4f& operator-= (simd4f& lhs, const simd4f& rhs) 
    { return lhs = lhs - rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4f operator* (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_mul_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
		return vmulq_f32(lhs, rhs);
#else
        return simd4f(lhs[0]*rhs[0],
                      lhs[1]*rhs[1],
                      lhs[2]*rhs[2],
                      lhs[3]*rhs[3]);
#endif
    }
    inline simd4f& operator*= (simd4f& lhs, const simd4f& rhs) 
    { return lhs = lhs * rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4f operator/ (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_div_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
#if 1 //may be not so accuracy ( ref. http://rcl-rs-vvg.blogspot.tw/2010/08/simd-etudes.html )
		const float32x4_t q_inv0 = vrecpeq_f32( rhs );
		const float32x4_t q_step0 = vrecpsq_f32( q_inv0, rhs );
		const float32x4_t q_inv1 = vmulq_f32( q_step0, q_inv0 );
		return vmulq_f32( lhs, q_inv1 );		 
#else
        float _lhs[4]; lhs.store(_lhs);
        float _rhs[4]; rhs.store(_rhs);
        return simd4f(_lhs[0]/_rhs[0],
                      _lhs[1]/_rhs[1],
                      _lhs[2]/_rhs[2],
                      _lhs[3]/_rhs[3]);
#endif
#else
        return simd4f(lhs[0]/rhs[0],
                      lhs[1]/rhs[1],
                      lhs[2]/rhs[2],
                      lhs[3]/rhs[3]);
#endif
    }
    inline simd4f& operator/= (simd4f& lhs, const simd4f& rhs) 
    { return lhs = lhs / rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator== (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmpeq_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
#if 0
float _lhs[4]; lhs.store(_lhs);
        float _rhs[4]; rhs.store(_rhs);
        return simd4f_bool(_lhs[0]==_rhs[0],
                      _lhs[1]==_rhs[1],
                      _lhs[2]==_rhs[2],
                      _lhs[3]==_rhs[3]);
#else
		return (float32x4_t)vceqq_f32(lhs, rhs);
#endif
#else
        return simd4f_bool(lhs[0]==rhs[0],
                           lhs[1]==rhs[1],
                           lhs[2]==rhs[2],
                           lhs[3]==rhs[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator!= (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmpneq_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
#if 0
		float _lhs[4]; lhs.store(_lhs);
        float _rhs[4]; rhs.store(_rhs);
        return simd4f_bool(_lhs[0]!=_rhs[0],
                      _lhs[1]!=_rhs[1],
                      _lhs[2]!=_rhs[2],
                      _lhs[3]!=_rhs[3]);
#else		
		return (float32x4_t)vmvnq_s32((int32x4_t)vceqq_f32(lhs, rhs));
#endif
#else
        return simd4f_bool(lhs[0]!=rhs[0],
                           lhs[1]!=rhs[1],
                           lhs[2]!=rhs[2],
                           lhs[3]!=rhs[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator< (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmplt_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON) 
#if 1
		return (float32x4_t)vcltq_f32(lhs, rhs);
#else
		float _lhs[4]; lhs.store(_lhs);
		float _rhs[4]; rhs.store(_rhs);
		return simd4f_bool(_lhs[0]<_rhs[0],
					  _lhs[1]<_rhs[1],
					  _lhs[2]<_rhs[2],
					  _lhs[3]<_rhs[3]);
#endif
#else
        return simd4f_bool(lhs[0]<rhs[0],
                           lhs[1]<rhs[1],
                           lhs[2]<rhs[2],
                           lhs[3]<rhs[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator> (const simd4f& lhs, const simd4f& rhs) 
    { 
        return rhs < lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator<= (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmple_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON) 
#if 1
		return (float32x4_t)vcleq_f32(lhs, rhs);
#else
		float _lhs[4]; lhs.store(_lhs);
		float _rhs[4]; rhs.store(_rhs);
		return simd4f_bool(_lhs[0]<=_rhs[0],
					  _lhs[1]<=_rhs[1],
					  _lhs[2]<=_rhs[2],
					  _lhs[3]<=_rhs[3]);
#endif
#else
        return simd4f_bool(lhs[0]<=rhs[0],
                           lhs[1]<=rhs[1],
                           lhs[2]<=rhs[2],
                           lhs[3]<=rhs[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f_bool operator>= (const simd4f& lhs, const simd4f& rhs) 
    { 
        return rhs <= lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd4f min (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_min_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
		return vminq_f32(lhs, rhs);
#else
        return simd4f(std::min(lhs[0],rhs[0]),
                      std::min(lhs[1],rhs[1]),
                      std::min(lhs[2],rhs[2]),
                      std::min(lhs[3],rhs[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f max (const simd4f& lhs, const simd4f& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_max_ps(lhs, rhs); 
#elif defined(DLIB_HAVE_NEON)
		return vmaxq_f32(lhs, rhs);
#else
        return simd4f(std::max(lhs[0],rhs[0]),
                      std::max(lhs[1],rhs[1]),
                      std::max(lhs[2],rhs[2]),
                      std::max(lhs[3],rhs[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f reciprocal (const simd4f& item) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_rcp_ps(item); 
#elif defined(DLIB_HAVE_NEON)
#if 1
		float _lhs[4]; item.store(_lhs);
		return simd4f(1.0f/_lhs[0],
					  1.0f/_lhs[1],
					  1.0f/_lhs[2],
					  1.0f/_lhs[3]);
#else 
		float32x4_t recip = vrecpeq_f32(item);
		return vmulq_f32(recip, vrecpsq_f32(recip, item));
#endif		
#else
        return simd4f(1.0f/item[0],
                      1.0f/item[1],
                      1.0f/item[2],
                      1.0f/item[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f reciprocal_sqrt (const simd4f& item) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_rsqrt_ps(item); 
#elif defined(DLIB_HAVE_NEON)
#if 1 // TODO : optz
		float _lhs[4]; item.store(_lhs);
		return simd4f(1.0f/std::sqrt(_lhs[0]),
					  1.0f/std::sqrt(_lhs[1]),
					  1.0f/std::sqrt(_lhs[2]),
					  1.0f/std::sqrt(_lhs[3]));
#else 
		return vrsqrteq_f32(item);
#endif
#else
        return simd4f(1.0f/std::sqrt(item[0]),
                      1.0f/std::sqrt(item[1]),
                      1.0f/std::sqrt(item[2]),
                      1.0f/std::sqrt(item[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline float dot(const simd4f& lhs, const simd4f& rhs);
    inline float sum(const simd4f& item)
    {
#ifdef DLIB_HAVE_SSE41
        return dot(simd4f(1), item);
#elif defined(DLIB_HAVE_SSE3)
        simd4f temp = _mm_hadd_ps(item,item);
        return _mm_cvtss_f32(_mm_hadd_ps(temp,temp));
#elif defined(DLIB_HAVE_SSE2) && (!defined(_MSC_VER) || _MSC_VER!=1400)
        simd4f temp = _mm_add_ps(item,_mm_movehl_ps(item,item));
        simd4f temp2 = _mm_shuffle_ps(temp,temp,1);
        return _mm_cvtss_f32(_mm_add_ss(temp,temp2));
#elif defined(DLIB_HAVE_NEON) //TODO : optz 
		float32x2_t r = vadd_f32(vget_high_f32(item), vget_low_f32(item));
		return vget_lane_f32(vpadd_f32(r, r), 0);
#else
        return item[0]+item[1]+item[2]+item[3];
#endif
    }

// ----------------------------------------------------------------------------------------

    inline float dot(const simd4f& lhs, const simd4f& rhs)
    {
#ifdef DLIB_HAVE_SSE41
        return _mm_cvtss_f32(_mm_dp_ps(lhs, rhs, 0xff));
#else
        return sum(lhs*rhs);
#endif
    }
   
// ----------------------------------------------------------------------------------------

    inline simd4f sqrt(const simd4f& item)
    {
#ifdef DLIB_HAVE_SSE2
        return _mm_sqrt_ps(item);
#elif defined(DLIB_HAVE_NEON)
#if 0 //TODO: optz
		float32x4_t recipsq = vrsqrteq_f32(item);
		return vrecpeq_f32(recipsq);
#else
		float _lhs[4]; item.store(_lhs);
		return simd4f(std::sqrt(_lhs[0]),
					  std::sqrt(_lhs[1]),
					  std::sqrt(_lhs[2]),
					  std::sqrt(_lhs[3]));
#endif		
#else
        return simd4f(std::sqrt(item[0]),
                      std::sqrt(item[1]),
                      std::sqrt(item[2]),
                      std::sqrt(item[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f ceil(const simd4f& item)
    {
#ifdef DLIB_HAVE_SSE41
        return _mm_ceil_ps(item);
#elif defined(DLIB_HAVE_SSE2) || defined(DLIB_HAVE_NEON)
        float temp[4];
        item.store(temp);
        temp[0] = std::ceil(temp[0]);
        temp[1] = std::ceil(temp[1]);
        temp[2] = std::ceil(temp[2]);
        temp[3] = std::ceil(temp[3]);
        simd4f temp2;
        temp2.load(temp);
        return temp2;
#else
        return simd4f(std::ceil(item[0]),
                      std::ceil(item[1]),
                      std::ceil(item[2]),
                      std::ceil(item[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4f floor(const simd4f& item)
    {
#ifdef DLIB_HAVE_SSE41
        return _mm_floor_ps(item);
#elif defined(DLIB_HAVE_SSE2) || defined(DLIB_HAVE_NEON)
        float temp[4];
        item.store(temp);
        temp[0] = std::floor(temp[0]);
        temp[1] = std::floor(temp[1]);
        temp[2] = std::floor(temp[2]);
        temp[3] = std::floor(temp[3]);
        simd4f temp2;
        temp2.load(temp);
        return temp2;
#else
        return simd4f(std::floor(item[0]),
                      std::floor(item[1]),
                      std::floor(item[2]),
                      std::floor(item[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    // perform cmp ? a : b
    inline simd4f select(const simd4f_bool& cmp, const simd4f& a, const simd4f& b)
    {
#ifdef DLIB_HAVE_SSE41
        return _mm_blendv_ps(b,a,cmp);
#elif defined(DLIB_HAVE_SSE2)
        return _mm_or_ps(_mm_and_ps(cmp,a) , _mm_andnot_ps(cmp,b));
#elif defined(DLIB_HAVE_NEON)
#if 1 
		//simd4f cmp1(cmp);
		//simd4f r1 = vandq_s32((int32x4_t)cmp,(int32x4_t)((float32x4_t)a));
		//simd4f r2 = vbicq_s32((int32x4_t)((float32x4_t)b), (int32x4_t)cmp);
		//std::cout << " cmp1 = " << cmp1 << " a = " << a << " b = " << b << std::endl;
		//std::cout << " r1 = " << r1 << " r2 = " << r2 << std::endl;
		return (float32x4_t)vorrq_s32(
		vandq_s32((int32x4_t)((float32x4_t)cmp),(int32x4_t)((float32x4_t)a)) , 
		vbicq_s32((int32x4_t)((float32x4_t)b), (int32x4_t)((float32x4_t)cmp)));
#else
		simd4f v(cmp);
		float _cmp[4]; v.store(_cmp);
		return simd4f(_cmp[0]?a[0]:b[0],
					  _cmp[1]?a[1]:b[1],
					  _cmp[2]?a[2]:b[2],
					  _cmp[3]?a[3]:b[3]);
#endif

#else
        return simd4f(cmp[0]?a[0]:b[0],
                      cmp[1]?a[1]:b[1],
                      cmp[2]?a[2]:b[2],
                      cmp[3]?a[3]:b[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_sIMD4F_Hh_

