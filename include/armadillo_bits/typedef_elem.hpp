// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup typedef_elem
//! @{


#if (defined(ARMA_U8_TYPE) && defined(ARMA_S8_TYPE))
    typedef ARMA_U8_TYPE     u8;
    typedef ARMA_S8_TYPE     s8;
#else
  #if   UCHAR_MAX >= 0xff
    typedef unsigned char    u8;
    typedef          char    s8;
  #elif defined(UINT8_MAX)
    typedef          uint8_t u8;
    typedef           int8_t s8;
  #else
    #error "don't know how to typedef 'u8' on this system"
  #endif
#endif

// NOTE: "char" can be either "signed char" or "unsigned char"
// NOTE: https://en.wikipedia.org/wiki/C_data_types


#if   USHRT_MAX >= 0xffff
  typedef unsigned short    u16;
  typedef          short    s16;
#elif defined(UINT16_MAX)
  typedef          uint16_t u16;
  typedef           int16_t s16;
#else
  #error "don't know how to typedef 'u16' on this system"
#endif


#if   UINT_MAX  >= 0xffffffff
  typedef unsigned int      u32;
  typedef          int      s32;
#elif defined(UINT32_MAX)
  typedef          uint32_t u32;
  typedef           int32_t s32;
#else
  #error "don't know how to typedef 'u32' on this system"
#endif


#if   ULLONG_MAX >= 0xffffffffffffffff
  typedef unsigned long long u64;
  typedef          long long s64;
#elif defined(UINT64_MAX)
  typedef          uint64_t  u64;
  typedef           int64_t  s64;
#else
    #error "don't know how to typedef 'u64' on this system"
#endif


// for compatibility with earlier versions of Armadillo
typedef unsigned long ulng_t;
typedef          long slng_t;


#if defined(ARMA_64BIT_WORD)
  typedef u64 uword;
  typedef s64 sword;
  
  typedef u32 uhword;
  typedef s32 shword;

  #define ARMA_MAX_UWORD  0xffffffffffffffff
  #define ARMA_MAX_UHWORD 0xffffffff
#else
  typedef u32 uword;
  typedef s32 sword;

  typedef u16 uhword;
  typedef s16 shword;
  
  #define ARMA_MAX_UWORD  0xffffffff
  #define ARMA_MAX_UHWORD 0xffff
#endif


typedef std::complex<float>  cx_float;
typedef std::complex<double> cx_double;

typedef void* void_ptr;


//


#if defined(ARMA_BLAS_64BIT_INT)
  typedef long long blas_int;
  #define ARMA_MAX_BLAS_INT 0x7fffffffffffffffULL
#else
  typedef int       blas_int;
  #define ARMA_MAX_BLAS_INT 0x7fffffffU
#endif


//


#if defined(ARMA_USE_MKL_TYPES)
  // for compatibility with MKL
  typedef MKL_Complex8  blas_cxf;
  typedef MKL_Complex16 blas_cxd;
#else
  // standard BLAS and LAPACK prototypes use "void*" pointers for complex arrays
  typedef void blas_cxf;
  typedef void blas_cxd;
#endif


//


// Attempt to capture all supported float16 types.
// If C++23 or newer is used, we have a native type;
// otherwise, there are a few possibilities.
#if defined(ARMA_HAVE_CXX23)
  #if defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
    #define ARMA_HAVE_FP16
    typedef std::float16_t fp16;
  #endif

#elif defined(__GNUG__) && !defined(__clang__) && defined(ARMA_FORCE_USE_FP16)
  // All Armadillo-supported GCC versions support FP16.
  #if defined(__FLT16_MAX__) && defined(__ARM_FP16_FORMAT_IEEE)
    #define ARMA_HAVE_FP16
    typedef _Float16 fp16;
  #elif defined(__FLT16_MAX__) && defined(__SSE2__)
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=116122 for why __SSE2__ is needed.
    #define ARMA_HAVE_FP16
    typedef _Float16 fp16;
  #endif

#elif defined(__clang__) && defined(__is_identifier) && defined(ARMA_FORCE_USE_FP16)
  // NOTE: clang is_identifier behavior returns 0 if the symbol is an identifier.
  #if !(__is_identifier(_Float16))
    #define ARMA_HAVE_FP16
    typedef _Float16 fp16;
  #endif
#endif


//


// If we can detect that the implementation of FP16 is going to be software emulated,
// then it's going to be really slow.  Disable it and tell the user---unless they
// force the issue.
#if defined(ARMA_HAVE_FP16)
  #if defined(__aarch64__)
    #if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
      // We have to have the scalar intrinsics for native FP16 support.
      #define ARMA_BAD_FP16
    #endif
  #elif defined(__x86_64__) || defined(__i386__)
    #if !defined(__AVX512FP16__)
      // Without the AVX512-FP16 extensions, FP16 support is non-native (emulated).
      #define ARMA_BAD_FP16
    #endif
  #else
    // We have an architecture that does not define any macros that we can use.
    #define ARMA_BAD_FP16
  #endif

  #if defined(ARMA_BAD_FP16)
    #if defined(ARMA_FORCE_USE_FP16)
      #pragma message ("WARNING: 16-bit floating point support enabled (via ARMA_FORCE_USE_FP16), but native hardware support not detected---use of fp16 could be very slow!")

      // An additional warning: if C++23 is not enabled, many function definitions might not exist.
      // Whether this is a problem depends on what the user is doing.
      #if !defined(ARMA_HAVE_CXX23)
        #pragma message("WARNING: C++23 mode not enabled but 16-bit floating point support is forced (via ARMA_FORCE_USE_FP16); compilation may fail as some std:: functions may not work on fp16s!");
      #endif
    #else
      #undef ARMA_HAVE_FP16
    #endif
  #endif

  #undef ARMA_BAD_FP16
#elif defined(ARMA_FORCE_USE_FP16) && !defined(ARMA_HAVE_FP16)
  #pragma message("WARNING: 16-bit floating point support is forced (via ARMA_FORCE_USE_FP16), but no usable fp16 type could be detected!   Disabled.");
#endif



//


// NOTE: blas_len is the fortran type for "hidden" arguments that specify the length of character arguments;
// NOTE: it varies across compilers, compiler versions and systems (eg. 32 bit vs 64 bit);
// NOTE: the default setting of "size_t" is an educated guess.
// NOTE: ---
// NOTE: for gcc / gfortran:  https://gcc.gnu.org/onlinedocs/gfortran/Argument-passing-conventions.html
// NOTE: gcc 7 and earlier: int
// NOTE: gcc 8 and 9:       size_t
// NOTE: ---
// NOTE: for ifort (intel fortran compiler): 
// NOTE: "Intel Fortran Compiler User and Reference Guides", Document Number: 304970-006US, 2009, p. 301
// NOTE: http://www.complexfluids.ethz.ch/MK/ifort.pdf
// NOTE: the type is unsigned 4-byte integer on 32 bit systems
// NOTE: the type is unsigned 8-byte integer on 64 bit systems
// NOTE: ---
// NOTE: for NAG fortran: https://www.nag.co.uk/nagware/np/r62_doc/manual/compiler_11_1.html#AUTOTOC_11_1
// NOTE: Chrlen = usually int, or long long on 64-bit Windows
// NOTE: ---
// TODO: flang:  https://github.com/flang-compiler/flang/wiki
// TODO: other compilers: http://fortranwiki.org/fortran/show/Compilers

#if !defined(ARMA_FORTRAN_CHARLEN_TYPE)
  #if defined(__GNUC__) && !defined(__clang__)
    #if (__GNUC__ <= 7)
      #define ARMA_FORTRAN_CHARLEN_TYPE int
    #else
      #define ARMA_FORTRAN_CHARLEN_TYPE size_t
    #endif
  #else
    // TODO: determine the type for other compilers
    #define ARMA_FORTRAN_CHARLEN_TYPE size_t
  #endif
#endif

typedef ARMA_FORTRAN_CHARLEN_TYPE blas_len;


//! @}
