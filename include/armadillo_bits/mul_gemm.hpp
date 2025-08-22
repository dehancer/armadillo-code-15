// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup gemm
//! @{



//! for tiny square matrices, size <= 4x4
template<const bool do_trans_A=false, const bool use_alpha=false, const bool use_beta=false>
struct gemm_emul_tinysq
  {
  template<typename eT, typename TA, typename TB>
  arma_cold
  inline
  static
  void
  apply
    (
          Mat<eT>& C,
    const TA&      A,
    const TB&      B,
    const eT       alpha = eT(1),
    const eT       beta  = eT(0)
    )
    {
    arma_debug_sigprint();
    
    switch(A.n_rows)
      {
      case  4:  gemv_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply( C.colptr(3), A, B.colptr(3), alpha, beta );
      // fallthrough
      case  3:  gemv_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply( C.colptr(2), A, B.colptr(2), alpha, beta );
      // fallthrough
      case  2:  gemv_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply( C.colptr(1), A, B.colptr(1), alpha, beta );
      // fallthrough
      case  1:  gemv_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply( C.colptr(0), A, B.colptr(0), alpha, beta );
      // fallthrough
      default:  ;
      }
    }
  
  };



//! emulation of gemm(), for non-complex matrices only, as it assumes only simple transposes (ie. doesn't do hermitian transposes)
template<const bool do_trans_A=false, const bool do_trans_B=false, const bool use_alpha=false, const bool use_beta=false>
struct gemm_emul_large
  {
  template<typename eT, typename TA, typename TB>
  arma_hot
  inline
  static
  void
  apply
    (
          Mat<eT>& C,
    const TA&      A,
    const TB&      B,
    const eT       alpha = eT(1),
    const eT       beta  = eT(0)
    )
    {
    arma_debug_sigprint();

    // When we do A * B.t(), it is actually faster to transpose B entirely and perform A * B.
    // The algorithm necessary to do A * B.t() fast without transpose is a bit tedious.
    if (do_trans_A == false && do_trans_B == true)
      {
      Mat<eT> Bt;
      op_strans::apply_mat_noalias(Bt, B);
      gemm_emul_large<false, false, use_alpha, use_beta>::apply(C, A, Bt, alpha, beta);
      return;
      }

    const uword A_n_rows = (do_trans_A) ? A.n_cols : A.n_rows;
    const uword A_n_cols = (do_trans_A) ? A.n_rows : A.n_cols;

    const uword B_n_cols = (do_trans_B) ? B.n_rows : B.n_cols;

    // Block size should be such that 3 matrices fit in L1 cache.
    // Typical L1 caches are from 2kb to 64kb... let's just pick 16kb to work on a wide range of systems.
    // Thus, each block really can't be more than 16/3 ~= 5kb;
    // so we will chunk into the following sizes:
    //
    //    A: m x n
    //    B: n x k
    //    C: m x k
    //
    // Ideally, we pick the same size for everything, but A/B/C may be too small.
    constexpr static const uword ideal_dim = std::sqrt(16384 / (3 * sizeof(eT)));

    const uword m = std::min(A_n_rows, ideal_dim);
    // If we didn't use all of our allotted space, then the size we will use is (2 * m x ideal_dim2 + ideal_dim2 x ideal_dim2).
    // (The derivation below comes from the quadratic formula...)
    const uword ideal_dim2 = (m == ideal_dim) ? ideal_dim : std::sqrt(std::pow(m, 2) + 16384 / sizeof(eT)) - m;
    const uword n = std::min(A_n_cols, ideal_dim2);
    // Now the size we will use is (2 * m * n + n * ideal_dim3).
    // 16384/sizeof(eT) = 2 * m * n + n * ideal_dim3
    // n * ideal_dim3 = (16384/sizeof(eT)) - 2 * m * n
    // ideal_dim3 = ((16384/sizeof(eT)) - 2 * m * n) / n
    //            = (16384/(sizeof(eT) * n)) - 2 * m
    const uword ideal_dim3 = (n == ideal_dim2) ? ideal_dim2 : (16384 / (sizeof(eT) * n)) - 2 * m;
    const uword k = std::min(B_n_cols, ideal_dim3);

    // Increase all sizes to the next power of 2.
    uword m_pow2 = 1;
    while (2 * m_pow2 <= m)
      m_pow2 *= 2;

    uword n_pow2 = 1;
    while (2 * n_pow2 <= n)
      n_pow2 *= 2;

    uword k_pow2 = 1;
    while (2 * k_pow2 <= k)
      k_pow2 *= 2;

    if (m_pow2 < A_n_rows && m_pow2 < 16)
      m_pow2 = A_n_rows;
    if (n_pow2 < A_n_cols && n_pow2 < 16)
      n_pow2 = A_n_cols;
    if (k_pow2 < B_n_cols && k_pow2 < 16)
      k_pow2 = B_n_cols;

    // Compute size of work arrays that are needed.
    uword work_size = (do_trans_A == true && do_trans_B == false) ? 0 : n_pow2;
    arma_aligned podarray<eT> work(work_size);

    #if defined(ARMA_USE_OPENMP)
    #pragma omp parallel for schedule(static) firstprivate(work)
    #endif
    for (uword mm = 0; mm < A_n_rows; mm += m_pow2)
      {
      const uword ms = std::min(m_pow2, A_n_rows - mm);
      for (uword nn = 0; nn < A_n_cols; nn += n_pow2)
        {
        const uword ns = std::min(n_pow2, A_n_cols - nn);
        for (uword kk = 0; kk < B_n_cols; kk += k_pow2)
          {
          const uword ks = std::min(k_pow2, B_n_cols - kk);

          // now inside the chunk, do all of the work
          // we must iterate in the opposite order if transposing B
          if (do_trans_B == false)
            {
            for (uword r = 0; r < ms; ++r)
              {
              // copy data to the working memory if needed so it is contiguous
              if (do_trans_A == false)
                {
                work.copy_row_subvec(A, mm + r, nn, ns);
                }

              for (uword c = 0; c < ks; ++c)
                {
                const eT acc = (do_trans_A == false) ? op_dot::direct_dot(ns, work.memptr(), B.colptr(kk + c) + nn) :
                             /* do_trans_A == true */  op_dot::direct_dot(ns, A.colptr(mm + r) + nn, B.colptr(kk + c) + nn);

                if (nn == 0)
                  {
                       if (use_alpha == false && use_beta == false) { C(mm + r, kk + c) =         acc;                            }
                  else if (use_alpha == true  && use_beta == false) { C(mm + r, kk + c) = alpha * acc;                            }
                  else if (use_alpha == false && use_beta == true ) { C(mm + r, kk + c) =         acc + beta * C(mm + r, kk + c); }
                  else if (use_alpha == true  && use_beta == true ) { C(mm + r, kk + c) = alpha * acc + beta * C(mm + r, kk + c); }
                  }
                else
                  {
                       if (use_alpha == false && use_beta == false) { C(mm + r, kk + c) +=         acc;                            }
                  else if (use_alpha == true  && use_beta == false) { C(mm + r, kk + c) += alpha * acc;                            }
                  else if (use_alpha == false && use_beta == true ) { C(mm + r, kk + c) +=         acc + beta * C(mm + r, kk + c); }
                  else if (use_alpha == true  && use_beta == true ) { C(mm + r, kk + c) += alpha * acc + beta * C(mm + r, kk + c); }
                  }
                }
              }
            }
          else /* do_trans_B == true */
            {
            for (uword c = 0; c < ks; ++c)
              {
              work.copy_row_subvec(B, kk + c, nn, ns);

              for (uword r = 0; r < ms; ++r)
                {
                const eT acc = op_dot::direct_dot(ns, A.colptr(mm + r) + nn, work.memptr());

                if (nn == 0)
                  {
                       if (use_alpha == false && use_beta == false) { C(mm + r, kk + c) =         acc;                            }
                  else if (use_alpha == true  && use_beta == false) { C(mm + r, kk + c) = alpha * acc;                            }
                  else if (use_alpha == false && use_beta == true ) { C(mm + r, kk + c) =         acc + beta * C(mm + r, kk + c); }
                  else if (use_alpha == true  && use_beta == true ) { C(mm + r, kk + c) = alpha * acc + beta * C(mm + r, kk + c); }
                  }
                else
                  {
                       if (use_alpha == false && use_beta == false) { C(mm + r, kk + c) +=         acc;                            }
                  else if (use_alpha == true  && use_beta == false) { C(mm + r, kk + c) += alpha * acc;                            }
                  else if (use_alpha == false && use_beta == true ) { C(mm + r, kk + c) +=         acc + beta * C(mm + r, kk + c); }
                  else if (use_alpha == true  && use_beta == true ) { C(mm + r, kk + c) += alpha * acc + beta * C(mm + r, kk + c); }
                  }
                }
              }
            }
          }
        }
      }
    }
  };



template<const bool do_trans_A=false, const bool do_trans_B=false, const bool use_alpha=false, const bool use_beta=false>
struct gemm_emul
  {
  template<typename eT, typename TA, typename TB>
  arma_hot
  inline
  static
  void
  apply
    (
          Mat<eT>& C,
    const TA&      A,
    const TB&      B,
    const eT       alpha = eT(1),
    const eT       beta  = eT(0),
    const typename arma_not_cx<eT>::result* junk = nullptr
    )
    {
    arma_debug_sigprint();
    arma_ignore(junk);
    
    gemm_emul_large<do_trans_A, do_trans_B, use_alpha, use_beta>::apply(C, A, B, alpha, beta);
    }
  
  
  
  template<typename eT>
  arma_hot
  inline
  static
  void
  apply
    (
          Mat<eT>& C,
    const Mat<eT>& A,
    const Mat<eT>& B,
    const eT       alpha = eT(1),
    const eT       beta  = eT(0),
    const typename arma_cx_only<eT>::result* junk = nullptr
    )
    {
    arma_debug_sigprint();
    arma_ignore(junk);
    
    // "better than nothing" handling of hermitian transposes for complex number matrices
    
    Mat<eT> tmp_A;
    Mat<eT> tmp_B;
    
    if(do_trans_A)  { op_htrans::apply_mat_noalias(tmp_A, A); }
    if(do_trans_B)  { op_htrans::apply_mat_noalias(tmp_B, B); }
    
    const Mat<eT>& AA = (do_trans_A == false) ? A : tmp_A;
    const Mat<eT>& BB = (do_trans_B == false) ? B : tmp_B;
    
    gemm_emul_large<false, false, use_alpha, use_beta>::apply(C, AA, BB, alpha, beta);
    }

  };



//! \brief
//! Wrapper for BLAS dgemm function, using template arguments to control the arguments passed to dgemm.
//! Matrix 'C' is assumed to have been set to the correct size (ie. taking into account transposes)

template<const bool do_trans_A=false, const bool do_trans_B=false, const bool use_alpha=false, const bool use_beta=false>
struct gemm
  {
  template<typename eT, typename TA, typename TB>
  inline
  static
  void
  apply_blas_type( Mat<eT>& C, const TA& A, const TB& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    arma_debug_sigprint();
    
    if( (A.n_rows <= 4) && (A.n_rows == A.n_cols) && (A.n_rows == B.n_rows) && (B.n_rows == B.n_cols) && (is_cx<eT>::no) ) 
      {
      if(do_trans_B == false)
        {
        gemm_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply(C, A, B, alpha, beta);
        }
      else
        {
        Mat<eT> BB(B.n_rows, B.n_rows, arma_nozeros_indicator());
        
        op_strans::apply_mat_noalias_tinysq(BB, B);
        
        gemm_emul_tinysq<do_trans_A, use_alpha, use_beta>::apply(C, A, BB, alpha, beta);
        }
      }
    else
      {
      #if defined(ARMA_USE_ATLAS)
        {
        arma_debug_print("atlas::cblas_gemm()");
        
        arma_conform_assert_atlas_size(A,B);
        
        atlas::cblas_gemm<eT>
          (
          atlas_CblasColMajor,
          (do_trans_A) ? ( is_cx<eT>::yes ? atlas_CblasConjTrans : atlas_CblasTrans ) : atlas_CblasNoTrans,
          (do_trans_B) ? ( is_cx<eT>::yes ? atlas_CblasConjTrans : atlas_CblasTrans ) : atlas_CblasNoTrans,
          C.n_rows,
          C.n_cols,
          (do_trans_A) ? A.n_rows : A.n_cols,
          (use_alpha) ? alpha : eT(1),
          A.mem,
          (do_trans_A) ? A.n_rows : C.n_rows,
          B.mem,
          (do_trans_B) ? C.n_cols : ( (do_trans_A) ? A.n_rows : A.n_cols ),
          (use_beta) ? beta : eT(0),
          C.memptr(),
          C.n_rows
          );
        }
      #elif defined(ARMA_USE_BLAS)
        {
        arma_debug_print("blas::gemm()");
        
        arma_conform_assert_blas_size(A,B);
        
        const char trans_A = (do_trans_A) ? ( is_cx<eT>::yes ? 'C' : 'T' ) : 'N';
        const char trans_B = (do_trans_B) ? ( is_cx<eT>::yes ? 'C' : 'T' ) : 'N';
        
        const blas_int m   = blas_int(C.n_rows);
        const blas_int n   = blas_int(C.n_cols);
        const blas_int k   = (do_trans_A) ? blas_int(A.n_rows) : blas_int(A.n_cols);
        
        const eT local_alpha = (use_alpha) ? alpha : eT(1);
        
        const blas_int lda = (do_trans_A) ? k : m;
        const blas_int ldb = (do_trans_B) ? n : k;
        
        const eT local_beta  = (use_beta) ? beta : eT(0);
        
        arma_debug_print( arma_str::format("blas::gemm(): trans_A: %c") % trans_A );
        arma_debug_print( arma_str::format("blas::gemm(): trans_B: %c") % trans_B );
        
        blas::gemm<eT>
          (
          &trans_A,
          &trans_B,
          &m,
          &n,
          &k,
          &local_alpha,
          A.mem,
          &lda,
          B.mem,
          &ldb,
          &local_beta,
          C.memptr(),
          &m
          );
        }
      #else
        {
        gemm_emul<do_trans_A, do_trans_B, use_alpha, use_beta>::apply(C,A,B,alpha,beta);
        }
      #endif
      }
    }
  
  
  
  //! immediate multiplication of matrices A and B, storing the result in C
  template<typename eT, typename TA, typename TB>
  inline
  static
  void
  apply( Mat<eT>& C, const TA& A, const TB& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    gemm_emul<do_trans_A, do_trans_B, use_alpha, use_beta>::apply(C,A,B,alpha,beta);
    }
  
  
  
  template<typename TA, typename TB>
  arma_inline
  static
  void
  apply
    (
          Mat<float>& C,
    const TA&         A,
    const TB&         B,
    const float alpha = float(1),
    const float beta  = float(0)
    )
    {
    gemm<do_trans_A, do_trans_B, use_alpha, use_beta>::apply_blas_type(C,A,B,alpha,beta);
    }
  
  
  
  template<typename TA, typename TB>
  arma_inline
  static
  void
  apply
    (
          Mat<double>& C,
    const TA&          A,
    const TB&          B,
    const double alpha = double(1),
    const double beta  = double(0)
    )
    {
    gemm<do_trans_A, do_trans_B, use_alpha, use_beta>::apply_blas_type(C,A,B,alpha,beta);
    }
  
  
  
  template<typename TA, typename TB>
  arma_inline
  static
  void
  apply
    (
          Mat< std::complex<float> >& C,
    const TA&                         A,
    const TB&                         B,
    const std::complex<float> alpha = std::complex<float>(1),
    const std::complex<float> beta  = std::complex<float>(0)
    )
    {
    gemm<do_trans_A, do_trans_B, use_alpha, use_beta>::apply_blas_type(C,A,B,alpha,beta);
    }
  
  
  
  template<typename TA, typename TB>
  arma_inline
  static
  void
  apply
    (
          Mat< std::complex<double> >& C,
    const TA&                          A,
    const TB&                          B,
    const std::complex<double> alpha = std::complex<double>(1),
    const std::complex<double> beta  = std::complex<double>(0)
    )
    {
    gemm<do_trans_A, do_trans_B, use_alpha, use_beta>::apply_blas_type(C,A,B,alpha,beta);
    }
  
  };



//! @}
