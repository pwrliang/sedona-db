/**********************************************************************
 *
 * GEOS - Geometry Engine Open Source
 * http://geos.osgeo.org
 *
 * Copyright (C) 2020 Crunchy Data
 *
 * This is free software; you can redistribute and/or modify it under
 * the terms of the GNU Lesser General Public Licence as published
 * by the Free Software Foundation.
 * See the COPYING file for more information.
 *
 **********************************************************************/

#pragma once
#include "gpuspatial/utils/cuda_utils.h"

namespace gpuspatial {

/**
 * \class DD
 *
 * \brief
 * Wrapper for DoubleDouble higher precision mathematics
 * operations.
 */
class DD {
 private:
  static constexpr double SPLIT = 134217729.0;  // 2^27+1, for IEEE double
  double hi;
  double lo;

  DEV_HOST_INLINE int signum() const {
    if (hi > 0) return 1;
    if (hi < 0) return -1;
    if (lo > 0) return 1;
    if (lo < 0) return -1;
    return 0;
  }

  DEV_HOST_INLINE DD rint() const {
    if (isNaN()) return *this;
    return (*this + 0.5).floor();
  }

 public:
  DEV_HOST DD(double p_hi, double p_lo) : hi(p_hi), lo(p_lo) {};
  DEV_HOST DD(double x) : hi(x), lo(0.0) {};
  DEV_HOST DD() : hi(0.0), lo(0.0) {};

  DEV_HOST_INLINE bool operator==(const DD& rhs) const {
    return hi == rhs.hi && lo == rhs.lo;
  }

  DEV_HOST_INLINE bool operator!=(const DD& rhs) const {
    return hi != rhs.hi || lo != rhs.lo;
  }

  DEV_HOST_INLINE bool operator<(const DD& rhs) const {
    return (hi < rhs.hi) || (hi == rhs.hi && lo < rhs.lo);
  }

  DEV_HOST_INLINE bool operator<=(const DD& rhs) const {
    return (hi < rhs.hi) || (hi == rhs.hi && lo <= rhs.lo);
  }

  DEV_HOST_INLINE bool operator>(const DD& rhs) const {
    return (hi > rhs.hi) || (hi == rhs.hi && lo > rhs.lo);
  }

  DEV_HOST_INLINE bool operator>=(const DD& rhs) const {
    return (hi > rhs.hi) || (hi == rhs.hi && lo >= rhs.lo);
  }

  DEV_HOST_INLINE friend DD operator+(const DD& lhs, const DD& rhs);
  DEV_HOST_INLINE friend DD operator+(const DD& lhs, double rhs);
  DEV_HOST_INLINE friend DD operator-(const DD& lhs, const DD& rhs);
  DEV_HOST_INLINE friend DD operator-(const DD& lhs, double rhs);
  DEV_HOST_INLINE friend DD operator*(const DD& lhs, const DD& rhs);
  DEV_HOST_INLINE friend DD operator*(const DD& lhs, double rhs);
  DEV_HOST_INLINE friend DD operator/(const DD& lhs, const DD& rhs);
  DEV_HOST_INLINE friend DD operator/(const DD& lhs, double rhs);

  DEV_HOST_INLINE static DD determinant(const DD& x1, const DD& y1, const DD& x2,
                                        const DD& y2) {
    return (x1 * y2) - (y1 * x2);
  }

  DEV_HOST_INLINE static DD determinant(double x1, double y1, double x2, double y2) {
    return determinant(DD(x1), DD(y1), DD(x2), DD(y2));
  }

  DEV_HOST_INLINE static DD abs(const DD& d) {
    if (d.isNaN()) return d;
    if (d.isNegative()) return d.negate();
    return d;
  }

  DEV_HOST_INLINE static DD pow(const DD& d, int exp) {
    if (exp == 0.0) return DD(1.0);

    DD r(d);
    DD s(1.0);
    int n = std::abs(exp);

    if (n > 1) {
      /* Use binary exponentiation */
      while (n > 0) {
        if (n % 2 == 1) {
          s.selfMultiply(r);
        }
        n /= 2;
        if (n > 0) r = r * r;
      }
    } else {
      s = r;
    }

    /* Compute the reciprocal if n is negative. */
    if (exp < 0) return s.reciprocal();
    return s;
  }

  DEV_HOST_INLINE static DD trunc(const DD& d) {
    if (d.isNaN()) return d;
    if (d.isPositive()) return d.floor();
    return d.ceil();
  }

  DEV_HOST_INLINE bool isNaN() const { return std::isnan(hi); }

  DEV_HOST_INLINE bool isNegative() const { return hi < 0.0 || (hi == 0.0 && lo < 0.0); }

  DEV_HOST_INLINE bool isPositive() const { return hi > 0.0 || (hi == 0.0 && lo > 0.0); }

  DEV_HOST_INLINE bool isZero() const { return hi == 0.0 && lo == 0.0; }

  DEV_HOST_INLINE double doubleValue() const { return hi + lo; }

  DEV_HOST_INLINE double ToDouble() const { return doubleValue(); }

  DEV_HOST_INLINE int intValue() const { return (int)hi; }

  DEV_HOST_INLINE DD negate() const {
    if (isNaN()) {
      return *this;
    }
    return DD(-hi, -lo);
  }

  DEV_HOST_INLINE DD reciprocal() const {
    double hc, tc, hy, ty, C, c, U, u;
    C = 1.0 / hi;
    c = SPLIT * C;
    hc = c - C;
    u = SPLIT * hi;
    hc = c - hc;
    tc = C - hc;
    hy = u - hi;
    U = C * hi;
    hy = u - hy;
    ty = hi - hy;
    u = (((hc * hy - U) + hc * ty) + tc * hy) + tc * ty;
    c = ((((1.0 - U) - u)) - C * lo) / hi;
    double zhi = C + c;
    double zlo = (C - zhi) + c;
    return DD(zhi, zlo);
  }

  DEV_HOST_INLINE DD floor() const {
    if (isNaN()) return *this;
    double fhi = std::floor(hi);
    double flo = 0.0;
    // Hi is already integral.  Floor the low word
    if (fhi == hi) {
      flo = std::floor(lo);
    }
    return DD(fhi, flo);
  }

  DEV_HOST_INLINE DD ceil() const {
    if (isNaN()) return *this;
    double fhi = std::ceil(hi);
    double flo = 0.0;
    // Hi is already integral.  Ceil the low word
    if (fhi == hi) {
      flo = std::ceil(lo);
    }
    return DD(fhi, flo);
  }

  DEV_HOST_INLINE void selfAdd(const DD& d) { selfAdd(d.hi, d.lo); }

  DEV_HOST_INLINE void selfAdd(double yhi, double ylo) {
    double H, h, T, t, S, s, e, f;
    S = hi + yhi;
    T = lo + ylo;
    e = S - hi;
    f = T - lo;
    s = S - e;
    t = T - f;
    s = (yhi - e) + (hi - s);
    t = (ylo - f) + (lo - t);
    e = s + T;
    H = S + e;
    h = e + (S - H);
    e = t + h;

    double zhi = H + e;
    double zlo = e + (H - zhi);
    hi = zhi;
    lo = zlo;
  }

  DEV_HOST_INLINE void selfAdd(double y) {
    double H, h, S, s, e, f;
    S = hi + y;
    e = S - hi;
    s = S - e;
    s = (y - e) + (hi - s);
    f = s + lo;
    H = S + f;
    h = f + (S - H);
    hi = H + h;
    lo = h + (H - hi);
  }

  DEV_HOST_INLINE void selfSubtract(const DD& d) { selfAdd(-d.hi, -d.lo); }

  DEV_HOST_INLINE void selfSubtract(double p_hi, double p_lo) { selfAdd(-p_hi, -p_lo); }

  DEV_HOST_INLINE void selfSubtract(double y) { selfAdd(-y, 0.0); }

  DEV_HOST_INLINE void selfMultiply(double yhi, double ylo) {
    double hx, tx, hy, ty, C, c;
    C = SPLIT * hi;
    hx = C - hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = hi - hx;
    hy = c - yhi;
    C = hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (hi * ylo + lo * yhi);
    double zhi = C + c;
    hx = C - zhi;
    double zlo = c + hx;
    hi = zhi;
    lo = zlo;
  }

  DEV_HOST_INLINE void selfMultiply(const DD& d) { selfMultiply(d.hi, d.lo); }

  DEV_HOST_INLINE void selfMultiply(double y) { selfMultiply(y, 0.0); }

  DEV_HOST_INLINE void selfDivide(double yhi, double ylo) {
    double hc, tc, hy, ty, C, c, U, u;
    C = hi / yhi;
    c = SPLIT * C;
    hc = c - C;
    u = SPLIT * yhi;
    hc = c - hc;
    tc = C - hc;
    hy = u - yhi;
    U = C * yhi;
    hy = u - hy;
    ty = yhi - hy;
    u = (((hc * hy - U) + hc * ty) + tc * hy) + tc * ty;
    c = ((((hi - U) - u) + lo) - C * ylo) / yhi;
    u = C + c;
    hi = u;
    lo = (C - u) + c;
  }

  DEV_HOST_INLINE void selfDivide(const DD& d) { selfDivide(d.hi, d.lo); }

  DEV_HOST_INLINE void selfDivide(double y) { selfDivide(y, 0.0); }
};

DEV_HOST_INLINE DD operator+(const DD& lhs, const DD& rhs) {
  DD rv(lhs);
  rv.selfAdd(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator+(const DD& lhs, double rhs) {
  DD rv(lhs);
  rv.selfAdd(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator-(const DD& lhs, const DD& rhs) {
  DD rv(lhs);
  rv.selfSubtract(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator-(const DD& lhs, double rhs) {
  DD rv(lhs);
  rv.selfSubtract(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator*(const DD& lhs, const DD& rhs) {
  DD rv(lhs);
  rv.selfMultiply(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator*(const DD& lhs, double rhs) {
  DD rv(lhs);
  rv.selfMultiply(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator/(const DD& lhs, const DD& rhs) {
  DD rv(lhs);
  rv.selfDivide(rhs);
  return rv;
}

DEV_HOST_INLINE DD operator/(const DD& lhs, double rhs) {
  DD rv(lhs);
  rv.selfDivide(rhs);
  return rv;
}
}  // namespace gpuspatial
