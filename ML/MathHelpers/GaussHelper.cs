using System;

namespace ML.MathHelpers
{
    public static class GaussHelper
    {
        #region Standard Normal Cummulative Distribution Normal Density Function

        private const double _a1 = 0.254829592;
        private const double _a2 = -0.284496736;
        private const double _a3 = 1.421413741;
        private const double _a4 = -1.453152027;
        private const double _a5 = 1.061405429;
        private const double _p = 0.3275911;

        /// <summary>
        /// The Error Function: https://www.johndcook.com/blog/csharp_erf/;
        /// </summary>
        public static double Erf( double x )
        {
            // Save the sign of x;
            int sign = (x < 0) ? -1 : 1;

            x = Math.Abs(x);

            // A&S formula 7.1.26;
            double t = 1.0 / (1.0 + _p * x);
            double y = 1.0 - (((((_a5 * t + _a4) * t) + _a3) * t + _a2) * t + _a1) * t * Math.Exp(-x * x);

            return sign * y;
        }

        public static double Phi(double z)
        {
            var x = z / Math.Sqrt(2.0);

            return 0.5 * ( 1 + Erf(x));
        }
        #endregion

        #region Inverse Standard Normal Cummulative Distribution

        private const double _pLow = 0.02425;
        private const double _pHigh = 1 - _pLow;

        private static readonly double[] _a = new double[]
        { -3.969683028665376e+01 , 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };

        private static readonly double[] _b = new double[]
        { -5.447609879822406e+01 , 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01 };

        private static readonly double[] _c = new double[]
        { -7.784894002430293e-03 , -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };

        private static readonly double[] _d = new double[]
        { 7.784695709041462e-03 , 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00 };

        /// <summary>
        /// Inverse Normal CDF(Acklam's Approximation)
        /// https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
        /// </summary>
        public static double InvPhi( double p )
        {
            // Rational approximation for lower region;
            var x = 0.0;
            if( p > 0.0 && p < _pLow )
            {
                var q = Math.Sqrt(-2 * Math.Log(p));

                x = (((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) / 
                     ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1);
            }
            
            // Rational approximation for central region;

            if( p >= _pLow && p <= _pHigh )
            {
                var q = p - 0.5;
                var r = q * q;

                x = (((((_a[0] * r + _a[1]) * r + _a[2]) * r + _a[3]) * r + _a[4]) * r + _a[5]) * q /
                    (((((_b[0] * r + _b[1]) * r + _b[2]) * r + _b[3]) * r + _b[4]) * r + 1);

            }

            // Rational approximation for upper region;

            if( p > _pHigh && p < 1)
            {
                var q = Math.Sqrt(-2 * Math.Log(1 - p));
                x = - (((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) /
                       ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1);
            }

            return x;
        }
        #endregion
    }
}
