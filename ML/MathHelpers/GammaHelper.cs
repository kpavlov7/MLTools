using System;

namespace ML.MathHelpers
{
    public static class GammaHelper
    {
        // http://en.wikipedia.org/wiki/Euler-Mascheroni_constant Euler-Mascheroni constant;
        public const double GammaCoef = 0.577215664901532860606512090082;

        // The value of the g constant in the Lanczos approximation;
        private const double LanczosG = 607.0 / 128.0;


        // Maximum allowed numerical error.
        private const double Epsilon = 10e-15;

        // Lanczos coefficients
        private static readonly double[] LanczosCoef = {
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        .33994649984811888699e-4,
        .46523628927048575665e-4,
        -.98374475304879564677e-4,
        .15808870322491248884e-3,
        -.21026444172410488319e-3,
        .21743961811521264320e-3,
        -.16431810653676389022e-3,
        .84418223983852743293e-4,
        -.26190838401581408670e-4,
        .36899182659531622704e-5,
    };

        // Avoid repeated computation of log of 2 PI in logGamma
        private static readonly double HalfLog2Pi = 0.5 * Math.Log(2.0 * Math.PI);

        // The constant value of &radic;(2&pi;)
        private static readonly double SqrtTwoPi = 2.506628274631000502;

        // limits for switching algorithm in digamma C limit.
        private static readonly double CLimit = 49;

        // S limit.
        private static readonly double SLimit = 1e-5;


        // Constants for the computation of double invGamma1pm1(double).
        // Copied from DGAM1 in the NSWC library.

        // The constant A0 defined in DGAM1
        private static readonly double A0 = .611609510448141581788E-08;

        // The constant A1 defined in DGAM1
        private static readonly double A1 = .624730830116465516210E-08;

        // The constant B1 defined in DGAM1
        private static readonly double B1 = .203610414066806987300E+00;

        // The constant B2 defined in DGAM1
        private static readonly double B2 = .266205348428949217746E-01;

        // The constant B3 defined in DGAM1
        private static readonly double B3 = .493944979382446875238E-03;

        // The constant B4 defined in DGAM1
        private static readonly double B4 = -.851419432440314906588E-05;

        // The constant B5 defined in DGAM1
        private static readonly double B5 = -.643045481779353022248E-05;

        // The constant B6 defined in DGAM1
        private static readonly double B6 = .992641840672773722196E-06;

        // The constant B7 defined in DGAM1
        private static readonly double B7 = -.607761895722825260739E-07;

        // The constant B8 defined in DGAM1
        private static readonly double B8 = .195755836614639731882E-09;

        // The constant P0 defined in DGAM1
        private static readonly double P0 = .6116095104481415817861E-08;

        // The constant P1 defined in DGAM1
        private static readonly double P1 = .6871674113067198736152E-08;

        // The constant P2 defined in DGAM1
        private static readonly double P2 = .6820161668496170657918E-09;

        // The constant P3 defined in DGAM1
        private static readonly double P3 = .4686843322948848031080E-10;

        // The constant P4 defined in DGAM1
        private static readonly double P4 = .1572833027710446286995E-11;

        // The constant P5 defined in DGAM1
        private static readonly double P5 = -.1249441572276366213222E-12;

        // The constant P6 defined in DGAM1
        private static readonly double P6 = .4343529937408594255178E-14;

        // The constant Q1 defined in DGAM1
        private static readonly double Q1 = .3056961078365221025009E+00;

        // The constant Q2 defined in DGAM1
        private static readonly double Q2 = .5464213086042296536016E-01;

        // The constant Q3 defined in DGAM1
        private static readonly double Q3 = .4956830093825887312020E-02;

        // The constant Q4 defined in DGAM1
        private static readonly double Q4 = .2692369466186361192876E-03;

        // The constant C defined in DGAM1
        private static readonly double C = -.422784335098467139393487909917598E+00;

        // The constant C0 defined in DGAM1
        private static readonly double C0 = .577215664901532860606512090082402E+00;

        // The constant C1 defined in DGAM1
        private static readonly double C1 = -.655878071520253881077019515145390E+00;

        // The constant C2 defined in DGAM1
        private static readonly double C2 = -.420026350340952355290039348754298E-01;

        // The constant C3 defined in DGAM1
        private static readonly double C3 = .166538611382291489501700795102105E+00;

        // The constant C4 defined in DGAM1
        private static readonly double C4 = -.421977345555443367482083012891874E-01;

        // The constant C5 defined in DGAM1
        private static readonly double C5 = -.962197152787697356211492167234820E-02;

        // The constant C6 defined in DGAM1
        private static readonly double C6 = .721894324666309954239501034044657E-02;

        // The constant C7 defined in DGAM1
        private static readonly double C7 = -.116516759185906511211397108401839E-02;

        // The constant C8 defined in DGAM1
        private static readonly double C8 = -.215241674114950972815729963053648E-03;

        // The constant C9 defined in DGAM1
        private static readonly double C9 = .128050282388116186153198626328164E-03;

        // The constant C10 defined in DGAM1
        private static readonly double C10 = -.201348547807882386556893914210218E-04;

        // The constant C11 defined in DGAM1
        private static readonly double C11 = -.125049348214267065734535947383309E-05;

        // The constant C12 defined in DGAM1
        private static readonly double C12 = .113302723198169588237412962033074E-05;

        // The constant C13 defined in DGAM1
        private static readonly double C13 = -.205633841697760710345015413002057E-06;

        #region LogGamma Function

        /// <summary>
        /// Double precision  implementation in the NSWC Library of Mathematics Subroutines
        /// Gamma: http://mathworld.wolfram.com/GammaFunction.html
        /// Lanczos Approximation: http://mathworld.wolfram.com/LanczosApproximation.html
        /// Paul Godfrey, A note on the computation of the convergent Lanczos complex Gamma
        /// approximation: http://my.fit.edu/~gabdo/gamma.txt
        /// </summary>
        public static double LogGamma(double x)
        {
            double ret;

            if (double.IsNaN(x) || (x <= 0.0))
            {
                ret = double.NaN;
            }
            else if (x < 0.5)
            {
                return LogGamma1p(x) - Math.Log(x);
            }
            else if (x <= 2.5)
            {
                return LogGamma1p((x - 0.5) - 0.5);
            }
            else if (x <= 8.0)
            {
                int n = (int)Math.Floor(x - 1.5);
                double prod = 1.0;
                for (int i = 1; i <= n; i++)
                {
                    prod *= x - i;
                }
                return LogGamma1p(x - (n + 1)) + Math.Log(prod);
            }
            else
            {
                double sum = Lanczos(x);
                double tmp = x + LanczosG + .5;
                ret = ((x + .5) * Math.Log(tmp)) - tmp +
                    HalfLog2Pi + Math.Log(sum / x);
            }

            return ret;
        }
        #endregion

        #region Regularized Gamma Function

        ///<summary>
        /// Returns the regularized gamma function P(a, x).
        ///</summary>
        public static double RegularizedGammaP(double a, double x)
        {
            return RegularizedGammaP(a, x, Epsilon, int.MaxValue);
        }

        ///<summary>
        /// Returns the regularized gamma function P(a, x).
        /// Regularized Gamma Function, equation (1): http://mathworld.wolfram.com/RegularizedGammaFunction.html
        /// Incomplete Gamma Function, equation (4): http://mathworld.wolfram.com/IncompleteGammaFunction.html
        /// Confluent Hypergeometric Function of the First Kind, equation (1):
        /// http://mathworld.wolfram.com/ConfluentHypergeometricFunctionoftheFirstKind.html
        ///</summary>
        public static double RegularizedGammaP(
            double a,
            double x,
            double epsilon,
            int maxIterations)
        {
            double ret;

            if (double.IsNaN(a) || double.IsNaN(x) || (a <= 0.0) || (x < 0.0))
            {
                ret = Double.NaN;
            }
            else if (x == 0.0)
            {
                ret = 0.0;
            }
            else if (x >= a + 1)
            {
                // use regularizedGammaQ because it should converge faster in this
                // case.
                ret = 1.0 - RegularizedGammaQ(a, x, epsilon, maxIterations);
            }
            else
            {
                // calculate series
                double n = 0.0; // current element index
                double an = 1.0 / a; // n-th element in the series
                double sum = an; // partial sum
                while (Math.Abs(an / sum) > epsilon &&
                       n < maxIterations &&
                       sum < double.PositiveInfinity)
                {
                    // compute next element in the series
                    n += 1.0;
                    an *= x / (a + n);

                    // update partial sum
                    sum += an;
                }
                if (n >= maxIterations)
                {
                    throw new ArgumentException("Max iteration number exceeded.", nameof(n));
                }
                else if (double.IsInfinity(sum))
                {
                    ret = 1.0;
                }
                else
                {
                    ret = Math.Exp(-x + (a * Math.Log(x)) - LogGamma(a)) * sum;
                }
            }

            return ret;
        }

        ///</summary> Returns the regularized gamma function Q(a, x) = 1 - P(a, x).///<summary>
        public static double RegularizedGammaQ(double a, double x)
        {
            return RegularizedGammaQ(a, x, Epsilon, int.MaxValue);
        }

        ///<summary>
        /// Returns the regularized gamma function Q(a, x) = 1 - P(a, x).
        /// The implementation of this method is based on:
        /// Regularized Gamma Function</a>, equation (1):http://mathworld.wolfram.com/RegularizedGammaFunction.html
        /// Regularized incomplete gamma function; Continued fraction representations (formula 06.08.10.0003):
        /// http://functions.wolfram.com/GammaBetaErf/GammaRegularized/10/0003/
        ///</summary>
        public static double RegularizedGammaQ(
            double a,
            double x,
            double epsilon,
            int maxIterations)
        {
            double ret;

            if (double.IsNaN(a) || double.IsNaN(x) || (a <= 0.0) || (x < 0.0))
            {
                ret = double.NaN;
            }
            else if (x == 0.0)
            {
                ret = 1.0;
            }
            else if (x < a + 1.0)
            {
                // Use regularizedGammaP because it should converge faster in this case.
                ret = 1.0 - RegularizedGammaP(a, x, epsilon, maxIterations);
            }
            else
            {
                // Define the argumetns for the Continued Fraction.
                double fa(int n, double[] xa) => ((2.0 * n) + 1.0) - xa[1] + xa[0];
                double fb(int n, double[] xa) => n * (xa[1] - n);

                var cf = new ContinuedFraction(fa, fb);

                ret = 1.0 / cf.Evaluate(epsilon, maxIterations, x, a);
                ret = Math.Exp(-x + (a * Math.Log(x)) - LogGamma(a)) * ret;
            }

            return ret;
        }

        public static double InverseRegularizedGammaP(double a, double p)
        {
            if (a <= 0.0)
            {
                throw new ArgumentException("The value should be positive.", nameof(a));
            }

            double x, err, t, u, pp;
            double lna1 = 0.0;
            double afac = 0.0;
            double a1 = a - 1;
            double gln = LogGamma(a);

            if (p >= 1.0)
            {
                return Math.Max(100, a + 100 * Math.Sqrt(a));
            }

            if (p <= 0.0)
            {
                return 0.0;
            }

            if (a > 1.0)
            {
                lna1 = Math.Log(a1);
                afac = Math.Exp(a1 * (lna1 - 1) - gln);
                pp = (p < 0.5) ? p : 1 - p;
                t = Math.Sqrt(-2 * Math.Log(pp));

                x = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;

                if (p < 0.5)
                {
                    x = -x;
                }
                x = Math.Max(1e-3, a * Math.Pow(1 - 1 / (9 * a) - x / (3 * Math.Sqrt(a)), 3));
            }
            else
            {
                t = 1.0 - a * (0.253 + a * 0.12);
                if (p < t)
                {
                    x = Math.Pow(p / t, 1 / a);
                }
                else
                {
                    x = 1 - Math.Log(1 - (p - t) / (1 - t));
                }
            }
            for (int j = 0; j < 12; j++)
            {
                if (x <= 0.0)
                {
                    return 0.0;
                }
                err = RegularizedGammaP(a, x) - p;
                if (a > 1)
                {
                    t = afac * Math.Exp(-(x - a1) + a1 * (Math.Log(x) - lna1));
                }
                else
                {
                    t = Math.Exp(-x + a1 * Math.Log(x) - gln);
                }
                u = err / t;
                x -= (t = u / (1 - 0.5 * Math.Min(1, u * ((a - 1) / x - 1))));
                if (x <= 0)
                {
                    x = 0.5 * (x + t);
                }
                if (Math.Abs(t) < Epsilon * x)
                {
                    break;
                }
            }
            return x;
        }
        #endregion

        #region Gamma Function
        ///<summary>
        /// Computes the digamma function of x.
        /// This is an independently written implementation of the algorithm described in
        /// Jose Bernardo, Algorithm AS 103: Psi (Digamma) Function, Applied Statistics, 1976.
        ///
        /// Some of the constants have been changed to increase accuracy at the moderate expense
        /// of run-time.  The result should be accurate to within 10^-8 absolute tolerance for
        /// x >= 10^-5 and within 10^-8 relative tolerance for x > 0.
        ///
        /// Performance for large negative values of x will be quite expensive (proportional to
        /// |x|).  Accuracy for negative values of x should be about 10^-8 absolute for results
        /// less than 10^5 and 10^-8 relative for results larger than that.
        ///
        /// digamma(x) to within 10-8 relative or absolute error whichever is smaller.
        /// Digamma: http://en.wikipedia.org/wiki/Digamma_function
        /// Bernardo&apos;s original article: http://www.uv.es/~bernardo/1976AppStatist.pdf
        ///</summary>
        public static double Digamma(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
            {
                return x;
            }

            if (x > 0 && x <= SLimit)
            {
                // use method 5 from Bernardo AS103
                // accurate to O(x)
                return -GammaCoef - 1 / x;
            }

            if (x >= CLimit)
            {
                // use method 4 (accurate to O(1/x^8)
                double inv = 1 / (x * x);
                //            1       1        1         1
                // log(x) -  --- - ------ + ------- - -------
                //           2 x   12 x^2   120 x^4   252 x^6
                return Math.Log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
            }

            return Digamma(x + 1) - 1 / x;
        }

        /// <summary>
        /// Computes the trigamma function of x.
        /// This function is derived by taking the derivative of the implementation
        /// of digamma. http://en.wikipedia.org/wiki/Trigamma_function
        /// </summary>
        public static double Trigamma(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
            {
                return x;
            }

            if (x > 0 && x <= SLimit)
            {
                return 1 / (x * x);
            }

            if (x >= CLimit)
            {
                double inv = 1 / (x * x);
                //  1    1      1       1       1
                //  - + ---- + ---- - ----- + -----
                //  x      2      3       5       7
                //      2 x    6 x    30 x    42 x
                return 1 / x + inv / 2 + inv / x * (1.0 / 6 - inv * (1.0 / 30 + inv / 42));
            }

            return Trigamma(x + 1) + 1 / (x * x);
        }

        /// <summary>
        /// Returns the Lanczos approximation used to compute the gamma function.
        /// The Lanczos approximation is related to the Gamma function by the
        /// following equation
        /// gamma(x) = sqrt(2 * pi) / x* (x + g + 0.5) ^ (x + 0.5)
        /// exp(-x - g - 0.5) * lanczos(x)
        /// </summary>
        public static double Lanczos(double x)
        {
            double sum = 0.0;
            for (int i = LanczosCoef.Length - 1; i > 0; --i)
            {
                sum += LanczosCoef[i] / (x + i);
            }
            return sum + LanczosCoef[0];
        }

        /// <summary>
        /// Returns the value of 1 / &Gamma;(1 + x) - 1 for -0&#46;5 &le; x &le;
        /// 1&#46;5. This implementation is based on the double precision
        /// implementation in the NSWC Library of Mathematics Subroutines,
        ///</summary>
        public static double InvGamma1pm1(double x)
        {

            if (x < -0.5)
            {
                throw new ArgumentException("The number is too small.", nameof(x));
            }
            if (x > 1.5)
            {
                throw new ArgumentException("The number is too large.", nameof(x));
            }

            double ret;
            double t = x <= 0.5 ? x : (x - 0.5) - 0.5;
            if (t < 0.0)
            {
                double a = A0 + t * A1;
                double b = B8;
                b = B7 + t * b;
                b = B6 + t * b;
                b = B5 + t * b;
                b = B4 + t * b;
                b = B3 + t * b;
                b = B2 + t * b;
                b = B1 + t * b;
                b = 1.0 + t * b;

                double c = C13 + t * (a / b);
                c = C12 + t * c;
                c = C11 + t * c;
                c = C10 + t * c;
                c = C9 + t * c;
                c = C8 + t * c;
                c = C7 + t * c;
                c = C6 + t * c;
                c = C5 + t * c;
                c = C4 + t * c;
                c = C3 + t * c;
                c = C2 + t * c;
                c = C1 + t * c;
                c = C + t * c;
                if (x > 0.5)
                {
                    ret = t * c / x;
                }
                else
                {
                    ret = x * ((c + 0.5) + 0.5);
                }
            }
            else
            {
                double p = P6;
                p = P5 + t * p;
                p = P4 + t * p;
                p = P3 + t * p;
                p = P2 + t * p;
                p = P1 + t * p;
                p = P0 + t * p;

                double q = Q4;
                q = Q3 + t * q;
                q = Q2 + t * q;
                q = Q1 + t * q;
                q = 1.0 + t * q;

                double c = C13 + (p / q) * t;
                c = C12 + t * c;
                c = C11 + t * c;
                c = C10 + t * c;
                c = C9 + t * c;
                c = C8 + t * c;
                c = C7 + t * c;
                c = C6 + t * c;
                c = C5 + t * c;
                c = C4 + t * c;
                c = C3 + t * c;
                c = C2 + t * c;
                c = C1 + t * c;
                c = C0 + t * c;

                if (x > 0.5)
                {
                    ret = (t / x) * ((c - 0.5) - 0.5);
                }
                else
                {
                    ret = x * c;
                }
            }

            return ret;
        }



        /// <summary>
        /// Returns the value of log &Gamma;(1 + x) for -0&#46;5 &le; x &le; 1&#46;5.
        /// This implementation is based on the double precision implementation in
        /// the NSWC Library of Mathematics Subroutines</summary>
        public static double LogGamma1p(double x)
        {

            if (x < -0.5)
            {
                throw new ArgumentException("Too small value.", nameof(x));
            }
            if (x > 1.5)
            {
                throw new ArgumentException("Too large value.", nameof(x));
            }

            return -Math.Log(InvGamma1pm1(x) + 1);
        }

        /// <summary>
        /// Returns the value of Gamma(x). Based on the NSWC Library of
        /// Mathematics Subroutines double precision implementation,
        /// </summary>
        public static double Gamma(double x)
        {

            if ((x == Math.Round(x)) && (x <= 0.0))
            {
                return double.NaN;
            }

            double ret;
            double absX = Math.Abs(x);
            if (absX <= 20.0)
            {
                if (x >= 1.0)
                {

                    // From the recurrence relation Gamma(x) = (x - 1) * ... * (x - n) * Gamma(x - n),
                    // then Gamma(t) = 1 / [1 + invGamma1pm1(t - 1)], where t = x - n. This means that t must satisfy
                    //-0.5 <= t - 1 <= 1.5.
                    double prod = 1.0;
                    double t = x;
                    while (t > 2.5)
                    {
                        t -= 1.0;
                        prod *= t;
                    }
                    ret = prod / (1.0 + InvGamma1pm1(t - 1.0));
                }
                else
                {

                    // From the recurrence relation Gamma(x) = Gamma(x + n + 1) / [x * (x + 1) * ... * (x + n)]
                    // then Gamma(x + n + 1) = 1 / [1 + invGamma1pm1(x + n)], which requires -0.5 <= x + n <= 1.5.
                    double prod = x;
                    double t = x;
                    while (t < -0.5)
                    {
                        t += 1.0;
                        prod *= t;
                    }
                    ret = 1.0 / (prod * (1.0 + InvGamma1pm1(t)));
                }
            }
            else
            {
                double y = absX + LanczosG + 0.5;
                double gammaAbs = SqrtTwoPi / absX *
                                        Math.Pow(y, absX + 0.5) *
                                        Math.Exp(-y) * Lanczos(absX);
                if (x > 0.0)
                {
                    ret = gammaAbs;
                }
                else
                {

                    // From the reflection formula Gamma(x) * Gamma(1 - x) * sin(pi * x) = pi
                    // and the recurrence relation Gamma(1 - x) = -x * Gamma(-x) it is found
                    // Gamma(x) = -pi / [x * sin(pi * x) * Gamma(-x)].
                    ret = -Math.PI / (x * Math.Sin(Math.PI * x) * gammaAbs);
                }
            }
            return ret;
        }
    }
    #endregion
}

