using System;
using System.Collections.Generic;
using System.Text;

namespace ML.MathHelpers
{
    public class ContinuedFraction
    {
        private static readonly double Epsilon = 10e-9;
        private readonly Func<int, double[], double> _fa;
        private readonly Func<int, double[], double> _fb;

        public ContinuedFraction(Func<int, double[], double> fa, Func<int, double[], double> fb)
        {
            _fa = fa;
            _fb = fb;
        }

        protected double GetA(Func<int, double[], double> f, int n, params double[] x)
        {
            return f(n, x);
        }

        protected double GetB(Func<int, double[], double> f, int n, params double[] x)
        {
            return f(n, x);
        }

        public double Evaluate(params double[] x)
        {
            return Evaluate(Epsilon, int.MaxValue, x);
        }

        public double Evaluate(double epsilon, params double[] x)
        {
            return Evaluate(epsilon, int.MaxValue, x);
        }

        public double Evaluate(int maxIterations, params double[] x)
        {
            return Evaluate(Epsilon, maxIterations, x);
        }

        ///<summary>
        /// Evaluates the continued fraction at the value x.
        /// The implementation of this method is based on the modified Lentz algorithm as described
        /// on page 18 ff. in:
        /// I. J. Thompson,  A. R. Barnett. "Coulomb and Bessel Functions of Complex Arguments and Order."
        /// http://www.fresco.org.uk/papers/Thompson-JCP64p490.pdf
        /// Continued Fraction @ MathWorld: http://mathworld.wolfram.com/ContinuedFraction.html
        ///</summary>
        public double Evaluate(double epsilon, int maxIterations, params double[] x)
        {

            double hPrev = GetA(_fa, 0, x);
            double eps = 1e-50;

            //zero check
            if (Math.Abs(hPrev - 0.0) < eps)
            {
                hPrev = eps;
            }

            int n = 1;
            double dPrev = 0.0;
            double cPrev = hPrev;
            double hN = hPrev;

            while (n < maxIterations)
            {
                double a = GetA(_fa, n, x);
                double b = GetB(_fb, n, x);

                double dN = a + b * dPrev;
                if (Math.Abs(dN - 0.0) < eps)
                {
                    dN = eps;
                }
                double cN = a + b / cPrev;
                if (Math.Abs(cN - 0.0) < eps)
                {
                    cN = eps;
                }

                dN = 1 / dN;
                double deltaN = cN * dN;
                hN = hPrev * deltaN;

                if (double.IsInfinity(hN))
                {
                    throw new Exception("It doesn't converge.");
                }
                if (double.IsNaN(hN))
                {
                    throw new Exception("It doesn't converge.");
                }

                if (Math.Abs(deltaN - 1.0) < epsilon)
                {
                    break;
                }

                dPrev = dN;
                cPrev = cN;
                hPrev = hN;
                n++;
            }

            if (n >= maxIterations)
            {
                throw new Exception("The maximum number of iterations exceeded.");
            }

            return hN;
        }
    }
}
