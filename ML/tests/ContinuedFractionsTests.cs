using ML.MathHelpers;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace ML.tests
{
    public class ContinuedFractionsTests
    {
        private const int DecimalPrecision = 6;
        [Fact]
        ///https://oeis.org/A001203
        public void continued_fration_pi()
        {
            var seq = new[] {3d, 7d, 15d, 1d, 292d, 1d, 1d};
            // Define the argumetns for the Continued Fraction.
            double fa(int n, double[] a) => a[n];
            double fb(int n, double[] a) => 1;

            var cf = new ContinuedFraction(fa,fb);

            var pi = cf.Evaluate(seq.Length, seq);
            Assert.Equal(pi, Math.PI, DecimalPrecision);
        }
    }
}
