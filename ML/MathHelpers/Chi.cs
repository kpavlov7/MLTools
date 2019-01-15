using System;
using System.Collections.Generic;
using System.Text;

namespace ML.MathHelpers
{
    /// <summary>
    /// Chi^2 Distribution
    /// </summary>
    public class Chi
    {
        /// <summary>
        /// Degrees of freedom;
        /// </summary>
        public int R;

        private Random Random;

        public Chi(int r, Random random)
        {
            R = r;
            Random = random;
        }

        public double GetPdf(double x)
        {
            return Math.Pow(x, (R * 0.5) - 1d) * Math.Exp(-x * 0.5) 
                   / (GammaHelper.Gamma(0.5 * R) * Math.Pow(2, R * 0.5));
        }

        public double GetCdf(double x)
        {
            return GammaHelper.RegularizedGammaP(R * 0.5, x * 0.5);
        }

        public double GetInvCdf(double p)
        {
            return 2 * GammaHelper.InverseRegularizedGammaP(R * 0.5, p);
        }

        public float[] GenerateSample(int n)
        {
            var samples = new float[n];

            for (var i = 0; i < samples.Length; i++)
            {
                samples[i] = (float)GetInvCdf(Random.NextDouble());
            }

            return samples;
        }

        public double GenerateValue()
        {
            return GetInvCdf(Random.NextDouble());
        }
    }
}
