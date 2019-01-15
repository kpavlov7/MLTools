using System;
namespace ML.MathHelpers
{
    public class NormalDistribution
    {
        public double Sigma { get; set; }
        public double Mean { get; set; }

        private readonly double SigSqrt2;

        private Random Random;

        public NormalDistribution(double mean, double sigma, Random random)
        {
            Sigma = sigma;
            Mean = mean;
            SigSqrt2 = Sigma * Math.Sqrt(2);
            Random = random;
        }

        public double GetPdf(double x)
        {

            return (1.0 / (SigSqrt2 * Math.Sqrt(Math.PI)))
                   * Math.Exp(-((x - Mean) * (x - Mean)) / (SigSqrt2 * SigSqrt2));
        }

        public double GetCdf(double x)
        {
            return GaussHelper.Phi((x - Mean) / Sigma);
        }

        public double GetInvCdf(double p)
        {
            return Sigma * GaussHelper.InvPhi(p) + Mean;
        }

        public float[] GenerateSample(int n)
        {
            var samples = new float[n];

            for(var i = 0; i < samples.Length; i++)
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
