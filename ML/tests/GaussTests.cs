using ML.MathHelpers;
using System;
using Xunit;

namespace ML.tests
{
    public class GaussTests
    {
        private const int Seed = 12;

        [Fact]
        public void inverse_gauss()
        {
            var rand = new Random(Seed);
            var n = 1000;
            var samples = new float[n];

            for(var i = 0; i < samples.Length; i++)
            {
                samples[i] = (float)GaussHelper.InvPhi(rand.NextDouble());
            }

            Assert.True(Math.Abs(0 - samples.Mean()) < 0.1);
            Assert.True(Math.Abs(1.0 - samples.Variance()) < 0.5);
        }

        [Fact]
        public void gauss()
        {
            // Probability check
            Assert.True(Math.Abs(0.5     - GaussHelper.Phi(0))   < 0.01);
            Assert.True(Math.Abs(0.75804 - GaussHelper.Phi(0.7)) < 0.01);
            Assert.True(Math.Abs(0.95543 - GaussHelper.Phi(1.7)) < 0.01);
            Assert.True(Math.Abs(0.99997 - GaussHelper.Phi(4))   < 0.01);
        }
    }
}
