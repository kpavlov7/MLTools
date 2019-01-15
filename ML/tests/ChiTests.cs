using ML.MathHelpers;
using System;
using Xunit;

namespace ML.tests
{
    public class ChiTests
    {
        private const int Seed = 12;

        [Fact]
        public void inverse_chi()
        {
            var rand = new Random(Seed);
            var n = 1000;
            var r = 10;

            var chi = new Chi(r, rand);

            var samples = chi.GenerateSample(n);

            Assert.True(Math.Abs(r     - samples.Mean())     < 0.5);
            Assert.True(Math.Abs(2 * r - samples.Variance()) < 1.5);
        }

        [Fact]
        public void chi()
        {
            var rand = new Random(Seed);
            var r = 10;

            var chi = new Chi(r, rand);

            Assert.True(Math.Abs(0.05  - chi.GetCdf(3.94))  < 0.001);
            Assert.True(Math.Abs(0.999 - chi.GetCdf(29.59)) < 0.001);
        }
    }
}
