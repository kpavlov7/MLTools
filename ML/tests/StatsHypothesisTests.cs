using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;
using static ML.MathHelpers.StatHypothesis;

namespace ML.tests
{
    public class StatsHypothesisTests
    {
        /// <summary>
        /// http://www.r-tutor.com/elementary-statistics/goodness-fit/chi-squared-test-independence
        /// </summary>
        [Fact]
        public void chi()
        {
            var tbl = new ushort[,] {
                {7 , 1 , 3 },
                {87, 18, 84},
                {12, 3 , 4 },
                {9 , 1 , 7 }
            };

            var f1 = new ushort[] { 11, 189, 19, 17 };
            var f2 = new ushort[] { 115, 23, 98 };
            var n = 236;
            var alpha = 0.05;
            var rand = new Random(12);

            var chiHypothesis = new ChiHypothesis(alpha, rand);
            chiHypothesis.CalculateStatistics(tbl, f1, f2, n);

            Assert.Equal(5.4885, chiHypothesis.Statistics, 4);
            Assert.Equal(0.4828, chiHypothesis.PValue, 4);
            Assert.Equal(6, chiHypothesis.DF);
        }
    }
}
