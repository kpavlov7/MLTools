using System;

namespace ML.MathHelpers
{
    public static class RandomHelper
    {
        public static double NextDouble(this Random random, double min, double max){
            if(max < min)
            {
                throw new ArgumentException("'Max' should be greater than 'Min'");
            }
            var delta = max - min;
            return random.NextDouble() * delta + min;
        }
    }
}
