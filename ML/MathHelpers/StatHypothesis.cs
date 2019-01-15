using System;
using System.Collections.Generic;

namespace ML.MathHelpers
{
    public class StatHypothesis
    {
        public class ChiHypothesis
        {
            /// <summary>
            /// Significance Level;
            /// </summary>
            public double Alpha;
            /// <summary>
            /// Degrees of Freedom;
            /// </summary>
            public int DF;
            private Random Random;

            public double Statistics;
            public double PValue;
            public bool IsNullHypothesisAccepted;

            public ChiHypothesis(double alpha, Random random)
            {
                Alpha = alpha;
                Random = random;
            }

            public void EvaluateHypothesis(int[] occurrences1, int[] occurrences2)
            {
                if (occurrences1.Length != occurrences2.Length)
                {
                    throw new ArgumentException("Both occurrence array should be with the same length.");
                }

                var categoryMapper1 = new Dictionary<int, int>();
                var categoryMapper2 = new Dictionary<int, int>();

                var conditionalFreq1 = new List<ushort>();
                var conditionalFreq2 = new List<ushort>();

                for (int i = 0; i < occurrences1.Length; i++)
                {
                    if (!categoryMapper1.ContainsKey(occurrences1[i]))
                    {
                        categoryMapper1[occurrences1[i]] = conditionalFreq1.Count;
                        conditionalFreq1.Add(1);
                    }
                    else
                    {
                        conditionalFreq1[categoryMapper1[occurrences1[i]]]++;
                    }

                    if (!categoryMapper2.ContainsKey(occurrences2[i]))
                    {
                        categoryMapper2[occurrences2[i]] = conditionalFreq2.Count;
                        conditionalFreq2.Add(1);
                    }
                    else
                    {
                        conditionalFreq2[categoryMapper2[occurrences2[i]]]++;
                    }
                }

                var freqTable = new ushort[conditionalFreq1.Count, conditionalFreq2.Count];

                for (int i = 0; i < occurrences1.Length; i++)
                {
                    freqTable[categoryMapper1[occurrences1[i]], categoryMapper2[occurrences2[i]]]++;
                }

                CalculateStatistics(freqTable, conditionalFreq1.ToArray(), conditionalFreq2.ToArray(), occurrences1.Length);
            }

            public void CalculateStatistics(ushort[,] f, ushort[] f1, ushort[] f2, double n)
            {
                Statistics = 0d;
                DF = (f1.Length - 1) * (f2.Length - 1);
                for (int i = 0; i < f1.Length; i++)
                {
                    for (int j = 0; j < f2.Length; j++)
                    {
                        var ef = (f1[i] * f2[j]) / n;
                        Statistics += ((f[i, j] - ef) * (f[i, j] - ef)) / ef;
                    }
                }

                var chiSqr = new Chi(DF, Random);
                PValue = 1d - chiSqr.GetCdf(Statistics);
                IsNullHypothesisAccepted = PValue <= Alpha;
            }
        }
    }
}