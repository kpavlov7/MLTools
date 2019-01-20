using System.Collections.Generic;

namespace Benchmark
{
    public class BenchMetrics
    {
        public enum Metrics
        {
            /// <summary>
            /// Ratio of correct classifications
            /// </summary>
            ClassAccuracy,
            /// <summary>
            /// Metric between 0 and 1 use for unsupervised classifiers:
            /// https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
            /// </summary>
            Purity
        }

        public class MetricsInput
        {

            public int ClassLabel;
            public int ClassPrediction;

            public double RegressLabel;
            public double RegressPrediction;

            public void AddResult(int prediction, int label)
            {
                ClassPrediction = prediction;
                ClassLabel = label;
            }
        }

        public class MetricsGenerator
        {
            private Dictionary<Metrics, IMetric> _bag;
            private MetricsInput _input;

            public MetricsGenerator()
            {
                _bag = new Dictionary<Metrics, IMetric>();
                _input = new MetricsInput();
            }

            public void AddResult(int prediction, int label)
            {
                _input.AddResult(prediction, label);
            }

            public void UpdateMetrics()
            {
                foreach(var metric in _bag)
                {
                    metric.Value.Add(_input);
                }
            }

            public void Add(Metrics metric)
            {
                switch (metric)
                {
                    case Metrics.Purity:
                        _bag.Add(metric, new Purity());
                        break;
                }
            }

            public double? GetMetric(Metrics metric)
            {
                if (_bag.ContainsKey(metric)) return _bag[metric].Get();
                else return null;
            }
        }

        public class Purity: IMetric
        {
            private readonly Dictionary<int, Dictionary<int, int>> _clusterDict;
            private int _count;

            public Purity()
            {
                _clusterDict = new Dictionary<int, Dictionary<int, int>>();
            }

            public void Add(MetricsInput input)
            {
                var cluster = input.ClassPrediction;
                var label = input.ClassLabel;

                if (_clusterDict.ContainsKey(cluster))
                {
                    if (_clusterDict[cluster].ContainsKey(label))
                    {
                        _clusterDict[cluster][label]++;
                    }
                    else
                    {
                        _clusterDict[cluster].Add(label, 1);
                    }
                }
                else
                {
                    _clusterDict.Add(cluster, new Dictionary<int, int>() { { label, 1 } });
                }

                _count++;
            }

            public double Get()
            {
                var v = 0d;
                foreach(var cdict in _clusterDict)
                {
                    var maxCount = 0;

                    foreach(var ldict in cdict.Value)
                    {
                        if(ldict.Value > maxCount)
                        {
                            maxCount = ldict.Value;
                        }
                    }

                    v += maxCount;
                }
                return v / _count;
            }
        }

        public class ClassAccuracy : IMetric
        {
            private int _correctlyClasified;
            private int _count;

            public ClassAccuracy()
            {
                _count = 0;
                _correctlyClasified = 0;
            }

            public void Add(MetricsInput input)
            {
                _correctlyClasified += input.ClassPrediction == input.ClassLabel 
                    ? 1 
                    : 0;

                _count++;
            }

            public double Get()
            {
                return _correctlyClasified / (double) _count;
            }
        }
    }
}
