using System;
using System.Linq;
using ML.MathHelpers;

namespace ML
{
    /// <summary>
    /// The actual implementation of the Sparse and Dense Instances.
    /// Each implementations has particular algorithms taking advantage
    /// of the sparse and dense structure of the instances.
    /// </summary>
    public class Instances
    {
        /// <summary>
        /// The 1 * sigma value for Gaussian distribution. Used to convert the
        /// binary values{0,1} to {-1 * sigma, 1 * sigma} respectivelly
        /// in case of standardizing the instance.
        /// </summary>
        private const float GaussSigma = 1f;

        public class DenseInstance: IInstance
        {
            private float[] _values;
            private int _binaryOffset;
            public int Length => _values.Length;

            public DenseInstance(float[] values, int binaryOffset)
            {
                _values = values;
                _binaryOffset = binaryOffset;
            }

            /// <summary>
            /// Calculate L1 norm of the instance;
            /// </summary>
            public double L1Norm()
            {
                return _values.L1Norm();
            }

            public double Max()
            {
                return _values.Max();
            }

            public double Min()
            {
                return _values.Min();
            }

            public double[] MinMax()
            {
                var max = (double)float.MinValue;
                var min = (double)float.MaxValue;
                for (var j = 0; j < _values.Length; j++)
                {
                    if (max < _values[j])
                    {
                        max = _values[j];
                    }
                    if (min > _values[j])
                    {
                        min = _values[j];
                    }
                }

                return new[] {min, max};
            }

            public void Rescale(float[] min, float[] max)
            {
                for (int i = 0; i < _binaryOffset; i++)
                {
                    _values[i] = (_values[i] - min[i]) / (max[i] - min[i]);
                }
            }

            public void Standardize(float[] mean, float[] sigma)
            {
                for (var i = 0; i < _binaryOffset; i++)
                {
                    _values[i] = (_values[i] - mean[i]) / sigma[i];
                }

                // populate the binary values
                for (var i = _binaryOffset; i < _values.Length; i++)
                {
                    _values[i] = _values[i] == 0f ? - GaussSigma : GaussSigma;
                }
                _binaryOffset = _values.Length;

            }

            public float[] GetOrdinals()
            {
                var values = new float[_binaryOffset];

                Array.Copy(_values, values, values.Length);
                return values;
            }

            public double L2Dist(IInstance instance)
            {
                var dInstance = instance as DenseInstance;
                var isDense = dInstance == null ? false : true;
                var dist = 0.0;

                //TODO: implement the sparse case.
                if (isDense)
                {
                    if (dInstance._values.Length != _values.Length)
                    {
                        throw new ArgumentException("The length of both values should be the same.");
                    }

                    for (var i = 0; i < _values.Length; i++)
                    {
                        var v1 = _values[i];
                        var v2 = dInstance._values[i];
                        dist += (v1 - v2) * (double)(v1 - v2);
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }

                return Math.Sqrt(dist);
            }

            /// <summary>
            /// Calculates the euclidean distance between a row in two dimensional tensor and an instance.
            /// This method is a twin method of 'EucDist' in <see cref="ArrayHelper"/>
            /// </summary>
            /// <param name="count"> 
            /// Dimesionality of the axis we are building the distance upon.
            /// We have to be sure that 'count' doesn't exceed the dimensionality of 'tensor',
            /// because we don't access it within the method due to extra computation.</param>
            public double L2Dist(float[,] values, int index, int count)
            {
                if (count != _values.Length)
                {
                    throw new ArgumentException("The length of both values should be the same.");
                }

                var dist = 0.0;
                for (var i = 0; i < _values.Length; i++)
                {
                    var v1 = _values[i];
                    var v2 = values[index, i];
                    dist += (v1 - v2) * (double)(v1 - v2);
                }

                return Math.Sqrt(dist);
            }

            public IInstance Copy()
            {
                return new DenseInstance(GetValues(), _binaryOffset);
            }

            public float[] GetValues()
            {
                var valuesCopy = new float[_values.Length];

                Array.Copy(_values, valuesCopy, valuesCopy.Length);
                return valuesCopy;
            }

            public float GetValue(int index)
            {
                return _values[index];
            }
        }

        public class SparseInstance : IInstance
        {
            private float[] _values;
            private int[] _indices;
            private int _valuesLength { get; set; }
            private int[] _binaryIndices { get; set; }
            private int _binaryLength { get; set; }

            public int Length => _binaryLength + _valuesLength;

            public SparseInstance(
                float[] values,
                int[] indices,
                int[] binaryIndices,
                int valuesLength,
                int binaryLength)
            {
                _values = values;
                _indices = indices;
                _valuesLength = valuesLength;
                _binaryIndices = binaryIndices;
                _binaryLength = binaryLength;
            }

            /// <summary>
            /// Calculate L1 norm of the instance;
            /// </summary>
            public double L1Norm()
            {
                var norm = _values.L1Norm();
                norm += _binaryIndices.Length;

                return norm;
            }

            public double Max()
            {
                var max = _values.Max();

                if (_binaryIndices.Length > 0)
                {
                    max = Math.Max(1, max);
                }

                return max;
            }

            public double Min()
            {
                var min = _values.Min();

                if (_binaryIndices.Length > 0)
                {
                    min = Math.Min(0, min);
                }

                return min;
            }

            public double[] MinMax()
            {
                var max = (double)float.MinValue;
                var min = (double)float.MaxValue;

                for (var j = 0; j < _values.Length; j++)
                {
                    if (max < _values[j])
                    {
                        max = _values[j];
                    }
                    if (min > _values[j])
                    {
                        min = _values[j];
                    }
                }

                if (_binaryIndices.Length > 0)
                {
                    max = Math.Max(max, 1);
                }

                if (_binaryLength > _binaryIndices.Length)
                {
                    min = Math.Max(min, 0);
                }

                return new[] { min, max };
            }

            public float[] GetOrdinals()
            {
                var values = new float[_valuesLength];
                for (int i = 0; i < _indices.Length; i++)
                {
                    values[_indices[i]] = _values[i];
                }

                return values;
            }

            public double L2Dist(IInstance instance)
            {
                var sInstance = instance as SparseInstance;
                var isSparse = sInstance == null ? false : true;
                var dist = 0.0;

                // TODO: implement the dense case.
                if (isSparse)
                {
                    var ind1 = _indices;
                    var ind2 = sInstance._indices;

                    var val1 = _values;
                    var val2 = sInstance._values;

                    var k = 0;
                    var p = 0;

                    while (k < ind1.Length && p < ind2.Length)
                    {
                        if (ind1[k] < ind2[p])
                        {
                            dist += val1[k] * val1[k];
                            k++;
                        }
                        else
                        {
                            if (ind1[k] > ind2[p])
                            {
                                dist += val2[p] * val2[p];
                                p++;
                            }
                            else // ind1[k] == ind2[p]
                            {
                                dist += (val1[k] - val2[p]) * (double)(val1[k] - val2[p]);
                                k++;
                                p++;
                            }
                        }
                    }

                    var binInd1 = _binaryIndices;
                    var binInd2 = sInstance._binaryIndices;
                    k = 0;
                    p = 0;

                    var binaryDist = new bool[_binaryLength];

                    while (k < binInd1.Length && p < binInd2.Length)
                    {
                        if (binInd1[k] < binInd2[p])
                        {
                            binaryDist.SetValue(true, binInd1[k]);
                            k++;
                        }
                        else
                        {
                            if (binInd1[k] > binInd2[p])
                            {
                                binaryDist.SetValue(true, binInd2[p]);
                                p++;
                            }
                            else // binInd1[k] == binInd2[p]
                            {
                                k++;
                                p++;
                            }
                        }
                    }
                    for (var i = 0; i < binaryDist.Length; i++)
                    {
                        if (binaryDist[i])
                        {
                            dist += 1;
                        }
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
                return Math.Sqrt(dist);
            }

            /// <summary>
            /// Calculates the euclidean distance between a row in two dimensional tensor and an instance.
            /// This method is a twin method of 'EucDist' in <see cref="ArrayHelper"/>
            /// </summary>
            /// <param name="count"> 
            /// Dimesionality of the axis we are building the distance upon.
            /// We have to be sure that 'count' doesn't exceed the dimensionality of 'tensor',
            /// because we don't access it within the method due to extra computation.</param>
            public double L2Dist(float[,] values, int index, int count)
            {
                if (count != _valuesLength + _binaryLength)
                {
                    throw new ArgumentException("The length of both values should be the same.");
                }

                var dist = 0.0;
                var ind1 = _indices;

                var val1 = _values;

                var k = 0;
                var p = 0;

                while (k < ind1.Length && p < _valuesLength)
                {
                    if (ind1[k] == p)
                    {
                        dist += (val1[k] - values[index, k]) * (double)(val1[k] - values[index, k]);
                        k++;
                        p++;
                    }
                    else
                    {
                        dist += values[index, p] * values[index, p];
                        p++;
                    }
                }

                var binInd1 = _binaryIndices;
                k = 0;
                p = 0;

                while (k < binInd1.Length && p < _binaryLength)
                {
                    if (binInd1[k] == p)
                    {
                        dist += (1.0 - values[index, p + _valuesLength]) * (1.0 - values[index, p + _valuesLength]);
                        k++;
                        p++;
                    }
                    else
                    {
                        dist += values[index, p + _valuesLength] * values[index, p + _valuesLength];
                        p++;
                    }
                }

                return Math.Sqrt(dist);
            }

            public IInstance Copy()
            {
                var newValues = new float[_values.Length];
                var newIndices = new int[_indices.Length];
                var newBinaryIndices = new int[_binaryIndices.Length];
                var newValuesLength = _valuesLength;
                var newBinaryLength = _binaryLength;

                Array.Copy(_values, newValues, newValues.Length);
                Array.Copy(_indices, newIndices, newIndices.Length);
                Array.Copy(_binaryIndices, newBinaryIndices, newBinaryIndices.Length);

                return new SparseInstance(newValues, newIndices, newBinaryIndices, newValuesLength, newBinaryLength);
            }

            public float[] GetValues()
            {
                var values = new float[_valuesLength + _binaryLength];
                for (int i = 0; i < _indices.Length; i++)
                {
                    values[_indices[i]] = _values[i];
                }

                for (int i = 0; i < _binaryIndices.Length; i++)
                {
                    values[_valuesLength + _binaryIndices[i]] = 1;
                }

                return values;
            }

            public float GetValue(int index)
            {
                if (index >= _valuesLength + _binaryLength)
                {
                    throw new ArgumentException("The index exceeds the dimensions of the values.");
                }

                if (index < _valuesLength)
                {
                    for(var i = 0; i < _indices.Length; i++)
                    {
                        if(_indices[i] == index)
                        {
                            return _values[i];
                        }
                    }
                }

                if (_binaryIndices.Length > 0)
                {
                    var binaryIndex = index - _valuesLength;
                    if (binaryIndex < _binaryIndices[0])
                    {
                        return 0;
                    }

                    for (int i = 0; i < _binaryIndices.Length; i++)
                    {
                        if (binaryIndex == _binaryIndices[i])
                        {
                            return 1;
                        }
                    }
                }

                return 0;
            }

            public void Rescale(float[] min, float[] max)
            {
                for (int i = 0; i < _values.Length; i++)
                {
                    _values[i] = (_values[i] - min[_indices[i]]) / (max[_indices[i]] - min[_indices[i]]);
                }
            }

            public void Standardize(float[] mean, float[] sigma)
            {
                var n = _indices.Length + _binaryLength;
                var newValues = new float[n];
                var newIndices = new int[n];

                for (var i = 0; i < _indices.Length; i++)
                {
                    newValues[i] = (_values[i] - mean[_indices[i]]) / sigma[_indices[i]];
                    newIndices[i] = _indices[i];
                }

                // populate the zeroes
                for (var i = _indices.Length; i < _indices.Length + _binaryLength; i++)
                {
                    newValues[i] = -GaussSigma;
                    newIndices[i] = i;
                }

                // populate the ones
                for (var i = 0; i < _binaryIndices.Length; i++)
                {
                    newValues[_indices.Length + _binaryIndices[i]] = GaussSigma;
                }

                _valuesLength = Length;
                _binaryLength = 0;
                _indices = newIndices;
                _values = newValues;
                _binaryIndices = new int[0];
            }
        }
    }
}
