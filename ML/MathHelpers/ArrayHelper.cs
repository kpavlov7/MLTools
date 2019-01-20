using System.Collections.Generic;
using System.Linq;
using System;

namespace ML.MathHelpers
{
    public static class ArrayHelper
    {
        #region Arrays Operations

        public static float[] PairOpperation<T>(this IReadOnlyList<T> array, Func<T, T, double> f)
        {
            var count = array.Count;
            var r = new float[count * (count - 1) / 2];
            int k = 0;
            for(var i = 0; i < count; i++)
            {
                for(var j = i + 1; j < count; j++)
                {
                    r[k] = (float)f(array[i], array[j]);
                    k++;
                }
            }

            return r;
        }

        public static float[] AdjacentElementOpperation<T>(this IReadOnlyList<T> array, Func<T, T, double> f)
        {
            var count = array.Count;
            var r = new float[count - 1];
            for (var i = 0; i < count - 1; i++)
            {
                r[i] = (float)f(array[i], array[i + 1]);
            }

            return r;
        }

        public static int MaxArg<T>(this IReadOnlyList<T> array)
            where T : IComparable<T>
        {
            var count = array.Count;
            if (count < 1)
                throw new ArgumentException("The length of the array should be at least 1.");

            var index = 0;
            T max = array[0];

            for (var i = 1; i < count; i++)
            {
                if(array[i].CompareTo(max) > 0)
                {
                    index = i;
                    max = array[i];
                }
            }

            return index;
        }

        public static int MinArg<T>(this IReadOnlyList<T>  array)
            where T : IComparable<T>
        {
            var count = array.Count;
            if (count < 1)
                throw new ArgumentException("The length of the array should be at least 1.");

            var index = 0;
            T min = array[0];

            for (var i = 1; i < count; i++)
            {
                if (array[i].CompareTo(min) < 0)
                {
                    index = i;
                    min = array[i];
                }
            }

            return index;
        }

        public static T Max<T>(this IReadOnlyList<T> array) 
            where T : IComparable<T>
        {
            var count = array.Count;
            if (count < 1)
                throw new ArgumentException("The length of the array should be at least 1.");

            T max = array[0];

            for (int i = 1; i < count; i++)
            {
                if (array[i].CompareTo(max) > 0)
                {
                    max = array[i];
                }
            }

            return max;
        }

        public static T Min<T>(this IReadOnlyList<T> array)
            where T : IComparable<T>
        {
            var count = array.Count;
            if (count < 1)
                throw new ArgumentException("The length of the array should be at least 1.");

            T min = array[0];

            for (int i = 1; i < count; i++)
            {
                if (array[i].CompareTo(min) < 0)
                {
                    min = array[i];
                }
            }

            return min;
        }

        /// <summary>  Populates the array with the same value. </summary>
        public static void Repeat<T>(this T[] array, T value)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = value;
            }
        }
        /// <summary>
        /// Populates the array with the same value.
        /// With start index and count of the elements to be populated.
        /// </summary>
        public static void Repeat<T>(this T[] array, T value, int start, int count)
        {
            if (count > array.Length - start)
            {
                throw new ArgumentException("The segment length is greater than the array length.", nameof(array));
            }
            for (int i = start; i < start + count; i++)
            {
                array[i] = value;
            }
        }

        /// <summary>
        /// Calculate L1 norm of an array;
        /// </summary>
        public static double L1Norm(this IReadOnlyList<float> array)
        {
            var count = array.Count;
            if (count == 0)
            {
                throw new ArgumentException("Array length should be greater than 0.");
            }

            var norm = 0.0;
            var n = array.Count;

            for (var i = 0; i < count; i++)
            {
                norm += Math.Abs(array[i]);
            }

            return norm;
        }

        public static double L2Norm(this IReadOnlyList<float> array)
        {

            var count = array.Count;
            if (count == 0)
            {
                throw new ArgumentException("Array length should be greater than 0.");
            }

            var norm = 0.0;

            for (var i = 0; i < count; i++)
            {
                norm += array[i] * array[i];
            }

            return Math.Sqrt(norm);
        }

        public static T GetLast<T>(this IReadOnlyList<T> array)
        {
            var count = array.Count;
            if (count == 0)
            {
                throw new ArgumentException("Array length should be greater than 0.");
            }

            return array[count - 1];
        }

        public static bool IsHomogenious <T>(this IList<T> array, int scopeStart, int scopeCount)
        {
            var count = array.Count;

            if(scopeStart < 0 || scopeStart + scopeCount > count)
            {
                throw new ArgumentOutOfRangeException(
                    "Specified start or end are exceeding the boundries of the array.");
            }

            var scopeLimit = scopeStart + scopeCount - 1;

            for (var i = scopeStart; i < scopeLimit; i++)
            {
                if (!array[i].Equals(array[i + 1]))
                {
                    return false;
                }
            }

            return true;
        }
        /// <summary>
        /// https://stackoverflow.com/questions/273313/randomize-a-listt
        /// </summary>
        public static void Shuffle<T>(this IList<T> array, Random random)
        {
            var n = array.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = array[k];
                array[k] = array[n];
                array[n] = value;
            }
        }

        public static T[] SampleNoReplacement<T>(this T[] array, int count, Random random)
        {
            Shuffle(array, random);

            if (count >= array.Length)
            {
                return array;
            }

            return array.Take(count).ToArray();
        }

        public static T[] SampleReplacement<T>(this T[] array, int count, Random random)
        {
            var copy = new T[count];
            
            for(var i = 0; i < count; i++)
            {
                copy[i] = array[random.Next(array.Length)];
            }

            return copy;
        }

        /// <summary>
        /// Inserts value within sorted segment of an array. The sorted
        /// part of the array is in the end of the array i.e. for 'sortedLength' = 3
        /// and array.Length = 5: [n, n, s, s, s]
        /// </summary>
        /// <param name="sortedLength"> Length of the not sorted array segment; </param>
        public static void SortedInsert(this double[] array, int sortedLength, double value)
        {
            if (array.Length < 1)
            {
                throw new ArgumentException("The array should contain at least 1 element.", nameof(array));
            }

            if (sortedLength >= array.Length)
            {
                throw new ArgumentException(
                    "The sorted segment should at least with one value shorter than the actuall array.",
                    nameof(array));
            }
            var notSortedLength = array.Length - sortedLength;
            var insertAt = Array.BinarySearch(array, notSortedLength, sortedLength, value);

            if (insertAt < 0) // If the value doesn't exist in the sorted segment;
            {
                insertAt = ~insertAt;

                if (insertAt == -1) // If the value should be the first element;
                    // We place it at the last place of the not sorted segment.
                    insertAt = notSortedLength + insertAt;
                else // If the value is not the first element;
                {
                    // Replace position with the first smaller element;
                    insertAt = insertAt - 1;

                    var x = array[insertAt];

                    var stopAt = Math.Max(notSortedLength - 1, 1);

                    // Shift all smaller values one position towards the not sorted segment.
                    for (var i = insertAt; i >= stopAt; i--)
                    {
                        var y = array[i - 1];
                        array[i - 1] = x;
                        x = y;
                    }
                    // Place the value at the proper position;
                    array[insertAt] = value;
                }
            }
            else // If the value exists in the the sorted segment;
            {
                var x = array[insertAt];
                var stopAt = Math.Max(notSortedLength - 1, 1);

                // Shift all smaller values one position towards the not sorted segment.
                for (var i = insertAt; i >= stopAt; i--)
                {
                    var y = array[i - 1];

                    array[i - 1] = x;
                    x = y;
                }
                // Place the value at the proper position;
                array[insertAt] = value;
            }
        }

        public static double Variance(this IReadOnlyList<float> array)
        {
            var count = array.Count;

            if (count < 2)
            {
                throw new ArgumentException(
                    "The array should have at least two elements for the variance.",
                    nameof(array));
            }

            var sSum = 0.0;
            var sum = 0.0;

            for (var i = 0; i < count; i++)
            {
                sum += array[i];
                sSum += array[i] * array[i];
            }

            var mean = sum / count;

            return (sSum / count) - (mean * mean);
        }

        public static double Mean(this IReadOnlyList<float> array)
        {
            var count = array.Count;

            if (count < 1)
            {
                throw new ArgumentException(
                    "The array should have at least 1 element for the mean.",
                    nameof(array));
            }

            var sum = 0.0;

            for (var i = 0; i < count; i++)
            {
                sum += array[i];
            }

            return sum / count;
        }

        public static double Sum(this IReadOnlyList<float> array)
        {
            var count = array.Count;

            var sum = 0.0;

            for (var i = 0; i < count; i++)
            {
                sum += array[i];
            }

            return sum;
        }

        public static T LastElement<T>(this IReadOnlyList<T> array)
        {
            var count = array.Count;

            if (count < 1)
            {
                throw new ArgumentException(
                    "The array should have at least 1 element to return.",
                    nameof(array));
            }

            return array[count-1];
        }
        #endregion

        #region Matrix Flat Arithmetical Operations
        /// <summary>
        /// Calculates the euclidean distance between a row in two dimensional matrix and an array.
        /// </summary>
        /// <param name="count"> 
        /// Dimesionality of the axis we are building the distance upon.
        /// We have to be sure that 'count' doesn't exceed the dimensionality of 'matrix',
        /// because we don't access it within the method due to extra computation.</param>
        public static double L2Dist(
            this float[,] matrix, IReadOnlyList<float> array, int index, int count)
        {
            if (count != array.Count)
            {
                throw new ArgumentException(
                    "Array and matrix should have the same dimensionality along axis we are iterating over.");
            }
            var dist = 0.0;
            float delta;

            for (var i = 0; i < count; i++)
            {
                delta = matrix[index, i] - array[i];
                dist += delta * delta;
            }

            return Math.Sqrt(dist);
        }

        /// <summary>
        /// Calculates the euclidean distance between two rows in two dimensional matrix.
        /// </summary>
        /// <param name="count"> 
        /// Dimesionality of the axis we are building the distance upon.
        /// We have to be sure that 'count' doesn't exceed the dimensionality of 'matrix',
        /// because we don't access it within the method due to extra computation.</param>
        /// <returns></returns>
        public static double L2Dist(this float[,] matrix, int index1, int index2, int count)
        {
            var dist = 0.0;
            float delta;

            for (var i = 0; i < count; i++)
            {
                delta = matrix[index1, i] - matrix[index2, i];
                dist += delta * delta;
            }

            return Math.Sqrt(dist);
        }

        public static float[] Plus(this float[,] matrix, IReadOnlyList<float> array, int index, int count)
        {

            if (count != array.Count)
            {
                throw new ArgumentException(
                    "Array and matrix should have the same dimensionality along axis we are iterating over.");
            }

            var output = new float[count];

            for (var i = 0; i < count; i++)
            {
                output[i] = matrix[index, i] + array[i];
            }

            return output;
        }

        public static float[] ReduceSum(this float[,] matrix, int[] dim, int keptAxis = 0)
        {
            var reducedAxis = keptAxis == 0 ? 1 : 0;

            if (dim[0] <= 0 || dim[1] <= 0 || dim.Length != 2)
            {
                throw new ArgumentException(
                    "The axis lengths should be greater than 0.");
            }

            var output = new float[dim[keptAxis]];

            if (keptAxis == 0)
            {
                for (var i = 0; i < output.Length; i++)
                {
                    for (int j = 0; j < dim[reducedAxis]; j++)
                    {
                        output[i] = matrix[i, j];
                    }
                }
            }
            if (keptAxis == 1)
            {
                for (var i = 0; i < output.Length; i++)
                {
                    for (int j = 0; j < dim[reducedAxis]; j++)
                    {
                        output[i] = matrix[j, i];
                    }
                }
            }

            return output;
        }

        /// <summary>
        /// Calculate L1 norm of a 2D matrix in a ginven row;
        /// </summary>
        /// <param name="count"> 
        /// Dimesionality of the axis we are building the distance upon.
        /// We have to be sure that 'count' doesn't exceed the dimensionality of 'matrix',
        /// because we don't access it within the method due to extra computation.</param>
        public static double L1Norm(this float[,] matrix, int index, int count)
        {

            if (count == 0)
            {
                throw new ArgumentException("Array length should be greater than 0.");
            }

            var norm = 0.0;

            for (var i = 0; i < count; i++)
            {
                norm += Math.Abs(matrix[index, i]);
            }

            return norm;
        }

        /// <summary>
        /// Calculate the max of a 2D matrix in a given row;
        /// </summary>
        /// <param name="count"> 
        /// Dimesionality of the axis we are building the distance upon.
        /// We have to be sure that 'count' doesn't exceed the dimensionality of 'matrix',
        /// because we don't access it within the method due to extra computation.</param>
        public static double Max(this float[,] matrix, int index, int count)
        {

            if (count == 0)
            {
                throw new ArgumentException("Tensor second axis length should be greater than 0.");
            }

            var max = 0.0;

            for (var i = 0; i < count; i++)
            {
                max = max < matrix[index, i] ? matrix[index, i] : max;
            }

            return max;
        }

        /// <summary>
        /// Fills at certain row(index) of the matrix specified matrixs's span of between 'matrixOffset'
        /// and 'matrixOffset + count' with the values of the analogical span from an array.
        /// </summary>
        public static void FillTensor(
            this float[,] matrix,
            int index,
            int matrixOffset,
            int arrayOffset,
            int count,
            float[] array)
        {
            for (var i = 0; i< count; i++)
            {
                matrix[index, i + matrixOffset] = array[i + arrayOffset];
            }
        }
        #endregion
    }
}
