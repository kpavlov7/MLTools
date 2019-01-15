namespace ML
{
    public interface IInstance
    {
        int Length { get; }

        /// <summary> Gets the L1 Norm of an instance. </summary>
        double L1Norm();
        /// <summary> Gets the maximum value of an instance. </summary>
        double Max();
        /// <summary> Gets the minimum value of an instance. </summary>
        double Min();
        /// <summary> Gets the minimum and maximum value of an instance. </summary>
        double[] MinMax();
        /// <summary> Rescales the ordinal values in the instance within feature. </summary>
        void Rescale(float[] min, float[] max);
        /// <summary> Standardizes the ordinal values in the instance with feature mean and sigma. </summary>
        void Standardize(float[] mean, float[] sigma);
        /// <summary> Gets the ordinal values in an instance. </summary>
        float[] GetOrdinals();
        /// <summary> Gets the L2 distance between two instances. </summary>
        double L2Dist(IInstance instance);
        /// <summary> Gets the L2 distance between an instance and a row of a matrix. </summary>
        double L2Dist(float[,] values, int index, int count);
        /// <summary> Gets a copy of the values in the instance. </summary>
        float[] GetValues();
        /// <summary> Gets the value in the instance for a given index. </summary>
        float GetValue(int index);
        /// <summary> Gets a copy of the instance. </summary>
        IInstance Copy();
    }
}