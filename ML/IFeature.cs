using System.Collections.Generic;

namespace ML
{
    public interface IFeature
    {
        int Length { get; }
        bool IsRedundant { get; }

        /// <summary> 
        /// Gets a subsampled representation of the feature with
        /// instances occuring certain times in the subsample.
        /// We also map the new subset indices to the old indices.
        /// </summary>
        IFeature SubsampleFeature(
            IReadOnlyList<int> occurrences,
            Dictionary<int, int> indexMapper,
            int subsampleCount);

        /// <summary> Gets a copy of the feature's values. </summary>
        float[] GetValues();
    }
}
