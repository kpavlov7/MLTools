namespace ML
{
    /// <summary>
    /// The supported feature types of the Feature and Instance Representations.
    /// </summary>
    public enum InputFeatureTypes : byte
    {
        /// <summary>
        /// Takes values in [0; float.MaxValue <see cref="float.MaxValue"/>].
        /// </summary>
        Ordinal = 1 << 0,

        /// <summary>
        /// Takes values in {0, 1}.
        /// </summary>
        Flags = 1 << 1
    }
}
