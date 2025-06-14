﻿using System.Linq;
using BrightData.Types;

namespace BrightData.Analysis.Readers
{
    /// <summary>
    /// Index analysis results
    /// </summary>
    public class IndexAnalysis
    {
        internal IndexAnalysis(MetaData metaData)
        {
            MinIndex = metaData.GetNullable<uint>(Consts.MinIndex);
            MaxIndex = metaData.GetNullable<uint>(Consts.MaxIndex);
            NumDistinct = metaData.GetNullable<uint>(Consts.NumDistinct);
            Frequency = [
                ..metaData.GetStringsWithPrefix(Consts.FrequencyPrefix)
                .Select(k => (Label: k[Consts.FrequencyPrefix.Length..], Value: metaData.GetOrThrow<double>(k)))
            ];
        }

        /// <summary>
        /// Lowest observed index
        /// </summary>
        public uint? MinIndex { get; }

        /// <summary>
        /// Highest observed index
        /// </summary>
        public uint? MaxIndex { get; }

        /// <summary>
        /// Number of distinct items
        /// </summary>
        public uint? NumDistinct { get; }

        /// <summary>
        /// Ranked histogram
        /// </summary>
        public (string Label, double value)[] Frequency { get; }
    }
}
