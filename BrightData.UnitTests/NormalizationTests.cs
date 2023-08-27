﻿using BrightData.UnitTests.Helper;
using System.Linq;
using BrightData.DataTable;
using BrightData.Helper;
using FluentAssertions;
using Xunit;

namespace BrightData.UnitTests
{
    public class NormalizationTests : UnitTestBase
    {
        BrightDataTable GetTable()
        {
            var builder = _context.CreateTableBuilder();
            builder.AddColumn(BrightDataType.Double);
            builder.AddColumn(BrightDataType.Double);
            builder.AddRow(100d, 200d);
            builder.AddRow(200d, 300d);
            builder.AddRow(-50d, -100d);
            return builder.BuildInMemory();
        }

        static void ValidateNormalization(BrightDataTable normalized, BrightDataTable original)
        {
            var normalization = normalized.ColumnMetaData.Select(x => x.GetNormalization()).ToArray();

            foreach(var (normalizedRow, originalRow) in normalized.AllRows.Zip(original.AllRows)) {
                for (uint i = 0; i < original.ColumnCount; i++) {
                    var originalValue = originalRow.Get<double>(i);
                    var normalizedValue = normalizedRow.Get<double>(i);
                    normalizedValue.Should().BeInRange(-1.5, 1.5);
                    var reverseNormalized = normalization[i].ReverseNormalize(normalizedValue);
                    DoubleMath.AreApproximatelyEqual(reverseNormalized, originalValue, 0.1).Should().BeTrue();
                }
            }
        }

        [Fact]
        public void EuclideanNormalization()
        {
            using var table = GetTable();
            using var normalized = table.Normalize(NormalizationType.Euclidean);
            ValidateNormalization(normalized, table);
        }

        [Fact]
        public void FeatureScaleNormalization()
        {
            using var table = GetTable();
            using var normalized = table.Normalize(NormalizationType.FeatureScale);
            ValidateNormalization(normalized, table);
        }

        [Fact]
        public void ManhattanNormalization()
        {
            using var table = GetTable();
            using var normalized = table.Normalize(NormalizationType.Manhattan);
            ValidateNormalization(normalized, table);
        }

        [Fact]
        public void StandardNormalization()
        {
            using var table = GetTable();
            using var normalized = table.Normalize(NormalizationType.Standard);
            ValidateNormalization(normalized, table);
        }
    }
}