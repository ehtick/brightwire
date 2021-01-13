﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using FluentAssertions;
using Xunit;

namespace BrightData.UnitTests
{
    public class AnalysisTests
    {
        [Fact]
        public void DateAnalysis()
        {
            var d1 = new DateTime(2020, 1, 1);
            var d2 = new DateTime(2020, 2, 1);
            var analysis = (new[] { d1, d2, d2}).Analyze();

            analysis.MinDate.Should().Be(d1);
            analysis.MaxDate.Should().Be(d2);
            analysis.MostFrequent.Should().Be(d2.ToString(CultureInfo.InvariantCulture));
            analysis.NumDistinct.Should().Be(2);
            analysis.Total.Should().Be(3);
        }

        [Fact]
        public void DateAnalysisNoMostFrequent()
        {
            var d1 = new DateTime(2020, 1, 1);
            var d2 = new DateTime(2020, 2, 1);
            var analysis = (new[] { d1, d2 }).Analyze();

            analysis.MinDate.Should().Be(d1);
            analysis.MaxDate.Should().Be(d2);
            analysis.MostFrequent.Should().BeNull();
            analysis.NumDistinct.Should().Be(2);
            analysis.Total.Should().Be(2);
        }

        [Fact]
        public void IntegerAnalysis()
        {
            var analysis = new[] {1, 2, 3}.Analyze();
            analysis.Min.Should().Be(1);
            analysis.Max.Should().Be(3);
            analysis.Median.Should().Be(2);
            analysis.NumDistinct.Should().Be(3);
            analysis.Total.Should().Be(3);
        }

        [Fact]
        public void IntegerAnalysis2()
        {
            var analysis = new[] { 1, 2, 2, 3 }.Analyze();
            analysis.Min.Should().Be(1);
            analysis.Max.Should().Be(3);
            analysis.Median.Should().Be(2);
            analysis.NumDistinct.Should().Be(3);
            analysis.Mode.Should().Be(2);
            analysis.Total.Should().Be(4);
        }

        [Fact]
        public void StringAnalysis()
        {
            var analysis = new[] {"a", "ab", "abc"}.Analyze<string>();
            analysis.MinLength.Should().Be(1);
            analysis.MaxLength.Should().Be(3);
            analysis.NumDistinct.Should().Be(3);
            analysis.Total.Should().Be(3);
        }

        [Fact]
        public void StringAnalysis2()
        {
            var analysis = new[] { "a", "ab", "ab", "abc" }.Analyze<string>();
            analysis.MinLength.Should().Be(1);
            analysis.MaxLength.Should().Be(3);
            analysis.NumDistinct.Should().Be(3);
            analysis.MostFrequent.Should().Be("ab");
            analysis.Total.Should().Be(4);
        }
    }
}
