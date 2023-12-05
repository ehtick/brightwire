﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightData.UnitTests.Helper;
using FluentAssertions;
using Xunit;

namespace BrightData.UnitTests
{
    public class ClusteringTests : UnitTestBase
    {
        [Fact]
        public void Hierachical()
        {
            var clusterer = _context.NewHierachicalClustering();
            var vectors = new IReadOnlyVector[] {
                _context.CreateReadOnlyVector(0f, 0f, 0f),
                _context.CreateReadOnlyVector(0.1f, 0.1f, 0.1f),
                _context.CreateReadOnlyVector(0.8f, 0.8f, 0.8f),
            };
            var clusters = clusterer.Cluster(vectors, 2, DistanceMetric.Euclidean);
            clusters.Length.Should().Be(2);
            clusters[0].Should().BeEquivalentTo(new[] { 0U, 1U });
            clusters[1].Should().BeEquivalentTo(new[] { 2U });
        }

        [Fact]
        public void KMeans()
        {
            var clusterer = _context.NewKMeansClustering();
            var vectors = new IReadOnlyVector[] {
                _context.CreateReadOnlyVector(0f, 0f, 0f),
                _context.CreateReadOnlyVector(0.1f, 0.1f, 0.1f),
                _context.CreateReadOnlyVector(0.8f, 0.8f, 0.8f),
            };
            var clusters = clusterer.Cluster(vectors, 2, DistanceMetric.Euclidean);
            clusters.Length.Should().Be(2);
            clusters[0].Should().BeEquivalentTo(new[] { 0U, 1U });
            clusters[1].Should().BeEquivalentTo(new[] { 2U });
        }
    }
}
