﻿using System;
using BrightData.Helper;
using BrightData.UnitTests.Fixtures;
using Xunit;
using FluentAssertions;

namespace BrightData.UnitTests
{
    public partial class TensorOperationTests : UnitTestBase
    {
        public void TestAdd() 
        {
            var a = CreateRandomVector();
            var b = CreateRandomVector();
            var c = a.Add(b);
        }
    }
}