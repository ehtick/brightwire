using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;

namespace BrightWire.Models
{
    /// <summary>
    /// Represents a confusion matrix for evaluating classification model performance.
    /// Maps expected classifications against actual classifications with occurrence counts.
    /// </summary>
    public class ConfusionMatrix
    {
        /// <summary>
        /// Represents an actual classification result within an expected classification group.
        /// </summary>
        public class ActualClassification
        {
            /// <summary>
            /// The index of the actual classification into the label array.
            /// </summary>
            public int ClassificationIndex { get; set; }

            /// <summary>
            /// The number of times this actual classification occurred.
            /// </summary>
            public uint Count { get; set; }
        }

        /// <summary>
        /// Represents an expected (true) classification group containing actual classification results.
        /// </summary>
        public class ExpectedClassification
        {
            /// <summary>
            /// The index of the expected classification into the label array.
            /// </summary>
            public int ClassificationIndex { get; set; }

            /// <summary>
            /// Array of actual classifications observed for this expected classification.
            /// </summary>
            public ActualClassification[] ActualClassifications { get; set; } = [];
        }
        readonly Lazy<Dictionary<string, int>> _classificationTable;

        /// <summary>
        /// Labels for each classification, indexed by classification index.
        /// </summary>
        public string[] ClassificationLabels { get; set; } = [];

        /// <summary>
        /// Array of expected classifications, each containing the actual classification breakdown.
        /// </summary>
        public ExpectedClassification[] Classifications { get; set; } = [];

        /// <summary>
        /// Constructor
        /// </summary>
        public ConfusionMatrix()
        {
            _classificationTable = new(() => ClassificationLabels
                .Select((c, i) => (c, i))
                .ToDictionary(d => d.c, d => d.i))
            ;
        }

        /// <summary>
        /// Returns the confusion matrix serialized as an XML string.
        /// </summary>
        public string AsXml
        {
            get
            {
                var ret = new StringBuilder();
                using var writer = XmlWriter.Create(new StringWriter(ret));
                writer.WriteStartElement("confusion-matrix");
                foreach (var expected in Classifications)
                {
                    writer.WriteStartElement("expected-classification");
                    writer.WriteAttributeString("label",
                        expected.ClassificationIndex >= 0 && expected.ClassificationIndex < ClassificationLabels.Length
                            ? ClassificationLabels[expected.ClassificationIndex]
                            : "unknown");
                    foreach (var actual in expected.ActualClassifications)
                    {
                        writer.WriteStartElement("actual-classification");
                        writer.WriteAttributeString("label",
                            actual.ClassificationIndex >= 0 && actual.ClassificationIndex < ClassificationLabels.Length
                                ? ClassificationLabels[actual.ClassificationIndex]
                                : "unknown");
                        writer.WriteAttributeString("count", actual.Count.ToString());
                        writer.WriteEndElement();
                    }
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
                return ret.ToString();
            }
        }

        Dictionary<string, int> ClassificationTable => _classificationTable.Value;

        /// <summary>
        /// Returns the count of a given expected versus actual classification pair.
        /// </summary>
        /// <param name="expected">The label of the expected (true) classification.</param>
        /// <param name="actual">The label of the actual (predicted) classification.</param>
        /// <returns>The count of occurrences, or zero if either label is not recognized.</returns>
        public uint GetCount(string expected, string actual)
        {
            if (!ClassificationTable.TryGetValue(expected, out int expectedIndex))
                return 0;
            if (!ClassificationTable.TryGetValue(actual, out int actualIndex))
                return 0;

            var expectedClassification = Classifications.FirstOrDefault(c => c.ClassificationIndex == expectedIndex);
            if (expectedClassification != null)
            {
                var actualClassification = expectedClassification.ActualClassifications.FirstOrDefault(c => c.ClassificationIndex == actualIndex);
                if (actualClassification != null)
                    return actualClassification.Count;
            }

            return 0;
        }
    }
}
