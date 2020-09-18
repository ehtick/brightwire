﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrightData;
using BrightData.FloatTensors;
using BrightWire;
using BrightWire.ExecutionGraph;
using BrightWire.Models;
using BrightWire.TrainingData.Artificial;
using ExampleCode.Datasets;

namespace ExampleCode
{
    static class IntegerAddition
    {
        public static void Recurrent(IBrightDataContext context)
        {
            var graph = context.CreateGraphFactory();
            var data = context.IntegerAddition();

            // binary classification rounds each output to either 0 or 1
            var errorMetric = graph.ErrorMetric.BinaryClassification;

            // configure the network properties
            graph.CurrentPropertySet
                .Use(graph.GradientDescent.Adam)
                .Use(graph.GaussianWeightInitialisation(false, 0.3f, GaussianVarianceCalibration.SquareRoot2N))
            ;

            // create the engine
            var trainingData = graph.CreateDataSource(data.Training);
            var testData = trainingData.CloneWith(data.Test);
            var engine = graph.CreateTrainingEngine(trainingData, learningRate: 0.01f, batchSize: 16);

            // build the network
            const int HIDDEN_LAYER_SIZE = 32, TRAINING_ITERATIONS = 30;
            graph.Connect(engine)
                .AddSimpleRecurrent(graph.ReluActivation(), HIDDEN_LAYER_SIZE)
                .AddFeedForward(engine.DataSource.GetOutputSizeOrThrow())
                .Add(graph.ReluActivation())
                .AddBackpropagationThroughTime(errorMetric)
            ;

            // train the network for twenty iterations, saving the model on each improvement
            ExecutionGraph bestGraph = null;
            engine.Train(TRAINING_ITERATIONS, testData, errorMetric, bn => bestGraph = bn.Graph);

            // export the graph and verify it against some unseen integers on the best model
            var executionEngine = graph.CreateEngine(bestGraph ?? engine.Graph);
            var testData2 = graph.CreateDataSource(BinaryIntegers.Addition(context, 8));
            var results = executionEngine.Execute(testData2).ToArray();

            // group the output
            var groupedResults = new (Vector<float>[] Input, Vector<float>[] Target, Vector<float>[] Output)[8];
            for (var i = 0; i < 8; i++) {
                var input = new Vector<float>[32];
                var target = new Vector<float>[32];
                var output = new Vector<float>[32];
                for (var j = 0; j < 32; j++) {
                    input[j] = results[j].Input[0][i];
                    target[j] = results[j].Target[i];
                    output[j] = results[j].Output[i];
                }
                groupedResults[i] = (input, target, output);
            }

            // write the results
            foreach (var result in groupedResults) {
                Console.Write("First:     ");
                foreach (var item in result.Input)
                    _WriteAsBinary(item[0]);
                Console.WriteLine();

                Console.Write("Second:    ");
                foreach (var item in result.Input)
                    _WriteAsBinary(item[1]);
                Console.WriteLine();
                Console.WriteLine("           --------------------------------");

                Console.Write("Expected:  ");
                foreach (var item in result.Target)
                    _WriteAsBinary(item[0]);
                Console.WriteLine();

                Console.Write("Predicted: ");
                foreach (var item in result.Output)
                    _WriteAsBinary(item[0]);
                Console.WriteLine();
                Console.WriteLine();
            }
        }

        static void _WriteAsBinary(float value)
        {
            if (value >= 0.5)
                Console.Write("1");
            else
                Console.Write("0");
        }
    }
}
